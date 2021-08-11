#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import json
import logging
import os
import re
import shutil
import sys
import zlib
from collections import Counter, OrderedDict
from difflib import SequenceMatcher
from logging.config import dictConfig
from pathlib import Path
from typing import List
from urllib.parse import quote, urlparse

import lxml.html
import pandas as pd
import requests
import urllib3
import waybackpy
from fake_headers import Headers
from PIL import Image
from pytorch_lightning import seed_everything
from tenacity import retry
from tenacity.retry import retry_if_exception
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_random_exponential

urllib3.disable_warnings()

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / Path("../../data/wdc")

# NOTE: LOGLEVEL should not be 0(NOTSET)
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "verbose": {"format": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"},
        "simple": {"format": "%(levelname)-8s %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "logs/debug.log",
            "mode": "w",
            "formatter": "verbose",
            "level": int(os.getenv("LOGLEVEL", logging.DEBUG)),
        },
    },
    "loggers": {
        __name__: {
            "level": int(os.getenv("LOGLEVEL", logging.DEBUG)),
            "handlers": ["file"],
        }
    },
    "root": {"level": logging.INFO, "handlers": ["console"]},
}


def parse_resource_url(url, res_url):
    if "?file=" in res_url:
        res_url = "/" + res_url.split("?file=")[1]
        res_url = res_url.split("&")[0]

    parsed_url = urlparse(url)
    if res_url.startswith("//"):
        res_url = f"{parsed_url.scheme}:{res_url}"
    elif res_url.startswith("/"):
        res_url = f"{parsed_url.scheme}://{parsed_url.netloc}{res_url}"

    parsed_res_url = urlparse(res_url)

    return f"{parsed_res_url.scheme or parsed_url.scheme}://{parsed_res_url.netloc}{quote(parsed_res_url.path)}"


def filter_images(img_url):
    filter_words = [
        "icon",
        "logo",
        "placeholder",
        "banner",
        "flag",
        "button",
        "qrcode",
        "stern_voll",
        "mark",
    ]
    return img_url and not any(w in img_url.lower() for w in filter_words)


def filter_product_images(img_url):
    filter_words = ["main", "product"]
    return img_url and any(w in img_url.lower() for w in filter_words)


def get_url_from_archive(url, year=2017, month=11):
    user_agent = (
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0"
    )
    try:
        wayback = waybackpy.Url(url, user_agent)
        archive = wayback.near(year=year, month=month)
        archive_url = archive.archive_url
        archive_url = archive_url[:42] + "if_" + archive_url[42:]

        return archive_url
    except:
        return None


def is_image(filename: str) -> bool:
    try:
        with Image.open(filename) as img:
            img.verify()
            return True
    except:
        return False


def download_image(
    url: str, filename: str, header: Headers = Headers(headers=True),
) -> None:
    r = requests.get(
        url, stream=True, headers=header.generate(), timeout=10, verify=False
    )
    r.raise_for_status()

    if r.status_code == 200:
        r.raw.decode_content = True
        with open(filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)

        if not is_image(filename):
            os.remove(filename)

            raise Exception(f"Invalid image: {filename}")


def retried_response(exception):
    return str(exception).startswith("Invalid image: ") or (
        exception.response and exception.response.status_code in [403, 422, 429]
    )


@retry(
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(),
    retry=retry_if_exception(retried_response),
    reraise=True,
)
def download_image_with_retries(
    url: str, filename: str, header: Headers = Headers(headers=True)
) -> None:
    download_image(url, filename, header)


class ImageCrawler:
    def __init__(self) -> None:
        # self._indexer_url = "http://index.commoncrawl.org/CC-MAIN-2017-47-index"
        self._indexer_url = "http://127.0.0.1:8080/CC-MAIN-2017-47-index"
        self._webpage_url = "http://commoncrawl.s3.amazonaws.com/"

        self._id_url_mapping = pd.read_csv(
            DATA_DIR / "id_url_mapping.csv.gz", index_col="id"
        )
        self._images_dir = DATA_DIR / "images"
        self._images_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()

        self._logger = logging.getLogger(__name__)
        self._logger.info("log level: %d", self._logger.level)

        self._header = Headers(headers=True)

    def get_html(self, url: str) -> str:
        r = self._session.get(self._indexer_url, params={"url": url, "output": "json"})
        record = list(
            filter(
                lambda x: x.get("status") == "200",
                map(lambda x: json.loads(x), filter(None, r.text.split("\n"))),
            )
        )[0]
        page_url = self._webpage_url + record["filename"]
        start_byte = int(record["offset"])
        end_byte = start_byte + int(record["length"])
        headers = {"Range": f"bytes={start_byte}-{end_byte}"}

        r = self._session.get(page_url, headers=headers)
        data = zlib.decompress(r.content, wbits=zlib.MAX_WBITS | 16)
        data = data.decode(errors="ignore")

        blank_line_regex = r"(?:\r?\n){2,}"
        html = re.split(blank_line_regex, data, maxsplit=2)[2]

        return html

    def download_image(self, url: str, filename: str) -> None:
        if os.path.isfile(filename):
            return

        try:
            download_image_with_retries(url, filename, self._header)
        except Exception as e:
            self._logger.warning(e)

            archive_url = get_url_from_archive(url)
            if archive_url:
                self._logger.info(f"archive url: {archive_url}")

                try:
                    download_image(archive_url, filename, self._header)
                except Exception as e:
                    self._logger.warning(e)

    def get_image(self, pid: int, ptitle: str = "") -> List[str]:
        url = str(self._id_url_mapping.loc[pid].item())
        self._logger.info(f"pid: {pid} url: {url} ptitle: {ptitle}")

        html = self.get_html(url)
        self._logger.log(1, html)
        tree = lxml.html.fromstring(bytes(html, encoding="utf8"))

        try:
            title = tree.xpath("//title/text()")[0].strip()
        except:
            title = ""
        try:
            mtitle = tree.xpath("//meta[@property='og:title']/@content")[0].strip()
        except:
            mtitle = ""
        try:
            keyword = tree.xpath("//meta[@name='keywords']/@content")[0].strip()
        except:
            keyword = ""

        self._logger.debug(f"title: {title}")
        self._logger.debug(f"mtitle: {mtitle}")
        self._logger.debug(f"keyword: {keyword}")

        product_imgs = tree.xpath("//meta[@property='og:image']/@content")
        product_imgs.extend(tree.xpath("//meta[@property='twitter:image']/@content"))
        self._logger.debug(f"meta imgs: {product_imgs}")

        candidates = []
        for el in tree.xpath("//img"):
            img_text = []

            self._logger.debug(el.get("src", ""))

            node = el
            for _ in range(2):
                for attr in ["title", "alt"]:
                    text = node.get(attr, "")
                    if text:
                        img_text.append(text)
                node = node.getparent()

                if node is None:
                    break

            if img_text:
                self._logger.debug(img_text)

                scores = []
                for text in [title, mtitle, keyword]:
                    if text and text != "null":
                        scores.append(
                            max(
                                [
                                    SequenceMatcher(None, text, t).ratio()
                                    for t in img_text
                                ]
                            )
                        )

                if scores:
                    max_score = max(scores)
                    min_score = min(scores)
                    self._logger.debug(f"max score {max_score} min score {min_score}")

                    src = el.get("src", "")
                    if src and img_text and max_score > 0.5:
                        self._logger.debug(img_text)
                        candidates.append((src, max_score))

                    else:  # fallback to product title
                        scores = []
                        if ptitle:
                            scores.append(
                                max(
                                    [
                                        SequenceMatcher(None, ptitle, t).ratio()
                                        for t in img_text
                                    ]
                                )
                            )

                        if scores:
                            max_score = max(scores)
                            min_score = min(scores)
                            self._logger.debug(
                                f"ptitle max score {max_score} min score {min_score}"
                            )

                            src = el.get("src", "")
                            if src and img_text and max_score > 0.8:
                                self._logger.debug(img_text)
                                candidates.append((src, max_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        product_imgs.extend(map(lambda x: x[0], candidates))

        self._logger.debug(f"annot imgs: {product_imgs}")
        if len(product_imgs):
            product_imgs = list(filter(filter_images, product_imgs))
        else:
            # regex = r"background\-image:\s*url\s*\((?:'|\"|)(.*?)(?:'|\"|)\)"
            regex = r"(?:https?:)?/+(?:[/|.|\(|\)|\w|\s|-])*?\.(?:jpg|png|gif)"
            images = re.findall(regex, html)

            self._logger.debug(f"regex imgs: {images}")
            images = list(filter(filter_images, images))

            product_imgs = list(filter(filter_product_images, images))
            product_imgs.extend([u for u, c in Counter(images).items() if c >= 4])

            self._logger.debug(f"regex product imgs: {product_imgs}")

            if len(product_imgs) > 5:
                product_imgs = []

        product_imgs = list(
            OrderedDict.fromkeys(
                map(lambda x: parse_resource_url(url, x), product_imgs)
            )
        )
        self._logger.info(product_imgs)

        for i, url in enumerate(product_imgs[:5]):
            img_path = urlparse(url).path
            suffix = Path(img_path).suffix
            filename = self._images_dir / f"{pid}_{i}{suffix}"
            self.download_image(url, str(filename))

        return product_imgs[:5]


def check_images():
    ids = set(map(lambda x: int(x.stem[:-2]), (DATA_DIR / "images").glob("*")))
    categories = ["all", "cameras", "computers", "shoes", "watches"]
    for cate in categories:
        print(cate)
        training_sets = DATA_DIR / "nonnorm" / "training-sets" / f"{cate}_train"
        gold_standards = DATA_DIR / "nonnorm" / "gold-standards" / f"{cate}_gs.json.gz"

        for f in itertools.chain(sorted(training_sets.rglob("*.json.gz")), [gold_standards]):
            cnts = Counter()
            df = pd.read_json(f, lines=True)
            total = len(df)
            cnt = df.apply(
                lambda x: int(x["id_right"] in ids) + int(x["id_left"] in ids), axis=1
            )
            print(f"\t{f.stem[:-5]:>{25}}\t{total}\t{cnt.value_counts().to_json()}\t\t{(cnt.value_counts() / total).to_json()}")


def main():
    seed_everything(123)

    categories = ["all", "cameras", "computers", "shoes", "watches"]
    dfs = []
    for cate in categories:
        training_sets = DATA_DIR / "nonnorm" / "training-sets" / f"{cate}_train"
        gold_standards = DATA_DIR / "nonnorm" / "gold-standards" / f"{cate}_gs.json.gz"

        for f in itertools.chain(training_sets.rglob("*.json.gz"), [gold_standards]):
            pair_df = pd.read_json(f, lines=True)
            for suffix in ["left", "right"]:
                dfs.append(
                    pair_df[[f"id_{suffix}", f"title_{suffix}"]].rename(
                        columns={f"id_{suffix}": "id", f"title_{suffix}": "title"},
                    )
                )

    df = pd.concat(dfs).drop_duplicates().sample(frac=1).reset_index(drop=True)
    print(df)

    if len(sys.argv) == 1:
        dictConfig(LOGGING_CONFIG)
        logger = logging.getLogger(__name__)
        image_crawler = ImageCrawler()

        for row in df.itertuples():
            logger.info(row.Index)
            try:
                image_crawler.get_image(row.id, row.title)
            except Exception as e:
                logger.error(e)

            logger.info("")
    else:
        LOGGING_CONFIG["handlers"]["file"]["filename"] = "logs/tmp.log"
        dictConfig(LOGGING_CONFIG)
        image_crawler = ImageCrawler()

        for pid in map(int, sys.argv[1:]):
            ptitle = df[df["id"] == pid]["title"].item()
            image_crawler.get_image(pid, ptitle)


if __name__ == "__main__":
    main()
