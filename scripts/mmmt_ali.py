#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import multiprocessing
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path
from string import Template

PROJECT_DIR = Path(__file__).parent.parent
EXP_DIR = PROJECT_DIR / "logs" / "mmmt_ali"
EXP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ARGS = {
    "text_image": "seperated",
    "text_text": "cross",
    "use_text": True,
    "use_image": True,
    "model_name": "bert-base-chinese",
    "feature_type": "grid",
    "num_image_embeds": 1,
    "cate_level_name": "null",
    "cate_name": "null",
    "prod_num": 200,
    "seed": 123,
}
EXPT_TMP = Template(
    """{
  "model": {
    "class_path": "src.MultimodalMatcher",
    "init_args": {
      "text_image": "${text_image}",
      "text_text": "${text_text}",
      "use_text": "${use_text}",
      "use_image": "${use_image}",
      "model_name": "${model_name}",
      "feature_type": "${feature_type}",
      "num_image_embeds": "${num_image_embeds}"
    }
  },
  "data": {
    "class_path": "src.AliDataModule",
    "init_args": {
      "cate_level_name": "${cate_level_name}",
      "cate_name": "${cate_name}",
      "prod_num": "${prod_num}"
    }
  },
  "seed": "${seed}"
}"""
)
EXPTS = []
SEEDS = [123, 42, 1337]

for prod_num in [200, 400, 800]:
    for cate_level_name in ["null", "男装"]:
        for cate_name in ["null"]:
            for seed in SEEDS:
                kwargs_list = [
                    {
                        "text_image": "aligned",
                        "text_text": "cross",
                        "use_text": True,
                        "use_image": False,
                        "feature_type": "grid",
                        "num_image_embeds": 1,
                    },
                    {
                        "text_image": "aligned",
                        "text_text": "cross",
                        "use_text": True,
                        "use_image": True,
                        "feature_type": "grid",
                        "num_image_embeds": 1,
                    },
                    {
                        "text_image": "aligned",
                        "text_text": "cross",
                        "use_text": True,
                        "use_image": True,
                        "feature_type": "roi",
                        "num_image_embeds": 9,
                    },
                    # {
                    #     "text_image": "seperated",
                    #     "text_text": "cross",
                    #     "use_text": True,
                    #     "use_image": False,
                    #     "feature_type": "grid",
                    #     "num_image_embeds": 1,
                    # },
                    {
                        "text_image": "seperated",
                        "text_text": "cross",
                        "use_text": False,
                        "use_image": True,
                        "feature_type": "grid",
                        "num_image_embeds": 1,
                    },
                    {
                        "text_image": "seperated",
                        "text_text": "cross",
                        "use_text": False,
                        "use_image": True,
                        "feature_type": "roi",
                        "num_image_embeds": 9,
                    },
                    {
                        "text_image": "seperated",
                        "text_text": "cross",
                        "use_text": True,
                        "use_image": True,
                        "feature_type": "grid",
                        "num_image_embeds": 1,
                    },
                    {
                        "text_image": "seperated",
                        "text_text": "cross",
                        "use_text": True,
                        "use_image": True,
                        "feature_type": "roi",
                        "num_image_embeds": 9,
                    },
                ]

                for kwarg in kwargs_list:
                    kwarg["cate_level_name"] = cate_level_name
                    kwarg["cate_name"] = cate_name
                    kwarg["prod_num"] = prod_num
                    kwarg["seed"] = seed
                    EXPTS.append(EXPT_TMP.substitute(DEFAULT_ARGS, **kwarg))

for prod_num in [200, 400, 800]:
    for cate_level_name in ["null", "男装"]:
        for cate_name in ["null"]:
            for seed in SEEDS:
                kwargs_list = [
                    {
                        "text_image": "aligned",
                        "text_text": "dual",
                        "use_text": True,
                        "use_image": False,
                        "feature_type": "grid",
                        "num_image_embeds": 1,
                    },
                    {
                        "text_image": "aligned",
                        "text_text": "dual",
                        "use_text": True,
                        "use_image": True,
                        "feature_type": "grid",
                        "num_image_embeds": 1,
                    },
                    {
                        "text_image": "aligned",
                        "text_text": "dual",
                        "use_text": True,
                        "use_image": True,
                        "feature_type": "roi",
                        "num_image_embeds": 9,
                    },
                    # {
                    #     "text_image": "seperated",
                    #     "text_text": "dual",
                    #     "use_text": True,
                    #     "use_image": False,
                    #     "feature_type": "grid",
                    #     "num_image_embeds": 1,
                    # },
                    {
                        "text_image": "seperated",
                        "text_text": "dual",
                        "use_text": True,
                        "use_image": True,
                        "feature_type": "grid",
                        "num_image_embeds": 1,
                    },
                    {
                        "text_image": "seperated",
                        "text_text": "dual",
                        "use_text": True,
                        "use_image": True,
                        "feature_type": "roi",
                        "num_image_embeds": 9,
                    },
                ]

                for kwarg in kwargs_list:
                    kwarg["cate_level_name"] = cate_level_name
                    kwarg["cate_name"] = cate_name
                    kwarg["prod_num"] = prod_num
                    kwarg["seed"] = seed
                    EXPTS.append(EXPT_TMP.substitute(DEFAULT_ARGS, **kwarg))


os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["all_proxy"] = "socks5://127.0.0.1:7890"
os.environ["no_proxy"] = "localhost,127.0.0.0/8,*.local"


def argument_parser():
    parser = argparse.ArgumentParser(description="run experiments in parallel")
    parser.add_argument(
        "--fast-dev-run",
        nargs="?",
        type=int,
        default=None,
        const=5,
        help="numbers of fast dev run",
    )
    parser.add_argument(
        "--num-expt", type=int, default=1, help="how many experiments per gpu"
    )
    parser.add_argument(
        "--no-run", action="store_true", help="whether to not run command"
    )
    parser.add_argument("--gpus", nargs="+", default=["0"], help="availabled gpus")

    return parser


def run(exp_args, args):
    worker_id = int(multiprocessing.current_process().name.rsplit("-", 1)[1]) - 1
    gpu = args.gpus[worker_id % len(args.gpus)]

    exp_name = "_".join(map(str, exp_args["model"]["init_args"].values()))
    exp_name += "_" + "_".join(map(str, exp_args["data"]["init_args"].values()))

    outfile = EXP_DIR / f"{exp_name}_{exp_args['seed']}_out.log"
    errfile = EXP_DIR / f"{exp_name}_{exp_args['seed']}_err.log"

    if args.fast_dev_run:
        cmd = f"""./run.py \\
        --seed_everything {exp_args['seed']} \\
        --trainer.gpus {gpu}, --trainer.fast_dev_run {args.fast_dev_run} \\
        --model '{exp_args['model']}' \\
        --data '{exp_args['data']}'"""
    else:
        cmd = f"""./run.py \\
        --seed_everything {exp_args['seed']} \\
        --trainer.gpus {gpu}, \\
        --model '{exp_args['model']}' \\
        --data '{exp_args['data']}' \\
        >{outfile} 2>{errfile}
        """

    print(cmd)

    if not args.no_run:
        subprocess.call(cmd, shell=True)
        print(f"{exp_name} finished")
        if not args.fast_dev_run:
            (EXP_DIR / exp_name).touch()


if __name__ == "__main__":
    args = argument_parser().parse_args()
    pool = Pool(processes=len(args.gpus) * args.num_expt)

    for expt in EXPTS:
        pool.apply_async(
            run, kwds={"exp_args": json.loads(expt, strict=False), "args": args,},
        )

    pool.close()
    pool.join()
