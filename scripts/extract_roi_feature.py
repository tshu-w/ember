#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import warnings
from functools import partial
from pathlib import Path

import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_INSTANCE = 36


def extract_feature_argument_parser():
    parser = argparse.ArgumentParser(description="ROI feature extraction")
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="config file",
        default="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
        # "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    )
    parser.add_argument("--input", metavar="DIR", help="image directory", required=True)
    parser.add_argument("--output", metavar="DIR", help="result directory")
    parser.add_argument("--opts", help="extraction options", nargs="+", default=[])

    return parser


def collate(batch, aug):
    for x in batch:
        image = x["image"]
        im = aug.get_transform(image).apply_image(image)
        im = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))

        x["image"] = im

    return batch


class DetectionImageDataset(Dataset):
    def __init__(self, root, cfg):
        self.image_list = list(Path(root).rglob("*"))
        self.cfg = cfg

    def __getitem__(self, index):
        image_path = str(self.image_list[index])
        # image = cv2.imread(filename)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = Image.open(image_path).convert("RGB")
            image = np.asarray(image)[:, :, ::-1]

        if self.cfg.INPUT.FORMAT == "RGB":
            image = image.shape[:, :, ::-1]

        height, width = image.shape[:2]

        return {
            "image": image,
            "height": height,
            "width": width,
            "meta": {"filename": Path(image_path)},
        }

    def __len__(self):
        return len(self.image_list)


def get_config(args):
    cfg = get_cfg()
    cfg.OUTPUT_DIR = args.output

    cfg.merge_from_file(model_zoo.get_config_file(args.config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.config)

    cfg.merge_from_list(args.opts)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def get_model(cfg):
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    model.eval()

    return model


def extract_roi_features(args, cfg, model):
    dataset = DetectionImageDataset(args.input, cfg)
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST,
    )
    collate_fn = partial(collate, aug=aug)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    for batch in tqdm(dataloader):
        with torch.no_grad():
            images = model.preprocess_image(batch)
            features = model.backbone(images.tensor)
            proposals, _ = model.proposal_generator(images, features, None)
            instances, _ = model.roi_heads(images, features, proposals, None)

            in_features = [features[f] for f in model.roi_heads.in_features]
            roi_features = model.roi_heads.box_pooler(
                in_features, [x.pred_boxes for x in instances]
            )
            # roi_features = model.roi_heads.box_head(roi_features)

            idx = 0
            for i, x in enumerate(batch):
                num_instances = len(instances[i])
                torch.save(
                    roi_features[idx : idx + num_instances, :MAX_INSTANCE, :],
                    Path(args.output) / f"{x['meta']['filename'].stem}.pt",
                )

                idx += num_instances


def main(args):
    if args.output is None:
        args.output = str(Path(args.input).parent / "roi_features")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    cfg = get_config(args)
    model = get_model(cfg)
    extract_roi_features(args, cfg, model)


if __name__ == "__main__":
    args = extract_feature_argument_parser().parse_args()
    print(args)
    main(args)
