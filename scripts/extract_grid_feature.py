#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import warnings
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def extract_feature_argument_parser():
    parser = argparse.ArgumentParser(description="ROI feature extraction")
    parser.add_argument(
        "--backbone",
        help="backbone name",
        default="resnet152",
        # https://pytorch.org/vision/stable/models.html
        # https://lightning-flash.readthedocs.io/en/stable/reference/image_embedder.html
    )
    parser.add_argument("--input", metavar="DIR", help="image directory", required=True)
    parser.add_argument("--output", metavar="DIR", help="result directory")

    return parser


def collate_fn(batch):
    images = [x["image"] for x in batch]
    images = torch.stack(images)
    meta = [x["meta"] for x in batch]

    return {
        "images": images,
        "meta": meta,
    }


class ImageDataset(Dataset):
    def __init__(self, root, transforms=TRANSFORMS):
        self.image_list = list(Path(root).rglob("*"))
        self.transforms = transforms

    def __getitem__(self, index):
        image_path = self.image_list[index]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = Image.open(image_path).convert("RGB")

        image = self.transforms(image)

        return {
            "image": image,
            "meta": {"filename": Path(image_path)},
        }

    def __len__(self):
        return len(self.image_list)


def extract_grid_features(args):
    dataset = ImageDataset(args.input)
    dataloader = DataLoader(
        dataset=dataset, batch_size=16, num_workers=4, collate_fn=collate_fn
    )
    model = getattr(torchvision.models, args.backbone)(pretrained=True)
    modules = list(model.children())[:-2]
    model = torch.nn.Sequential(*modules)
    model.cuda()
    model.eval()

    for batch in tqdm(dataloader):
        with torch.no_grad():
            features = model(batch["images"].cuda())
            for i, x in enumerate(batch["meta"]):
                torch.save(
                    features[i, :], Path(args.output) / f"{x['filename'].stem}.pt",
                )


def main(args):
    if args.output is None:
        args.output = str(Path(args.input).parent / "grid_features")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    extract_grid_features(args)


if __name__ == "__main__":
    args = extract_feature_argument_parser().parse_args()
    print(args)
    main(args)
