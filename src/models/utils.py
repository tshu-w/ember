#!/usr/bin/env python

from itertools import chain
from math import isqrt
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from transformers import PreTrainedTokenizer

FEATURE_SIZE = {
    "grid": (2048, 7, 7),
    "roi": (256, 7, 7),
    "e2e": (2048, 7, 7),
}


def collate_fn(
    batch,
    text_image: str,
    text_text: str,
    use_text: bool,
    use_image: bool,
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    num_image_embeds: int = 1,
):
    batch_size = len(batch)
    texts = images = None

    raws = [x["raw"] for x in batch]
    labels = torch.LongTensor([x["label"] for x in raws])

    if text_image == "aligned":
        if max_length is not None:
            if text_text == "cross":
                max_length = max_length - 2 * num_image_embeds - 2
            elif text_text == "dual":
                max_length = max_length - num_image_embeds - 1

    if use_text:
        texts = [x["texts"] for x in batch]
        text1, text2 = map(list, zip(*texts))

        if text_text == "cross":
            texts = tokenizer(
                text1,
                text2,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )  # B x 2N x D
        elif text_text == "dual":
            texts = [
                tokenizer(
                    text1,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ),
                tokenizer(
                    text2,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ),
            ]
            # 2 x [B x N x D]

    if use_image:
        images = [x["images"] for x in batch]
        images = list(chain.from_iterable(zip(*images)))
        images_tensor = pad_sequence(images, batch_first=True)

        images = torch.stack(
            [images_tensor[:batch_size, :], images_tensor[batch_size:, :]], dim=1
        )  # B x 2 x N x D x 7 x 7

    if text_image == "aligned":
        inputs = texts
        if text_text == "cross":
            inputs["input_modals"] = images
        elif text_text == "dual":
            if images is not None:
                for i in range(2):
                    inputs[i]["input_modals"] = images[:, i : i + 1]

    elif text_image == "seperated":
        inputs = texts, images

    return inputs, labels, raws


# https://pytorch.org/vision/stable/models.html
def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def split_into_two_factors(n: int):
    for i in range(isqrt(n), 0, -1):
        if n % i == 0:
            return n // i, i
