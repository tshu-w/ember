#!/usr/bin/env bash

export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890 no_proxy=localhost,127.0.0.0/8,*.local

###############################################################################
#                                    small                                    #
###############################################################################
./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "all", "training_size": "small"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "cameras", "training_size": "small"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "computers", "training_size": "small"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "shoes", "training_size": "small"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "watches", "training_size": "small"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "all", "training_size": "small"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "cameras", "training_size": "small"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "computers", "training_size": "small"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "shoes", "training_size": "small"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "watches", "training_size": "small"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

###############################################################################
#                                    medium                                   #
###############################################################################
./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "all", "training_size": "medium"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "cameras", "training_size": "medium"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "computers", "training_size": "medium"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "shoes", "training_size": "medium"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "watches", "training_size": "medium"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "all", "training_size": "medium"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "cameras", "training_size": "medium"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "computers", "training_size": "medium"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "shoes", "training_size": "medium"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "watches", "training_size": "medium"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

###############################################################################
#                                    large                                    #
###############################################################################
./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "all", "training_size": "large"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "cameras", "training_size": "large"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "computers", "training_size": "large"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "shoes", "training_size": "large"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --seed_everything 123 \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "watches", "training_size": "large"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "all", "training_size": "large"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "cameras", "training_size": "large"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "computers", "training_size": "large"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "shoes", "training_size": "large"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,

./run.py \
 --data '{"class_path": "src.WDCDataModule", "init_args": {"use_image": False, "cate": "watches", "training_size": "large"}}' \
 --model '{"class_path": "src.MMTSMatcher", "init_args": {"model_name": "bert-base-uncased", "max_length": 128}}' \
 --trainer.gpus 4,
