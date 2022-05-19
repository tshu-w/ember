<div align="center">

<h2 id="bridging-the-gap-between-reality-and-ideality-of-entity-matching-a-revisiting-and-benchmark-re-construction">Bridging the Gap between Reality and Ideality of Entity Matching:<br/>A Revisiting and Benchmark Re-Construction</h2>

<p>
<a href="https://arxiv.org/abs/2205.05889"><img src="http://img.shields.io/badge/arxiv-2205.05889-B31B1B.svg" alt="Arxiv" /></a>
<a href="https://ijcai-22.org"><img src="http://img.shields.io/badge/IJCAI-2022-4b44ce.svg" alt="Conference" /></a>
</p>

</div>

## Description
Code and data for the paper:

*Bridging the Gap between Reality and Ideality of Entity Matching: A Revisiting and Benchmark Re-Construction*

## Data
Details of the released data can be found in the [REAME](./data/ali/README.md) of the data.

## How to run
First, install dependencies
```console
# clone project
git clone https://github.com/tshu-w/ember
cd ember

# [SUGGESTED] use conda environment
conda env create -n ember -f environment.yaml
conda activate ember

# [OPTIONAL] install requirements directly
pip install -r requirements.txt
```

Next, to obtain the main results of the paper:
```console
bash scripts/download_images.sh

python scripts/run_ali.py --gpus 0 1 2 3
python scripts/test_ali.py --gpus 0 1 2 3
python scripts/run_dm_ali.py --gpus 0 1 2 3
python scripts/test_dm_ali.py --gpus 0 1 2 3

python scripts/print_results results/test -k test/f1 test/prc test/rec
```

You can also run experiments with the `run` script.
```console
# fit with the TextMatcher config
./run fit --config configs/ali_tm.yaml
# or specific command line arguments
./run fit --model TextMatcher --data AliDataModule --data.batch_size 32 --trainer.gpus 0,

# evaluate with the checkpoint
./run test --config configs/ali_tm.yaml --ckpt_path ckpt_path

# get the script help
./run --help
./run fit --help
```

## Citation
```
@inproceedings{wang-2022-bridging-gap,
  title={Bridging the Gap between Reality and Ideality of Entity Matching: A Revisiting and Benchmark Re-Construction},
  author={Tianshu Wang and Hongyu Lin and Cheng Fu and Xianpei Han and Le Sun and Feiyu Xiong and Hui Chen and Minlong Lu and Xiuwen Zhu},
  year={2022}
}
```
