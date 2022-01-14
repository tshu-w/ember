#!/usr/bin/env python

import argparse
import json
import multiprocessing
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path
from string import Template

PROJECT_DIR = Path(__file__).parent.parent
EXP_DIR = PROJECT_DIR / "logs" / "ali"
EXP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ARGS = {
    "config": "wdc_tm",
    "cat": "all",
    "extra_test": False,
    "num_workers": 32,
    "seed": 142,
}
EXPT_TMP = Template(
    """{
  "data": {
    "class_path": "src.WDCDataModule",
    "init_args": {
      "cat": "${cat}",
      "extra_test": "${extra_test}",
      "num_workers": "${num_workers}"
    }
  },
  "seed": "${seed}",
  "config": "${config}"
}"""
)
EXPTS = []
SEEDS = [142, 123]

for seed in SEEDS:
    for config in ["wdc_tm"]:
        for extra_test in [False, True]:
            for cat in ["all", "cameras", "computers", "shoes", "watches"]:
                kwargs = {
                    "cat": cat,
                    "config": config,
                    "seed": seed,
                    "extra_test": extra_test,
                }
                EXPTS.append(EXPT_TMP.substitute(DEFAULT_ARGS, **kwargs))

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["all_proxy"] = "socks5://127.0.0.1:7890"
os.environ["no_proxy"] = "localhost,127.0.0.0/8,*.local"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        "--no-run", action="store_true", help="whether not running command"
    )
    parser.add_argument("--gpus", nargs="+", default=["0"], help="availabled gpus")

    return parser


def run(exp_args, args):
    worker_id = int(multiprocessing.current_process().name.rsplit("-", 1)[1]) - 1
    gpu = args.gpus[worker_id % len(args.gpus)]

    exp_name = f"{exp_args['config']}_{exp_args['data']['init_args']['cat']}"

    outfile = EXP_DIR / f"{exp_name}_{exp_args['seed']}_out.log"
    errfile = EXP_DIR / f"{exp_name}_{exp_args['seed']}_err.log"

    if args.fast_dev_run:
        cmd = f"""./run fit \\
        --config configs/{exp_args['config']}.yaml \\
        --seed_everything {exp_args['seed']} \\
        --trainer.gpus {gpu}, --trainer.fast_dev_run {args.fast_dev_run} \\
        --data '{exp_args['data']}'"""
    else:
        cmd = f"""./run fit \\
        --config configs/{exp_args['config']}.yaml \\
        --seed_everything {exp_args['seed']} \\
        --trainer.gpus {gpu}, --trainer.default_root_dir wdc_results --trainer.max_epochs 30 \\
        --data '{exp_args['data']}' \\
        >{outfile} 2>{errfile}
        """

    print(exp_name, cmd, sep="\n")

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
            run,
            kwds={
                "exp_args": json.loads(expt, strict=False),
                "args": args,
            },
        )

    pool.close()
    pool.join()
