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
    "config": "ali_tm",
    "cat": "all",
    "test_name": "",
    "seed": 142,
    "num_workers": 32,
    "ckpt_path": "",
}
EXPT_TMP = Template(
    """{
  "data": {
    "class_path": "src.AliDataModule",
    "init_args": {
      "cat": "${cat}",
      "num_workers": "${num_workers}",
      "test_name": "${test_name}"
    }
  },
  "seed": "${seed}",
  "config": "${config}",
  "ckpt_path": "${ckpt_path}"
}"""
)
EXPTS = []
SEEDS = [142, 123]

cfg2dir = {
    "ali_tm": "textmatcher_alidatamodule/",
    "ali_vm": "visionmatcher_alidatamodule/",
    "ali_mm": "multimodalmatcher_alidatamodule/",
}

for seed in SEEDS:
    # for config in ["ali_tm", "ali_vm", "ali_mm"]:
    for config in ["ali_tm"]:
        for cat in ["all", "clothing", "shoes", "accessories"]:
            ckpt_path = next(
                (Path("./results") / cfg2dir[config]).glob(
                    f"*_{cat}_{seed}_*/checkpoints/*.ckpt"
                )
            )
            for k in [4, 10, 20, 40, 80, 100]:
                for test_name in ["", "nr", "nrs", "nc"]:
                    if k == 4:
                        test_name == test_name
                    elif k == 100:
                        test_name = f"i{test_name}"
                    else:
                        test_name = f"i{test_name}_{k}"

                    kwargs = {
                        "config": config,
                        "cat": cat,
                        "test_name": test_name,
                        "seed": seed,
                        "ckpt_path": ckpt_path,
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
        "--no-run", action="store_true", help="whether to not run command"
    )
    parser.add_argument("--gpus", nargs="+", default=["0"], help="availabled gpus")

    return parser


def run(exp_args, args):
    worker_id = int(multiprocessing.current_process().name.rsplit("-", 1)[1]) - 1
    gpu = args.gpus[worker_id % len(args.gpus)]

    exp_name = f"{exp_args['config']}_{exp_args['data']['init_args']['cat']}_{exp_args['data']['init_args']['test_name']}"

    outfile = EXP_DIR / f"{exp_name}_{exp_args['seed']}_out.log"
    errfile = EXP_DIR / f"{exp_name}_{exp_args['seed']}_err.log"

    if args.fast_dev_run:
        cmd = f"""./run test \\
        --config configs/{exp_args['config']}.yaml \\
        --seed_everything {exp_args['seed']} \\
        --trainer.gpus {gpu}, --trainer.fast_dev_run {args.fast_dev_run} \\
        --data '{exp_args['data']}' \\
        --ckpt_path {exp_args['ckpt_path']}
        """
    else:
        cmd = f"""./run test \\
        --config configs/{exp_args['config']}.yaml \\
        --seed_everything {exp_args['seed']} \\
        --trainer.gpus {gpu}, --trainer.default_root_dir test_ratio \\
        --data '{exp_args['data']}' \\
        --ckpt_path {exp_args['ckpt_path']} \\
        >{outfile} 2>{errfile}
        """

    print(exp_name, "\n", cmd)

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
