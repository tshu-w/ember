import argparse
import json
import multiprocessing
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path
from string import Template

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_DIR = Path(__file__).parent.parent
EXP_DIR = PROJECT_DIR / "results" / "logs" / "ali"
EXP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_ARGS = {
    "config": "ali_tm",
    "cat": "all",
    "seed": 142,
    "test_name": "",
    "ckpt_path": None,
}
EXPT_TMP = Template(
    """{
  "data": {
    "class_path": "src.datamodules.AliDataModule",
    "init_args": {
      "cat": "${cat}",
      "test_name": "${test_name}",
      "num_workers": 32
    }
  },
  "seed": "${seed}",
  "config": "${config}",
  "ckpt_path": "${ckpt_path}"
}"""
)
EXPTS = []
SEEDS = [142, 123]

for seed in SEEDS:
    for config in ["ali_tm", "ali_vm", "ali_mm"]:
        for cat in ["all", "clothing", "shoes", "accessories"]:
            ckpt_path = next(
                (Path("./results/fit") / f"{config}_{cat}_{seed}").rglob("*.ckpt")
            )
            for test_name in ["", "rl", "cfm", "om"]:
                kwargs = {
                    "config": config,
                    "cat": cat,
                    "test_name": test_name,
                    "seed": seed,
                    "ckpt_path": ckpt_path,
                }
                EXPTS.append(EXPT_TMP.substitute(DEFAULT_ARGS, **kwargs))


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

    exp_name = f"{exp_args['config']}_{exp_args['data']['init_args']['cat']}_{exp_args['data']['init_args']['test_name']}_{exp_args['seed']}"

    outfile = EXP_DIR / f"{exp_name}_out.log"
    errfile = EXP_DIR / f"{exp_name}_err.log"

    if args.fast_dev_run:
        cmd = f"""./run test \\
        --debug \\
        --config configs/{exp_args['config']}.yaml \\
        --seed_everything {exp_args['seed']} \\
        --trainer.gpus {gpu}, --trainer.fast_dev_run {args.fast_dev_run} \\
        --data '{exp_args['data']}' \\
        --ckpt_path {exp_args['ckpt_path']}
        """
    else:
        cmd = f"""./run test \\
        --config configs/{exp_args['config']}.yaml \\
        --name {exp_name} \\
        --seed_everything {exp_args['seed']} \\
        --trainer.gpus {gpu}, \\
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
