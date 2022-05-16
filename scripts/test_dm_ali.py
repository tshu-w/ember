import argparse
import multiprocessing
import subprocess
from multiprocessing import Pool
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
EXP_DIR = PROJECT_DIR / "results" / "logs" / "ali"
EXP_DIR.mkdir(parents=True, exist_ok=True)

EXPTS = []

for cat in ["all", "clothing", "shoes", "accessories"]:
    for test_file in [
        "test",
        "test_rl",
        "test_cfm",
        "test_om",
        "test_i",
        "test_irl",
        "test_icfm",
        "test_iom",
    ]:
        expt = {"cat": cat, "test_file": test_file}
        EXPTS.append(expt)


def argument_parser():
    parser = argparse.ArgumentParser(description="run experiments in parallel")
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
    cat = exp_args["cat"]
    test_file = exp_args["test_file"]

    exp_name = f"deepmatcher_{cat}_{test_file}"

    outfile = EXP_DIR / f"{exp_name}_out.log"
    errfile = EXP_DIR / f"{exp_name}_err.log"

    dir = Path(f"./data/ali/datasets/{cat}/")

    model_dir = f"deepmatcher_{cat}"

    cmd = f"""python scripts/dm.py \\
    --dir {dir} \\
    --gpu {gpu} \\
    --output results/test/{exp_name} \\
    --no-fit --test-file {test_file} --model-dir results/fit/{model_dir} \\
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
                "exp_args": expt,
                "args": args,
            },
        )

    pool.close()
    pool.join()
