#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Modified from: https://github.com/allenai/allennlp/blob/main/allennlp/commands/print_results.py

import argparse
import json
import os


def main(args: argparse.Namespace):
    """
    Prints results from an `argparse.Namespace` object.
    """
    path = args.path
    metrics_name = args.metrics_filename
    keys = args.keys

    results_dict = {}
    for root, _, files in os.walk(path):
        if metrics_name in files:
            full_name = os.path.join(root, metrics_name)
            with open(full_name) as file_:
                metrics = json.load(file_)
            results_dict[os.path.basename(root)] = metrics

    sorted_keys = sorted(list(results_dict.keys()))
    print(f"{os.path.basename(path)}, {', '.join(keys)}")
    for name in sorted_keys:
        results = results_dict[name]
        keys_to_print = (str(results.get(key, "N/A")) for key in keys)
        print(f"{name}, {', '.join(keys_to_print)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print experiment results in a helpful CSV format."
    )

    parser.add_argument(
        "path", type=str, help="Path to recursively search for experiment directories.",
    )
    parser.add_argument(
        "-k",
        "--keys",
        type=str,
        nargs="+",
        help='Keys to print from metrics.json. Keys not present in all metrics.json will result in "N/A"',
        default=[],
        required=False,
    )
    parser.add_argument(
        "-m",
        "--metrics-filename",
        type=str,
        help="Name of the metrics file to inspect.",
        default="metrics.json",
        required=False,
    )

    args = parser.parse_args()
    main(args)
