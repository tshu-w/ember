#!/usr/bin/env bash

export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890 no_proxy=localhost,127.0.0.0/8,*.local

FILE="./results/multimodalmatcher_alidatamodule/*/config.yaml"

for f in $FILE; do
    ./run.py --config $f --fit false --trainer.gpus 5, --trainer.default_root_dir tests --data.init_args.out_domain true
done

