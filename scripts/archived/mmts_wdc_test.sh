#!/usr/bin/env bash

export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890 no_proxy=localhost,127.0.0.0/8,*.local

FILE="./results/mmtsmatcher_wdcdatamodule/*/config.yaml"

for f in $FILE; do
    ./run.py --config $f --fit false --data.init_args.extended false --trainer.gpus 4, --trainer.default_root_dir tests
done
