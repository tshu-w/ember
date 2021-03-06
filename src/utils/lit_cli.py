import argparse
import json
import logging
import os
from collections import ChainMap, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import shtab
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI
from pytorch_lightning.utilities.metrics import metrics_to_scalars


class LitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("-n", "--name", default="none", help="Experiment name")
        parser.add_argument(
            "-d",
            "--debug",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Debug mode",
        )

        for arg in ["num_labels", "task_name"]:
            parser.link_arguments(
                f"data.init_args.{arg}",
                f"model.init_args.{arg}",
                apply_on="instantiate",
            )

    def before_instantiate_classes(self) -> None:
        config = self.config[self.subcommand]
        mode = "debug" if config.debug else self.subcommand
        timestamp = datetime.now().strftime("%m-%dT%H%M%S")

        config.trainer.default_root_dir = os.path.join(
            "results", mode, config.name, timestamp
        )

        if mode == "debug":
            config.trainer.logger = None

        logger = config.trainer.logger
        assert logger != True, "should assign trainer.logger with the specific logger."
        if logger:
            loggers = logger if isinstance(logger, Iterable) else [logger]
            for logger in loggers:
                logger.init_args.save_dir = os.path.join(
                    logger.init_args.get("save_dir", "results"), self.subcommand
                )
                logger.init_args.name = config.name
                logger.init_args.version = timestamp

    def after_run(self) -> None:
        results = {}

        if self.trainer.state.fn == TrainerFn.FITTING:
            if (
                self.trainer.checkpoint_callback
                and self.trainer.checkpoint_callback.best_model_path
            ):
                ckpt_path = self.trainer.checkpoint_callback.best_model_path
                # inhibit disturbing logging
                logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(
                    logging.WARNING
                )
                logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(
                    logging.WARNING
                )

                self.trainer.callbacks = []
                fn_kwargs = {
                    "model": self.model,
                    "datamodule": self.datamodule,
                    "ckpt_path": ckpt_path,
                }
                has_val_loader = (
                    self.trainer._data_connector._val_dataloader_source.is_defined()
                )
                has_test_loader = (
                    self.trainer._data_connector._test_dataloader_source.is_defined()
                )

                val_results = (
                    self.trainer.validate(**fn_kwargs) if has_val_loader else []
                )
                test_results = self.trainer.test(**fn_kwargs) if has_test_loader else []

                results = dict(ChainMap(*val_results, *test_results))
        else:
            results = metrics_to_scalars(self.trainer.logged_metrics)

        if results:
            results_str = json.dumps(results, ensure_ascii=False, indent=2)

            metrics_file = Path(self.trainer.log_dir) / "metrics.json"
            with metrics_file.open("w") as f:
                f.write(results_str)

    after_fit = after_validate = after_test = after_run

    def setup_parser(
        self,
        add_subcommands: bool,
        main_kwargs: dict[str, Any],
        subparser_kwargs: dict[str, Any],
    ) -> None:
        """Initialize and setup the parser, subcommands, and arguments."""
        # move default_config_files to subparser_kwargs
        if add_subcommands:
            default_configs = main_kwargs.pop("default_config_files", None)
            subparser_kwargs = defaultdict(dict, subparser_kwargs)
            for subcmd in self.subcommands():
                subparser_kwargs[subcmd]["default_config_files"] = default_configs

        self.parser = self.init_parser(**main_kwargs)
        shtab.add_argument_to(self.parser, ["-s", "--print-completion"])

        if add_subcommands:
            self._subcommand_method_arguments: dict[str, list[str]] = {}
            self._add_subcommands(self.parser, **subparser_kwargs)
        else:
            self._add_arguments(self.parser)
