#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import os
from collections import ChainMap
from datetime import datetime
from pathlib import Path

from pytorch_lightning.loggers import LoggerCollection
from pytorch_lightning.utilities.cli import LightningArgumentParser, LightningCLI
from rich import print


class LitCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument(
            "--fit", type=bool, default=True, help="Whether fit or not."
        )
        parser.add_argument(
            "--shared_attrs", nargs="+", default=["collate_fn", "transforms"],
        )

        for arg in ["use_text", "use_image", "feature_type", "num_image_embeds"]:
            parser.link_arguments(f"model.init_args.{arg}", f"data.init_args.{arg}")

    def before_instantiate_classes(self) -> None:
        if not self.config["fit"]:
            self.config["trainer"]["max_steps"] = 0

    def before_fit(self) -> None:
        # share attributes between module and datamodule
        if self.datamodule is not None:
            for attr in self.config["shared_attrs"]:
                if hasattr(self.model, attr) and not hasattr(self.datamodule, attr):
                    setattr(self.datamodule, attr, getattr(self.model, attr))

                if hasattr(self.datamodule, attr) and not hasattr(self.model, attr):
                    setattr(self.model, attr, getattr(self.datamodule, attr))

        # change the name (and version) of the logger based on the modules' name and
        # version
        exp_name = type(self.model).__name__
        exp_name += "_" + (type(self.datamodule).__name__ if self.datamodule else "")

        version = ""
        if hasattr(self.model, "get_version"):
            version = self.model.get_version()

        if self.datamodule is not None and hasattr(self.datamodule, "version"):
            version += ("_" if version else "") + self.datamodule.version

        version += "_" + str(self.config["seed_everything"])

        if version and self.config["fit"]:
            timestramp = datetime.now().strftime("%m%d-%H%M%S")
            version += "_" + timestramp

        version = version.replace("/", "-")

        if self.config["trainer"]["resume_from_checkpoint"]:
            exp_path = Path(self.config["trainer"]["resume_from_checkpoint"]).parents[1]
            if os.path.commonprefix([exp_path.name, version or ""]):
                version = exp_path.name

        print(f"Experiment Name: [bold]{exp_name}[/bold]")
        print(f"Version: [bold]{version}[/bold]")

        if not isinstance(self.trainer.logger, LoggerCollection):
            self.trainer.logger._name = exp_name.lower()
            if (
                hasattr(self.trainer.logger, "_version")
                and version
                and not os.getenv("PL_EXP_VERSION")
            ):
                self.trainer.logger._version = version.lower()

        if (
            self.config["trainer"]["auto_lr_find"]
            or self.config["trainer"]["auto_scale_batch_size"]
        ):
            self.trainer.tune(**self.fit_kwargs)

    def after_fit(self):
        ckpt_path = None

        if self.trainer.checkpoint_callback.best_model_path:
            # HACK: https://github.com/PyTorchLightning/pytorch-lightning/discussions/8759
            ckpt_path = self.trainer.checkpoint_callback.best_model_path
        elif self.config["config"]:
            config_dir = Path(self.config["config"][0]()).parent
            checkpoint_paths = list(config_dir.rglob("*.ckpt"))

            if len(checkpoint_paths) == 1:
                ckpt_path = checkpoint_paths[0]

        if ckpt_path:
            # Disable useless logger after fit
            logging.getLogger("pytorch_lightning.utilities.distributed").setLevel(
                logging.WARNING
            )
            logging.getLogger("pytorch_lightning.accelerators.gpu").setLevel(
                logging.WARNING
            )

            val_results = self.trainer.validate(ckpt_path=ckpt_path, verbose=False)
            test_results = self.trainer.test(ckpt_path=ckpt_path, verbose=False)

            results = dict(ChainMap(*val_results, *test_results))

            print(json.dumps(results, ensure_ascii=False, indent=2))

            metrics = Path(self.trainer.log_dir) / "metrics.json"
            with metrics.open("w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
