#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, Union

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.utilities.cli import LightningCLI, SaveConfigCallback
from rich import print


class LitCLI(LightningCLI):
    def __init__(
        self,
        model_class: Union[Type[LightningModule], Callable[..., LightningModule]],
        datamodule_class: Optional[
            Union[Type[LightningDataModule], Callable[..., LightningDataModule]]
        ] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_filename: str = "config.yaml",
        save_config_overwrite: bool = False,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Dict[str, Any] = None,
        seed_everything_default: int = None,
        description: str = "pytorch-lightning trainer command line tool",
        env_prefix: str = "PL",
        env_parse: bool = False,
        parser_kwargs: Dict[str, Any] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        shared_attrs: List[str] = [],
    ) -> None:
        self.shared_attrs = shared_attrs

        super().__init__(
            model_class,
            datamodule_class=datamodule_class,
            save_config_callback=save_config_callback,
            save_config_filename=save_config_filename,
            save_config_overwrite=save_config_overwrite,
            trainer_class=trainer_class,
            trainer_defaults=trainer_defaults,
            seed_everything_default=seed_everything_default,
            description=description,
            env_prefix=env_prefix,
            env_parse=env_parse,
            parser_kwargs=parser_kwargs,
            subclass_mode_model=subclass_mode_model,
            subclass_mode_data=subclass_mode_data,
        )

    def before_fit(self):
        # share attributes between module and datamodule
        if self.datamodule is not None:
            for attr in self.shared_attrs:
                if hasattr(self.model, attr) and not hasattr(self.datamodule, attr):
                    setattr(self.datamodule, attr, getattr(self.model, attr))

                if hasattr(self.datamodule, attr) and not hasattr(self.model, attr):
                    setattr(self.model, attr, getattr(self.datamodule, attr))

        # change the name (and version) of the logger based on the modules' name and version
        exp_name = type(self.model).__name__
        exp_name += "_" + (type(self.datamodule).__name__ if self.datamodule else "")

        version = None
        if self.datamodule is not None and hasattr(self.datamodule, "version"):
            version = self.datamodule.version

        if hasattr(self.model, "version"):
            version += "_" + self.model.version

        print(f"Experiment Name: [bold]{exp_name}[/bold]")
        print(f"Version: [bold]{version}[/bold]")

        if isinstance(self.trainer.logger, Iterable):
            loggers = self.trainer.logger
        else:
            loggers = [self.trainer.logger]

        for logger in loggers:
            if hasattr(logger, "_name"):
                logger._name = exp_name.lower()
            if (
                hasattr(logger, "_version")
                and version
                and not os.getenv("PL_EXP_VERSION")
            ):
                timestramp = datetime.now().strftime("%m%d-%H%M%S")
                version += "_" + timestramp
                logger._version = version.lower()

        if (
            self.config["trainer"]["auto_lr_find"]
            or self.config["trainer"]["auto_scale_batch_size"]
        ):
            self.trainer.tune(**self.fit_kwargs)

    def after_fit(self):
        if self.trainer.checkpoint_callback.best_model_path:
            self.trainer.test()
