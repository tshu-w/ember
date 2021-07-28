#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import warnings

import transformers
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.seed import seed_everything

from src.lit_cli import LitCLI

transformers.logging.set_verbosity_error()

warnings.simplefilter("ignore")
seed_everything(42, workers=True)


def main():
    cli = LitCLI(
        LightningModule,
        LightningDataModule,
        trainer_defaults={
            "max_epochs": 30,
            "default_root_dir": "results",
            "checkpoint_callback": False,
        },
        shared_attrs=["collate_fn", "transforms"],
        save_config_overwrite=True,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={
            "epilog": """
Availabed Modules: src.ViLTMatcher, src.MMTSMatcher
Availabed Datamodules: src.AliDataModule, src.WDCDataModule
            """,
            "formatter_class": argparse.RawDescriptionHelpFormatter,
        },
    )


if __name__ == "__main__":
    main()
