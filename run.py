#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import transformers
from pytorch_lightning import LightningDataModule, LightningModule
from src.lit_cli import LitCLI

transformers.logging.set_verbosity_error()


def main():
    cli = LitCLI(
        LightningModule,
        LightningDataModule,
        trainer_defaults={
            "max_epochs": 30,
            "default_root_dir": "results",
            "checkpoint_callback": False,
        },
        seed_everything_default=123,
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
