import logging
import copy
import os

import torch.distributed

from training.train import Trainer
from training.inference import Inferencer
from configs.base_cfg import Config

from torch.utils.data import Dataset
from torch import nn

import pandas as pd
import torch

from utils.base import seed_everything, pre_setup
from utils.ddp_utils import ddp_setup, ddp_cleanup

pipeline_logger = logging.getLogger("Pipeline")
pipeline_logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def training_phase(
    cfg: Config,
    trainer: Trainer.__class__,
    fold: int,
    phase: str,
    debug: bool,
    model_class: nn.Module.__class__,
    dataset_class: Dataset.__class__,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
):
    model = model_class(**cfg.get_model_setups(fold=fold))

    # TODO: add a function to copy and update cfg for phase2
    # adjust lr by 0.5*lr for phase2; adjust early stopping patience to "AUC" for phase2

    # model.freeze_head()
    train_dataset = dataset_class(train_df, usage="train", config=cfg)
    val_dataset = dataset_class(val_df, usage="val", config=cfg)

    train_ = trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=cfg.optimizer,
        config=cfg,
        fold=fold,
        phase=phase,
        debug=debug,
    )

    train_.train()


def inference_phase(
    cfg: Config,
    inferencer: Inferencer.__class__,
    fold: int,
    debug: bool,
    model_class: nn.Module.__class__,
    dataset_class: Dataset.__class__,
    val_df: pd.DataFrame,
):
    if cfg.ddp and torch.distributed.get_rank() != 0:
        return torch.distributed.barrier()

    if isinstance(cfg.test_dataset_path, str):
        cfg.test_dataset_path = [cfg.test_dataset_path]
    for ind, test_dataset_path in enumerate(cfg.test_dataset_path):
        model = model_class(**cfg.get_model_setups())
        if debug:
            load_model_path = f"{cfg.model_path}/fold_{fold}/last_model_fold{fold}.pth"
        else:
            best_metrics = cfg.metrics_to_monitor[0]
            load_model_path = f"{cfg.model_path}/fold_{fold}/best_{best_metrics}_fold{fold}.pth".lower()

        print(f"[Inference]: Loading model from {load_model_path}")
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        if test_dataset_path == "point_to_corresponding_val_dataset":
            test_df = val_df
        else:
            test_df = pd.read_csv(test_dataset_path)
        if debug:
            test_df = test_df.iloc[:2]

        test_dataset = dataset_class(test_df, usage="test", config=cfg)

        inference = inferencer(
            model=model,
            test_dataset=test_dataset,
            config=cfg,
            fold=fold,
            ind_test=ind,
            debug=debug,
        )
        inference.inference_and_save_predictions()

    if cfg.ddp:
        return torch.distributed.barrier()


def pipeline(
    cfg: Config,
    trainer: Trainer.__class__ | None = None,
    inferencer: Inferencer.__class__ | None = None,
    debug=False,
):
    if cfg.ddp:
        ddp_setup()

    # save config to file:
    if not os.path.exists(cfg.working_dir):
        raise Exception(
            f"working_dir {cfg.working_dir} does not exist, need to call setup_trail() first"
        )

    # save config to yaml file:
    if cfg.ddp:
        if os.environ["LOCAL_RANK"] == "0":
            cfg.save(f"{cfg.working_dir}/{cfg.trail_id}_config.yaml")
    else:
        cfg.save(f"{cfg.working_dir}/{cfg.trail_id}_config.yaml")

    # training phase:
    if isinstance(cfg.validation_dataset_path, list):
        assert len(cfg.validation_dataset_path) == cfg.folds
        assert isinstance(cfg.training_dataset_path, list)
        assert len(cfg.training_dataset_path) == cfg.folds
        k_folds = []
        for train_path, val_path in zip(
            cfg.training_dataset_path, cfg.validation_dataset_path
        ):
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            k_folds.append((train_df, val_df))
    elif cfg.validation_dataset_path is str:
        fold_length = cfg.folds
        k_folds = [
            (
                pd.read_csv(cfg.training_dataset_path),
                pd.read_csv(cfg.validation_dataset_path),
            )
        ] * fold_length
    else:
        raise ValueError("validation_dataset_path should be a list or a string")

    model_class = cfg.model_class
    dataset_class = cfg.dataset_class

    pre_setup(cfg)
    seed_everything(cfg.seed)

    for fold, (train_df, val_df) in enumerate(k_folds):
        if cfg.fold_range is not None:
            if fold not in cfg.fold_range:
                continue

        # training phase:
        if trainer is not None:
            if debug:
                train_df = train_df.sample(2)
                val_df = val_df.sample(2)

            training_phase(
                cfg=cfg,
                trainer=trainer,
                fold=fold,
                phase="phase1",
                debug=debug,
                model_class=model_class,
                dataset_class=dataset_class,
                train_df=train_df,
                val_df=val_df,
            )

            if cfg.ddp:
                torch.distributed.barrier()

        # inference phase:
        if inferencer is not None:
            inference_phase(
                cfg, inferencer, fold, debug, model_class, dataset_class, val_df
            )

        if debug:
            if fold > 0:
                break

    if cfg.ddp:
        ddp_cleanup()


def two_stage_pipeline(
    cfg: Config,
    trainer: Trainer.__class__ | None = None,
    inferencer: Inferencer.__class__ | None = None,
    debug=False,
):
    if cfg.ddp:
        ddp_setup()

    # save config to file:
    if not os.path.exists(cfg.working_dir):
        raise Exception(
            f"working_dir {cfg.working_dir} does not exist, need to call setup_trail() first"
        )

    # save config to yaml file:
    if cfg.ddp:
        if os.environ["LOCAL_RANK"] == "0":
            cfg.save(f"{cfg.working_dir}/{cfg.trail_id}_config.yaml")
    else:
        cfg.save(f"{cfg.working_dir}/{cfg.trail_id}_config.yaml")

    file_handler = logging.FileHandler(f"{cfg.log_dir}/{cfg.trail_id}_pipeline.log")
    file_handler.setFormatter(formatter)
    pipeline_logger.addHandler(file_handler)

    pipeline_logger.info(f"Initialize training pipeline for {cfg.trail_id}")
    # training phase:
    if isinstance(cfg.validation_dataset_path, list):
        assert len(cfg.validation_dataset_path) == cfg.folds
        assert isinstance(cfg.training_dataset_path, list)
        assert len(cfg.training_dataset_path) == cfg.folds
        k_folds = []
        for train_path, val_path in zip(
            cfg.training_dataset_path, cfg.validation_dataset_path
        ):
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            k_folds.append((train_df, val_df))
    elif cfg.validation_dataset_path is str:
        fold_length = cfg.folds
        k_folds = [
            (
                pd.read_csv(cfg.training_dataset_path),
                pd.read_csv(cfg.validation_dataset_path),
            )
        ] * fold_length
    else:
        raise ValueError("validation_dataset_path should be a list or a string")

    model_class = cfg.model_class
    dataset_class = cfg.dataset_class

    pre_setup(cfg)
    seed_everything(cfg.seed)

    pipeline_logger.info(f"Start training pipeline for {cfg.trail_id}")
    for fold, (train_df, val_df) in enumerate(k_folds):
        if cfg.fold_range is not None:
            if fold not in cfg.fold_range:
                continue

        # training phase one:
        if trainer is not None and cfg.phase == "phase1":
            pipeline_logger.info(f"Start training phase one for fold {fold}")
            if debug:
                train_df = train_df.sample(2)
                val_df = val_df.sample(2)

            training_phase(
                cfg=cfg,
                trainer=trainer,
                fold=fold,
                phase="phase1",
                debug=debug,
                model_class=model_class,
                dataset_class=dataset_class,
                train_df=train_df,
                val_df=val_df,
            )
            pipeline_logger.info(f"Training phase one for fold {fold} is done.")

            if cfg.ddp:
                torch.distributed.barrier()

        # training phase two:
        if trainer is not None:
            pipeline_logger.info(f"Start training phase two for fold {fold}")
            if debug:
                train_df = train_df.sample(2)
                val_df = val_df.sample(2)

            training_phase(
                cfg=cfg,
                trainer=trainer,
                fold=fold,
                phase="phase2",
                debug=debug,
                model_class=model_class,
                dataset_class=dataset_class,
                train_df=train_df,
                val_df=val_df,
            )
            pipeline_logger.info(f"Training phase two for fold {fold} is done.")

            if cfg.ddp:
                torch.distributed.barrier()

        # inference phase:
        if inferencer is not None:
            pipeline_logger.info(f"Start inference phase for fold {fold}")
            inference_phase(
                cfg, inferencer, fold, debug, model_class, dataset_class, val_df
            )
            pipeline_logger.info(f"Inference phase for fold {fold} is done.")

        pipeline_logger.info(f"Training pipeline for fold {fold} is done.")

        if debug:
            if fold > 0:
                break

    pipeline_logger.info(f"Training pipeline for trail {cfg.trail_id} is done.")
    if cfg.ddp:
        ddp_cleanup()
