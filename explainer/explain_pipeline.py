import logging
import os
from typing import Dict

import pandas as pd
import torch

from explainer.configs.explainer_cfg import ExplainerConfig
from explainer.cam_generator.base_generator import BaseGenerator
from utils.base import seed_everything

pipeline_logger = logging.getLogger("Pipeline")
pipeline_logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


def explain(
    cfg: ExplainerConfig,
    path_to_cases: Dict[str, str],
    debug=False,
):
    # save config to file:
    if not os.path.exists(cfg.working_dir):
        raise Exception(
            f"working_dir {cfg.working_dir} does not exist, please run training first"
        )

    file_handler = logging.FileHandler(f"{cfg.log_dir}/explainer_{cfg.trail_id}.log")
    file_handler.setFormatter(formatter)
    pipeline_logger.addHandler(file_handler)
    model_class = cfg.model_class
    dataset_class = cfg.dataset_class

    seed_everything(cfg.seed)

    # check and load k-fold models
    available_folds = os.listdir(os.path.join(cfg.working_dir, "model"))
    available_folds.sort()
    valid_folds = []
    valid_weights = []
    for fold in available_folds:
        _path = os.path.join(cfg.working_dir, "model", fold)
        if os.path.isdir(_path):
            available_weights = os.listdir(_path)
            for w in available_weights:
                if w.endswith(".pth") and cfg.best_model in w:
                    valid_folds.append(fold)
                    valid_weights.append(w)
                    break

    explainer_class: BaseGenerator.__class__ = cfg.explainer_class
    for test_name, _path in path_to_cases.items():
        explain_df = pd.read_csv(_path)
        if debug:
            explain_df = explain_df.iloc[:1]
        pipeline_logger.info(f"Start to explain test case {test_name}")
        pipeline_logger.info(f"Explain cases: {_path}")
        model_list = []
        for ind, (fold, weight) in enumerate(zip(valid_folds, valid_weights)):
            model = model_class(**cfg.get_model_setups())
            model.on_grad_cam = True
            model.name = fold

            load_model_path = os.path.join(cfg.working_dir, "model", fold, weight)
            checkpoint = torch.load(load_model_path, map_location="cpu")
            if hasattr(model, "load_prev_version"):
                model.load_prev_version(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])

            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()

            model_list.append(model)

        # load dataset
        dataset = dataset_class(explain_df, "val", cfg)

        # load explainer
        explainer = explainer_class(
            model_list=model_list,
            cam=cfg.cam,
            dataset=dataset,
            config=cfg,
            grad_cam_targets=cfg.grad_cam_targets,
            test_name=test_name,
            debug=debug,
        )

        # generate grad cam
        explainer.process()
