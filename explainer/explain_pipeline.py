import logging
import os
from typing import Dict, List

import pandas as pd
import torch

from explainer.configs.explainer_cfg import ExplainerConfig
from explainer.cam_generator.base_generator import BaseGenerator
from utils.base import seed_everything


def explain(
    cfg: ExplainerConfig,
    model_weight_paths: str | List[str],
    cases_to_explain: str | List[Dict] | pd.DataFrame,
    test_name: str = "cam_test",
    debug=False,
):
    model_class = cfg.model_class
    dataset_class = cfg.dataset_class

    seed_everything(cfg.seed)

    if isinstance(model_weight_paths, str):
        model_weight_paths = [model_weight_paths]
    valid_model_weight_paths = []
    for _path in model_weight_paths:
        if os.path.exists(_path):
            valid_model_weight_paths.append(_path)
        else:
            print(f"Model weight path {_path} does not exist, skip.")

    explainer_class = cfg.explainer_class
  
    model_list = []
    for ind, load_model_path in enumerate(valid_model_weight_paths):
        model = model_class(**cfg.get_model_setups())
        model.on_grad_cam = True
        model.name = f"fold_{ind}"

        if "custom_load_from_checkpoint" in dir(model):
            model.custom_load_from_checkpoint(load_model_path)
            print(f"Load model from {load_model_path} using custom load function.")
        else:
            checkpoint = torch.load(load_model_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"Load model from {load_model_path}.")

        if torch.cuda.is_available():
            model = model.cuda()
        # model.eval()

        model_list.append(model)

    # load dataset
    if isinstance(cases_to_explain, str):
        explain_df = pd.read_csv(cases_to_explain)
    elif isinstance(cases_to_explain, list):
        explain_df = pd.DataFrame(cases_to_explain)
    elif isinstance(cases_to_explain, pd.DataFrame):
        explain_df = cases_to_explain
    else:
        raise Exception(
            "cases_to_explain should be str (path to csv), list of dict or pd.DataFrame"
        )

    dataset = dataset_class(explain_df, "cam", cfg)

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
