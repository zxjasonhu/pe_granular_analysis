import os
from typing import Optional, List

import torch
import torch.nn as nn
from pytorch_grad_cam.base_cam import BaseCAM
from torch.utils.data import Dataset, DataLoader

from configs.base_cfg import Config


class BaseGenerator:
    def __init__(
        self,
        model_list: List[nn.Module],
        cam: BaseCAM.__class__,
        dataset: Dataset,
        config: Config,
        grad_cam_targets: Optional[list] = None,
        test_name: str = "cam_test",
        debug: bool = False,
    ):
        """
        :param model: only one model
        :param test_dataset:
        :param config:
        :param fold:
        :param debug:
        """
        # setup device
        if config.ddp:
            print("DDP is not supported, will use single GPU")
        self.rank = 0

        self.model_list = model_list

        self.grad_cam_target_layers_list = [
            model.grad_cam_target_layers for model in model_list
        ]

        if grad_cam_targets is not None:
            self.grad_cam_targets = grad_cam_targets

        self.config = config

        grad_cam_folder = os.path.join(self.config.working_dir, f"grad_cam_outputs")
        if not os.path.exists(grad_cam_folder):
            os.makedirs(grad_cam_folder, exist_ok=True)

        self.result_output_folder = os.path.join(grad_cam_folder, f"{test_name}")
        if not os.path.exists(self.result_output_folder):
            os.makedirs(self.result_output_folder, exist_ok=True)

        self.loader = DataLoader(
            dataset,
            num_workers=config.num_workers,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
        )
        self.grad_cam = cam

        self.test_name = test_name
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def process(self):
        self.clean()
        raise NotImplementedError("This method should be implemented in the subclass.")

    def clean(self):
        print("Cleaning up...")
        del self.loader
        del self.model_list
        del self.config
        del self.grad_cam

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc

        gc.collect()
        print("Explainer Cleaned.")


if __name__ == "__main__":
    pass
