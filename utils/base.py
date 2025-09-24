# Description: Base functions for the project
import os
from datetime import datetime
import random
from typing import Tuple, Optional

import numpy as np
import torch

import re
import nibabel as nib
from torch import Tensor


def pre_setup(config):
    if config.ddp:
        import cv2

        if cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL():
            cv2.setNumThreads(0)
            cv2.ocl.setUseOpenCL(False)


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False  # True
    torch.backends.cudnn.benchmark = True  # False


def get_time_str():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def setup_trail(working_dir, trail_id, config, **kwargs):
    # setup kwargs
    for key, value in kwargs.items():
        setattr(config, key, value)

    # setup trail
    config.trail_id = trail_id
    config.working_dir = f"{working_dir}/{trail_id}"
    config.model_path = f"{config.working_dir}/model"
    config.log_dir = f"{config.working_dir}/log"
    config.test_result_dir = f"{config.working_dir}/test_result"

    if config.ddp:
        print("[SETUP] Using DDP. -- local rank: ", os.environ["LOCAL_RANK"])
        if os.environ["LOCAL_RANK"] != "0":
            return config

    # setup working dir
    if not os.path.exists(config.working_dir):
        os.makedirs(config.working_dir)

    # setup model path
    if not os.path.exists(config.model_path):
        os.mkdir(config.model_path)

    # setup nnlog dir
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    # setup test result dir
    if not os.path.exists(config.test_result_dir):
        os.mkdir(config.test_result_dir)

    return config

def format_input_path(pid: Optional[str|list], study_uid: Optional[str|list], series_uid: Optional[str|list]) -> str:
    if isinstance(pid, list):
        pid = pid[0]
    if isinstance(study_uid, list):
        study_uid = study_uid[0]
    if isinstance(series_uid, list):
        series_uid = series_uid[0]

    suffix = ""
    if pid is not None and pid != "None":
        suffix = f"{pid}/"
    if study_uid is not None and study_uid != "None":
        suffix = f"{suffix}{study_uid}/"
    if series_uid is not None and series_uid != "None":
        suffix = f"{suffix}{series_uid}/"

    assert suffix != "", "At least one of pid, study_uid, series_uid should be provided."
    return suffix

def extract_coordinates(roi_str):
    """
    Extracts coordinates from a string representation of a list of integers.

    Parameters:
    - roi_str: A string representation of a list, e.g., "[1, 2, 3, 4, 5, 6]"

    Returns:
    - A tuple of integers: (xmin, ymin, zmin, xmax, ymax, zmax)
    """
    # Using regular expression to find all numbers in the string
    numbers = re.findall(r"-?\d+", roi_str)

    # Convert the found strings to integers
    coordinates = list(map(int, numbers))

    if len(coordinates) == 6:
        return tuple(coordinates)
    else:
        raise ValueError("The ROI string does not contain exactly 6 integers.")


def load_masks(image_folder, masks_to_load):
    masks = []
    for mask_name in masks_to_load:
        mask_path = os.path.join(image_folder, "segmentations", mask_name + ".nii.gz")
        mask = nib.load(mask_path).get_fdata().transpose((1, 0, 2))
        mask = mask.astype(np.float32)
        masks.append(mask)

    return masks


def extract_main_elements(ten: Tensor | Tuple[Tensor]) -> Tensor:
    """
    Extracts the main outputs from the outputs of a model.

    Parameters:
    - outputs: The outputs of a model.

    Returns:
    - A new tensor containing the main outputs.
    """
    if isinstance(ten, tuple):
        return ten[0]
    elif isinstance(ten, Tensor):
        if ten.ndim == 3:
            return ten[:, :, 0]
        elif ten.ndim == 2:
            return ten[:, 0]
        else:
            return ten
    else:
        raise ValueError(
            "The input should be a tensor or a tuple of tensors. current shape: ",
            ten.shape,
        )


def extract_aux_elements(ten: Tensor | Tuple[Tensor]) -> Tensor:
    """
    Extracts the auxiliary outputs from the outputs of a model.

    Parameters:
    - outputs: The outputs of a model.

    Returns:
    - A new tensor containing the auxiliary outputs.
    """
    if isinstance(ten, tuple):
        return ten[1]
    elif isinstance(ten, Tensor):
        if ten.ndim == 3:
            return ten[:, :, 1:]
        elif ten.ndim == 2:
            return ten[:, 1:]
        else:
            return ten
    else:
        raise ValueError(
            "The input should be a tensor or a tuple of tensors. current shape: ",
            ten.shape,
        )
