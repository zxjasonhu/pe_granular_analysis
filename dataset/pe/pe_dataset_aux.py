import os
import random

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from dataset.augmentations.pe_base_data_augmentation import get_pe_transforms
from dataset.base_dataset import dataset_getitem_error_logger, CTDataset
from utils.base import load_masks
from utils.image_operations import window_normalization, process_one_segment


class PECTVolumeAuxLoss(CTDataset):
    def __init__(self, dataframe: pd.DataFrame, usage: str = "train", config=None):
        super(PECTVolumeAuxLoss, self).__init__(dataframe, usage, config)

        if config is not None:
            self.img_depth = config.img_depth

            self.pad_seq_when_shorter = config.pad_seq_when_shorter

            if hasattr(config, "masked_label_portion"):
                self.masked_label_portion = config.masked_label_portion
            else:
                self.masked_label_portion = 0.0

        if usage in ["train", "val", "test"]:
            assert config is not None, "config is None"
            assert (
                config.aux_loss_dataset_path is not None
            ), "aux_loss_df is not provided in the config"
            assert (
                config.aux_loss_target is not None
            ), "aux_loss_target is not provided in the config"
            self.aux_loss_df = pd.read_csv(config.aux_loss_dataset_path)
            self.aux_loss_target = config.aux_loss_target

        self.augmentations = get_pe_transforms(config, target=usage)


    @dataset_getitem_error_logger
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_folder = row["image_folder"]

        # load nifti image
        nifti_path = os.listdir(image_folder)
        nifti_path = [
            x for x in nifti_path if x.endswith(".nii.gz") or x.endswith(".nii")
        ][0]
        image_path = os.path.join(image_folder, nifti_path)

        # will load w, h, length image, here c is the axis of slices
        image_sequence = (
            nib.load(image_path).get_fdata().astype(np.float32).transpose((1, 0, 2))
        )  # Convert to HWL for augmentations
        source_voxel_shape = image_sequence.shape

        # Normalize and process the image
        image_sequence = window_normalization(
            image_sequence, self.config.window_center, self.config.window_width
        )

        masks = None
        masks_to_load = self.config.masks
        if self.config.load_mask and masks_to_load is not None:
            masks = load_masks(image_folder, masks_to_load)
            assert len(masks) == 1, "The mask should have 3 channels."

        _ = process_one_segment(
            eval(row["lung"]) if isinstance(row["lung"], str) else row["lung"],
            ct_image=image_sequence,
            image_channels=self.config.image_channels,
            mask=masks[0] if masks is not None else None,
            usage=self.usage,
            xy_plane_enlarge_factor=0.0,
            z_axis_enlarge_factor=0.0,
            img_depth=self.img_depth,
            pad_seq_when_shorter=self.pad_seq_when_shorter,
            augmentations=self.augmentations,
            on_slice_augmentation=True,
            return_dict=True,
        )

        img = _["image"]  # c, d, h, w
        img = img.transpose((1, 0, 2, 3))  # d, c, h, w
        xy_boxes = _["bbox"]
        z_slices = _["center_slice_indices"]

        if self.usage != "train":
            _img_depth = len(z_slices)
        else:
            _img_depth = self.img_depth

        if self.config.random_seq and self.usage == "train":
            if random.random() < 0.5:
                img = np.flip(img, axis=0)

        tensor = torch.from_numpy(img)

        if self.usage in ["cam", "inference"]:
            return {
                "images": tensor,
                "xy_boxes": xy_boxes,
                "z_slices": z_slices,
                "pid": str(row["PatientID"]) if "PatientID" in row else "None",
                "study_uid": str(row["StudyInstanceUID"]) if "StudyInstanceUID" in row else "None",
                "series_uid": str(row["SeriesInstanceUID"]) if "SeriesInstanceUID" in row else "None",
                "input_folder": image_folder,
                "source_voxel_shape": source_voxel_shape,
            }

        # Main target
        target = torch.tensor(row[self.config.target_label], dtype=torch.float32)

        # Aux target
        suid = row["StudyInstanceUID"]

        # Preprocess to find the slice indices with the target value of 1
        local_df = self.aux_loss_df[self.aux_loss_df["StudyInstanceUID"] == suid]
        target_slices_with_aux_target = set(
            local_df[local_df[self.config.aux_loss_target] == 1]["slice_index"]
        )
        aux_target_tensor = torch.zeros((len(z_slices),), dtype=torch.float32)
        selected_aux_label_predictions = torch.ones_like(
            aux_target_tensor, dtype=torch.float32
        )

        _zmin, _zmax = z_slices[0], z_slices[-1]
        masked_label_portion = self.masked_label_portion
        if masked_label_portion > 0:
            _num_selected = int((1 - masked_label_portion) * (_zmax - _zmin + 1) + 0.5)
            _num_selected = min(_num_selected, _img_depth)
            _num_masks = _img_depth - _num_selected
            _masked_indices = np.linspace(0, _img_depth - 1, _num_masks, dtype=int)
            selected_aux_label_predictions[_masked_indices] = 0.0

        for ind, z_slice in enumerate(z_slices):
            if z_slice in target_slices_with_aux_target:
                aux_target_tensor[ind] = 1.0

        return {
            "images": tensor,
            "target": target,
            self.config.aux_loss_target: aux_target_tensor,
            "selected_aux_label_predictions": selected_aux_label_predictions,
        }

    def get_weights(self, debug=False):
        if debug:
            return np.ones((len(self.dataframe),))
        overall = self.dataframe[self.config.target_label].values
        weights = np.zeros((len(overall),))
        weights[overall == 0] = len(overall) / (len(overall) - overall.sum())
        weights[overall == 1] = len(overall) / overall.sum()
        return weights
