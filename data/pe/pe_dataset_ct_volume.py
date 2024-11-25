import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from data.augmentations.pe_base_data_augmentation import get_pe_transforms
from data.base_dataset import CTDataset, dataset_getitem_error_logger
from utils.image_operations import window_normalization, crop_and_pad


class PECTVolumeDataset(CTDataset):
    """
    This dataset is used for 3D CT volume classification.
    Augmentations are applied to the slice-level.
    """

    def __init__(self, dataframe: pd.DataFrame, usage: str = "train", config=None):
        super(PECTVolumeDataset, self).__init__(dataframe, usage, config)
        # Set up a specific logger for this class
        self.augmentations = get_pe_transforms(config, target=usage)
        if config is not None:
            self.img_depth = config.img_depth

            self.pad_seq_when_shorter = config.pad_seq_when_shorter
            # self.truncate_seq_when_longer_required = config.truncate_seq_when_longer_required

    @dataset_getitem_error_logger
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_folder = row["ImagePath"]

        # load nifti image
        nifti_path = os.listdir(image_folder)
        nifti_path = [
            x for x in nifti_path if x.endswith(".nii.gz") or x.endswith(".nii")
        ][0]
        image_path = os.path.join(image_folder, nifti_path)

        # will load w, h, length image, here c is the axis of slices
        image_sequence = (
            nib.load(image_path).get_fdata().transpose((1, 0, 2))
        )  # Convert to HWL for augmentations
        image_sequence = image_sequence.astype(np.float32)
        h, w, length = source_voxel_shape = image_sequence.shape

        # Z-axis VOI span: could be lung, cervical, lumbar whatever
        zmax, zmin = int(row["zmax"]), int(row["zmin"])
        channels_per_slice = self.config.image_channels
        one_side_channels = channels_per_slice // 2

        most_left, most_right = max(0, zmin - one_side_channels), min(
            length, zmax + one_side_channels
        )
        if most_right - most_left > 512:
            diff = most_right - most_left - 512
            most_left += diff // 2
            most_right -= diff - diff // 2

        # bound the zmin and zmax to the most_left and most_right
        zmin, zmax = max(zmin - most_left, 0), min(
            zmax - most_left, most_right - most_left - 1
        )
        image_sequence = image_sequence[:, :, most_left:most_right]  # HWL

        # Normalize and process the image
        image_sequence = window_normalization(
            image_sequence, self.config.window_center, self.config.window_width
        )
        bbox = row[["xmin", "ymin", "xmax", "ymax"]]
        # set bbox to int:
        bbox = bbox.astype(int)
        _ = crop_and_pad(image_sequence, bbox, return_dict=True)
        image_sequence, xy_boxes = _["cropped_img"], _["bbox"]

        # Preparing the final tensor
        image_sequence_tensor = np.zeros(
            (
                self.img_depth,
                channels_per_slice,
                self.config.img_size,
                self.config.img_size,
            ),
            dtype=np.float32,
        )

        # Determine the center slice indices
        if length < self.img_depth and self.pad_seq_when_shorter:
            z_slices = np.arange(zmin, zmax + 1)
        else:
            z_slices = np.linspace(zmin, zmax, self.img_depth, dtype=np.int16)
            length = self.img_depth

        # Load masks if required
        masks = None
        masks_to_load = self.config.masks
        mask_sequence_tensor = None
        if self.config.load_mask and masks_to_load is not None:
            masks = self.load_masks(
                image_folder, masks_to_load, most_left, most_right, bbox
            )
            mask_sequence_tensor = np.zeros(
                (self.img_depth, 1, self.config.img_size, self.config.img_size),
                dtype=np.float32,
            )

        multi_channel_image = np.zeros(
            (image_sequence.shape[0], image_sequence.shape[1], channels_per_slice),
            dtype=np.float32,
        )  # HWC

        for target_idx, center_idx in enumerate(z_slices):
            # Creating a multi-channel image cube
            slice_indices = np.arange(
                center_idx - one_side_channels, center_idx + one_side_channels + 1
            )
            valid_slices = slice_indices[
                (slice_indices >= 0) & (slice_indices < image_sequence.shape[2])
            ]
            multi_channel_image[:, :, valid_slices - slice_indices[0]] = image_sequence[
                :, :, valid_slices
            ]

            # Post-processing: set the values of the channels not included in valid_slices to zero
            invalid_slices = slice_indices[
                (slice_indices < 0) | (slice_indices >= image_sequence.shape[2])
            ]

            # Apply augmentations if required
            if self.augmentations:
                if masks is not None:
                    transformed = self.augmentations(
                        image=multi_channel_image, mask=masks[:, :, center_idx]
                    )
                    augmented_image = transformed["image"]
                    augmented_mask = transformed["mask"]
                    if len(augmented_mask.shape) == 3:
                        augmented_mask = augmented_mask.transpose((2, 0, 1))
                    mask_sequence_tensor[target_idx, -1, :, :] = augmented_mask
                else:
                    augmented_image = self.augmentations(image=multi_channel_image)[
                        "image"
                    ]

                augmented_image = augmented_image.transpose((2, 0, 1))  # CHW
                image_sequence_tensor[target_idx, :, :, :] = augmented_image

            # another way to fix the invalid slices
            if len(invalid_slices) < channels_per_slice:
                image_sequence_tensor[
                    target_idx, :, :, invalid_slices - slice_indices[0]
                ] = 0

        # concat image_sequence and mask_sequence
        if masks is not None:
            image_sequence_tensor = np.concatenate(
                (image_sequence_tensor, mask_sequence_tensor), axis=1
            )

        return {
            "images": torch.from_numpy(image_sequence_tensor),
            # 'masks': torch.tensor(mask_sequence_tensor, dtype=torch.float32) if mask_sequence_tensor is not None else None, # TODO separate masks and images
            "target": torch.tensor(row[self.config.target_label], dtype=torch.float32),
            "length": torch.tensor(length, dtype=torch.int64),
            "xy_boxes": xy_boxes,
            "z_slices": z_slices,
            "series_id": row["StudyInstanceUID"],
            "input_folder": image_folder,
            "source_voxel_shape": source_voxel_shape,
            "grad_cam_model_list": [],
        }

    def load_masks(self, image_folder, masks_to_load, most_left, most_right, bbox):
        masks = []
        for mask_name in masks_to_load:
            mask_path = os.path.join(
                image_folder, "segmentations", mask_name + ".nii.gz"
            )
            mask = nib.load(mask_path).get_fdata().transpose((1, 0, 2))
            mask = mask.astype(np.float32)
            mask = mask[:, :, most_left:most_right]
            mask = crop_and_pad(mask, bbox)
            masks.append(mask)

        # sum the masks and clip the values to 1
        masks = np.clip(np.sum(masks, axis=0), 0, 1)

        return masks
