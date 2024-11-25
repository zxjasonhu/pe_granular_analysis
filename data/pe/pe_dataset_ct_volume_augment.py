import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from data.pe.pe_dataset_ct_volume import PECTVolumeDataset
from data.base_dataset import dataset_getitem_error_logger
from utils.image_operations import window_normalization, crop_and_pad


class PECTVolumeAugmentDataset(PECTVolumeDataset):
    """
    This dataset is used for 3D CT volume classification.
    The same augmentations are applied to the whole volume.
    """

    def __init__(self, dataframe: pd.DataFrame, augmentations=None, config=None):
        super(PECTVolumeAugmentDataset, self).__init__(dataframe, augmentations, config)

    @dataset_getitem_error_logger
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_folder = row["ImagePath"]

        # load nifti image
        nifti_path = os.listdir(image_folder)
        nifti_path = [x for x in nifti_path if x.endswith(".nii.gz")][0]
        image_path = os.path.join(image_folder, nifti_path)

        # will load w, h, length image, here c is the axis of slices
        image_sequence = (
            nib.load(image_path).get_fdata().transpose((1, 0, 2))
        )  # Convert to HWL for augmentations
        image_sequence = image_sequence.astype(np.float32)
        h, w, length = image_sequence.shape

        image_sequence_tensor, y, length_tensor = self.augment_3d_ct_volume(
            row, image_sequence, length
        )
        return image_sequence_tensor, y, length_tensor

    def augment_3d_ct_volume(self, row, image_sequence, length):
        """
        Augment a 3D CT volume. augmentation unit is a 3D cube
        return a 4D tensor with shape (img_depth, channels, img_size, img_size)
        """
        # Z-axis VOI span: could be lung, cervical, lumbar whatever
        zmax = row["zmax"]
        zmin = row["zmin"]

        # find the buffer region
        channels_per_slice = self.config.channels
        one_side_channels = channels_per_slice // 2
        center_channel = one_side_channels

        most_left = max(0, zmin - one_side_channels)
        most_right = min(length, zmax + one_side_channels)

        # limit max length to 512:
        # cv2.Mat can only have 512 channels at most
        if most_right - most_left > 512:
            diff = most_right - most_left - 512
            left_offset = diff // 2
            right_offset = diff - left_offset
            most_left += left_offset
            most_right -= right_offset

        # offset the zmin and zmax
        # most_left may larger than zmin, most_right may smaller than zmax
        # that may lead to following process out of bound
        zmin = max(zmin - most_left, 0)
        zmax = min(zmax - most_left, most_right - most_left - 1)

        image_sequence = image_sequence[:, :, most_left:most_right]
        length = zmax - zmin + 1

        # Normalize the image
        image_sequence = window_normalization(
            image_sequence, self.config.window_center, self.config.window_width
        )

        # Crop and pad the image (and mask if it's loaded)
        xmin, ymin, xmax, ymax = row[["xmin", "ymin", "xmax", "ymax"]]  # .iloc[0]
        image_sequence = crop_and_pad(
            image_sequence,
            (xmin, ymin, xmax, ymax),
            bbox_enlarge_factor=self.config.xy_plane_enlarge_factor,
        )

        image_sequence = image_sequence.astype(np.float32)
        new_image_sequence = np.zeros(
            (self.config.img_size, self.config.img_size, image_sequence.shape[2])
        )

        if self.augmentations:
            for i in range(image_sequence.shape[2]):
                new_image_sequence[:, :, i] = self.augmentations(
                    image=image_sequence[:, :, i]
                )["image"]
            # image_sequence = self.augmentations(image=image_sequence)["image"]

        image_sequence = new_image_sequence.transpose(
            (2, 0, 1)
        )  # Convert back to LHW for PyTorch

        image_sequence_placeholder = np.zeros(
            (
                self.img_depth,
                channels_per_slice,
                self.config.img_size,
                self.config.img_size,
            )
        )

        if length < self.img_depth and self.pad_seq_when_shorter:
            center_slice_indices = np.arange(zmin, zmax + 1)
        else:
            center_slice_indices = np.linspace(
                zmin, zmax, self.img_depth, dtype=np.int16
            )
            length = self.img_depth

        # Fill the image_sequence_placeholder
        target_slice_indices = np.arange(0, length)

        # Handle center channel
        valid_center = (center_slice_indices >= 0) & (
            center_slice_indices < image_sequence.shape[0]
        )
        image_sequence_placeholder[
            target_slice_indices[valid_center], center_channel, :, :
        ] = image_sequence[center_slice_indices[valid_center], :, :]
        # image_sequence_placeholder[target_slice_indices[~valid_center], center_channel, :, :] = 0  # Pad with zeros

        for i in range(1, one_side_channels + 1):
            left_index = center_slice_indices - i
            right_index = center_slice_indices + i

            # Handle left channel
            valid_left = (left_index >= 0) & (left_index < image_sequence.shape[0])
            image_sequence_placeholder[
                target_slice_indices[valid_left], center_channel - i, :, :
            ] = image_sequence[left_index[valid_left], :, :]
            # image_sequence_placeholder[target_slice_indices[~valid_left], center_channel - i, :, :] = 0  # Pad with zeros

            # Handle right channel
            valid_right = (right_index >= 0) & (right_index < image_sequence.shape[0])
            image_sequence_placeholder[
                target_slice_indices[valid_right], center_channel + i, :, :
            ] = image_sequence[right_index[valid_right], :, :]
            # image_sequence_placeholder[target_slice_indices[~valid_right], center_channel + i, :, :] = 0  # Pad with zeros

        image_sequence_tensor = torch.tensor(
            image_sequence_placeholder, dtype=torch.float32
        )
        y = torch.tensor(row[self.config.target_label], dtype=torch.float32)
        length_tensor = torch.tensor(length, dtype=torch.int64)

        return {"images": image_sequence_tensor, "target": y, "length": length_tensor}
