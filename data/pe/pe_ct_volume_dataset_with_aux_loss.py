import pandas as pd
import torch

from data.pe.pe_dataset_ct_volume import PECTVolumeDataset
from data.base_dataset import dataset_getitem_error_logger


class PECTVolumeDatasetWithAuxLoss(PECTVolumeDataset):
    def __init__(self, dataframe: pd.DataFrame, usage: str = "train", config=None):
        super(PECTVolumeDatasetWithAuxLoss, self).__init__(dataframe, usage, config)

        assert (
            config.aux_loss_dataset_path is not None
        ), "aux_loss_df is not provided in the config"
        assert (
            config.aux_loss_target is not None
        ), "aux_loss_target is not provided in the config"

        self.aux_loss_df = pd.read_csv(config.aux_loss_dataset_path)
        self.aux_loss_target = config.aux_loss_target

    @dataset_getitem_error_logger
    def __getitem__(self, idx):
        if self.usage == "test":
            return super().__getitem__(idx)
        inputs_ = super().__getitem__(idx)
        center_slice_indices = inputs_["center_slice_indices"]

        row = self.dataframe.iloc[idx]
        suid = row["StudyInstanceUID"]

        # Preprocess to find the slice indices with the target value of 1
        local_df = self.aux_loss_df[self.aux_loss_df["StudyInstanceUID"] == suid]
        target_slices_with_aux_target = set(
            local_df[local_df[self.config.aux_loss_target] == 1]["SliceIndex"]
        )
        aux_target_tensor = torch.zeros(len(center_slice_indices), dtype=torch.float32)

        image_channels = self.config.image_channels
        one_side_channels = image_channels // 2

        # Populate the tensor with pe_present_on_image values more efficiently
        for i, slice_index in enumerate(center_slice_indices):
            start_slice_index = max(0, slice_index - one_side_channels)
            end_slice_index = slice_index + one_side_channels + 1

            # Check if any slice in the range has a value of 1 more efficiently
            aux_target_tensor[i] = any(
                slice_idx in target_slices_with_aux_target
                for slice_idx in range(start_slice_index, end_slice_index)
            )

        inputs_[self.config.aux_loss_target] = aux_target_tensor
        return inputs_
