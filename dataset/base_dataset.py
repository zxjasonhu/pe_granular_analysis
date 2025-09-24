import functools
import logging

import pandas as pd
from torch.utils.data import Dataset


def dataset_getitem_error_logger(func):
    @functools.wraps(func)
    def wrapper(self, idx, *args, **kwargs):
        try:
            return func(self, idx, *args, **kwargs)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error when getting item {idx}: {e}", exc_info=True)
            raise

    return wrapper


class CTDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, usage: str = "train", config=None):
        super(CTDataset, self).__init__()
        self.config = config
        self.dataframe = dataframe

        assert usage in ["train", "val", 'test', "inference", "cam"], "usage should be one of ['train', 'val', 'test', 'inference', 'cam']"
        if usage in ["inference", "cam"]:
            print("Inference/cam mode: No labels will be returned.")

        self.usage = usage
        self.augmentations = None

        # Set up a specific logger for this class
        self.logger = logging.getLogger(self.__class__.__name__)
        handler = logging.FileHandler(f"{self.__class__.__name__}.log")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def __len__(self):
        return len(self.dataframe)


if __name__ == "__main__":
    pass
    # ct_dataset = CTDataset()
