import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from analysis.binary_metrics import calculate_np_metrics
from configs.base_cfg import Config

from nnlog.inference_logger import InferenceLogger

from torch.nn.parallel import DistributedDataParallel as DDP


class Inferencer:
    def __init__(
        self,
        model: nn.Module,
        test_dataset: Dataset,
        config: Config,
        fold: int = 0,
        ind_test: int = 0,
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
        self.ddp = config.ddp
        if self.ddp:
            self.rank = int(os.environ["LOCAL_RANK"])

            self.device = torch.device(
                f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.result_output_folder = os.path.join(
            self.config.test_result_dir, f"test_id_{ind_test}"
        )
        if not os.path.exists(self.result_output_folder):
            os.makedirs(self.result_output_folder)

        # setup dataloader, specific to ddp
        if self.ddp:
            # self.world_size = int(os.environ["WORLD_SIZE"])
            self.model.to(self.device)
            self.model.eval()
            # self.model = DDP(
            #     self.model,
            #     device_ids=[self.rank],
            #     output_device=self.rank,
            #     broadcast_buffers=False,
            # )
        else:
            self.model.to(self.device)
            model.eval()

        self.test_loader = DataLoader(
            test_dataset,
            num_workers=config.num_workers,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
        )

        self.logger = InferenceLogger(
            self.result_output_folder,
            f"test_{ind_test}." + self.config.trail_id,
            ind_test,
            fold,
        )
        self.ind_test = ind_test
        self.fold = fold
        self.DEBUG = debug

    def inference_all(self):
        progress_bar = tqdm(self.test_loader, desc="Inference", leave=False)
        outputs_list = []
        labels_list = []
        with torch.no_grad():
            for inputs_ in progress_bar:
                images = inputs_["images"]
                labels = inputs_["target"]
                images, labels = images.to(self.device, non_blocking=True), labels.to(
                    self.device, non_blocking=True
                )

                outputs = self.model(images).detach().sigmoid()
                if self.config.require_flip_inference:
                    outputs += (
                        self.model(torch.flip(images, dims=(-1,))).detach().sigmoid()
                    )
                    outputs /= 2

                if self.DEBUG:
                    print(images.shape, labels.shape, outputs.shape)
                    print(outputs)

                outputs_list.append(outputs)
                labels_list.append(labels.detach())

            outputs_list = torch.cat(outputs_list).cpu()
            labels_list = torch.cat(labels_list).cpu()

            metrics = calculate_np_metrics(outputs=outputs_list, labels=labels_list)
            self.logger.finalize(metrics)

        return outputs_list.numpy(), labels_list.numpy()

    def inference_and_save_predictions(self):
        if self.ddp:
            if self.rank != 0:
                return

        # note that this is the original dataframe
        # will add a wrapper for the dataset to get the dataframe
        test_dataset_df = self.test_loader.dataset.dataframe

        y_preds, y_gts = self.inference_all()

        if y_preds.ndim == 2:
            _, n = y_preds.shape
            for i in range(n):
                test_dataset_df[f"pred_{i}"] = y_preds[:, i]
                test_dataset_df[f"gt_{i}"] = y_gts[:, i]
        else:
            test_dataset_df["pred"] = y_preds
            test_dataset_df["gt"] = y_gts

        test_dataset_df.to_csv(
            os.path.join(
                self.result_output_folder,
                f"{self.config.trail_id}_fold_{self.fold}.csv",
            ),
            index=False,
        )

        self.clean()

    def clean(self):
        del self.test_loader
        del self.model
        del self.config
        del self.device
        del self.fold

        self.logger.close()
        del self.logger

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc

        gc.collect()
        print("Inferencer Cleaned.")


if __name__ == "__main__":
    pass
