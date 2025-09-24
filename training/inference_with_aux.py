from analysis.binary_metrics import calculate_np_metrics
from configs.base_cfg import Config
from training.inference import Inferencer

import torch
import torch.nn as nn
from torch.utils.data import Dataset


from tqdm import tqdm


class InferencerWithAuxLoss(Inferencer):
    def __init__(
        self,
        model: nn.Module,
        test_dataset: Dataset,
        config: Config,
        fold: int = 0,
        ind_test: int = 0,
        debug: bool = False,
    ):
        super().__init__(model, test_dataset, config, fold, ind_test, debug)

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

                outputs = self.model(images)

                if isinstance(outputs, tuple):
                    outputs = outputs[0].detach().sigmoid()
                else:
                    outputs = outputs.detach().sigmoid()

                if self.config.require_flip_inference:
                    flip_outputs = self.model(torch.flip(images, dims=(-1,)))
                    if isinstance(flip_outputs, tuple):
                        flip_outputs = flip_outputs[0].detach().sigmoid()
                    else:
                        flip_outputs = flip_outputs.detach().sigmoid()
                    outputs += flip_outputs
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
