from typing import Optional
import torch
import torch.nn as nn
from torch import distributed as dist

# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from configs.base_cfg import Config
from analysis.binary_metrics import calculate_np_metrics
from training.train import Trainer
from utils.ddp_utils import sync_across_gpus
from utils.scheduler import ConstantLossWeightScheduler


class TrainerWithAuxLossMask(Trainer):
    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        config: Config,
        optimizer: Optional[str] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        fold: int = 0,
        phase: str = "phase1",
        debug: bool = False,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            fold=fold,
            phase=phase,
            debug=debug,
        )

        if isinstance(config.main_loss_weight, float):
            self.main_loss_weight_scheduler = ConstantLossWeightScheduler(
                weight=config.main_loss_weight
            )
        elif isinstance(config.main_loss_weight, dict):
            if phase == "phase1":
                self.main_loss_weight_scheduler = ConstantLossWeightScheduler(
                    weight=config.main_loss_weight["phase1"]
                )
            elif phase == "phase2":
                self.main_loss_weight_scheduler = ConstantLossWeightScheduler(
                    weight=config.main_loss_weight["phase2"]
                )
            else:
                raise ValueError("phase should be either phase1 or phase2")
        else:
            raise ValueError("main_loss_weight should be either float or dict")

        self.aux_criterion = nn.BCEWithLogitsLoss(reduction="none")

        # self.main_loss_weight_scheduler = LinearLossWeightScheduler(
        #     start_epoch=self.config.num_epochs // 3,
        #     end_epoch=self.config.num_epochs * 2 // 3,
        #     start_weight=0.05,
        #     end_weight=0.95,
        # )

        # self.scheduler = CosineAnnealingWarmRestarts(
        #     self.optimizer,
        #     T_0=45,  # int(config.num_epochs * 0.25) + 1,
        #     T_mult=1,
        #     eta_min=config.eta_min,
        # )

    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss = []
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training, {self.phase} Epoch [{epoch + 1}/{self.config.num_epochs}]",
            leave=False,
        )
        if self.ddp:
            self.train_loader.sampler.set_epoch(epoch)

        main_loss_weight = self.main_loss_weight_scheduler.get_weight(epoch)

        self.optimizer.zero_grad()
        loss_update_freq = self.config.real_batch_size // self.config.batch_size

        for index, inputs_ in enumerate(progress_bar):
            images = inputs_["images"]
            labels = inputs_["target"]
            aux_labels = inputs_[self.config.aux_loss_target]
            images, labels = images.to(self.device, non_blocking=True), labels.to(
                self.device, non_blocking=True
            )
            aux_labels = aux_labels.to(self.device, non_blocking=True)

            selected_aux_label_predictions = None
            if "selected_aux_label_predictions" in inputs_:
                selected_aux_label_predictions = inputs_["selected_aux_label_predictions"]
                selected_aux_label_predictions = selected_aux_label_predictions.to(
                    self.device, non_blocking=True
                )

            # TODO: add mask to the inputs or add length mask to the model
            outputs = self.model(images)

            # FIXME: add support for multi-output models
            if isinstance(outputs, tuple):
                main_loss = self.criterion(outputs[0], labels)
                if "selected_aux_label_predictions" in inputs_:
                    _condition = selected_aux_label_predictions > 0
                    _aux_outputs = outputs[1][_condition]
                    _aux_labels = aux_labels[_condition]
                    aux_loss = self.aux_criterion(_aux_outputs, _aux_labels).mean()
                else:
                    aux_loss = self.criterion(outputs[1], aux_labels).mean()
                loss = main_loss_weight * main_loss + (1 - main_loss_weight) * aux_loss
            else:
                loss = self.criterion(outputs, labels)

            loss /= loss_update_freq
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            if (index + 1) % loss_update_freq == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            local_loss = loss.detach()
            train_loss.append(local_loss)

            progress_bar.set_postfix(loss=local_loss.cpu().numpy())

            if self.debug:
                if isinstance(outputs, tuple):
                    print("output shape: ", outputs[0].shape, outputs[1].shape)
                    print("output", outputs[0])
                else:
                    print("output shape: ", outputs.shape)
                    print("output", outputs)
                print(images.shape, labels.shape)

        train_loss = torch.stack(train_loss)

        if self.ddp:
            train_loss = sync_across_gpus(train_loss, self.world_size)
            if self.debug:
                print("Average Loss: ", train_loss)

            torch.distributed.barrier()
            if dist.get_rank() == 0:
                return train_loss.mean().item()
            else:
                return 0

        return train_loss.mean().item()

    def validate(self, epoch):
        self.model.eval()
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Validation, {self.phase} Epoch [{epoch + 1}/{self.config.num_epochs}]",
            leave=False,
        )
        outputs_list = []
        labels_list = []
        val_loss = []

        main_loss_weight = self.main_loss_weight_scheduler.get_weight(epoch)

        with torch.no_grad():
            for inputs_ in progress_bar:
                images = inputs_["images"]
                labels = inputs_["target"]
                images, labels = images.to(self.device, non_blocking=True), labels.to(
                    self.device, non_blocking=True
                )

                aux_labels = inputs_[self.config.aux_loss_target]
                aux_labels = aux_labels.to(self.device, non_blocking=True)

                selected_aux_label_predictions = None
                if "selected_aux_label_predictions" in inputs_:
                    selected_aux_label_predictions = inputs_[
                        "selected_aux_label_predictions"
                    ]
                    selected_aux_label_predictions = selected_aux_label_predictions.to(
                        self.device, non_blocking=True
                    )

                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    main_loss = self.criterion(outputs[0], labels)
                    if "selected_aux_label_predictions" in inputs_:
                        _condition = selected_aux_label_predictions > 0
                        _aux_outputs = outputs[1][_condition]
                        _aux_labels = aux_labels[_condition]
                        aux_loss = self.aux_criterion(_aux_outputs, _aux_labels).mean()
                    else:
                        aux_loss = self.criterion(outputs[1], aux_labels).mean()
                    loss = (
                        main_loss_weight * main_loss + (1 - main_loss_weight) * aux_loss
                    )
                    main_output = outputs[0].detach().sigmoid()
                else:
                    loss = self.criterion(outputs, labels)
                    main_output = outputs.detach().sigmoid()

                outputs_list.append(main_output)
                labels_list.append(labels.detach())
                val_loss.append(loss.detach())

                progress_bar.set_postfix(loss=loss.detach().cpu().numpy())

        outputs_list = torch.cat(outputs_list, dim=0)
        labels_list = torch.cat(labels_list, dim=0)
        val_loss = torch.stack(val_loss)

        if self.ddp:
            outputs_list = sync_across_gpus(outputs_list, self.world_size)
            labels_list = sync_across_gpus(labels_list, self.world_size)
            val_loss = sync_across_gpus(val_loss, self.world_size)
            torch.distributed.barrier()
            if dist.get_rank() != 0:
                return 0, 0

        outputs_list = outputs_list.cpu()
        labels_list = labels_list.cpu()
        val_loss = val_loss.cpu()
        average_loss = val_loss.mean().item()

        if self.debug:
            print(f"validation output list shape:", outputs_list.shape)
            print(f"validation labels list shape:", labels_list.shape)

        validation_metrics = calculate_np_metrics(outputs_list, labels_list)

        self.logger.update_validation_metrics(validation_metrics)

        return average_loss, validation_metrics
