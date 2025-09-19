import os
import time
from random import random
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader

from tqdm import tqdm

from analysis.binary_metrics import calculate_np_metrics  # calculate_metrics
from configs.base_cfg import Config
from dataset.augmentations.mixup import mixup
from nnlog.training_logger import TrainingLogger
from sampler.sampler import (
    get_custom_distributed_weighted_sampler,
)
from utils.ddp_utils import sync_across_gpus


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: Config,
        optimizer: Optional[str] = None,
        scheduler: Optional[LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
        fold: int = 0,
        phase: str = "phase1",
        debug: bool = False,
    ):
        self.ddp = config.ddp

        # setup device
        if self.ddp:
            self.data_parallel = False
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["LOCAL_RANK"])
            self.device = torch.device(
                f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
            )
            print(f"Rank {self.rank} is using device {self.device}.")
        else:
            self.rank = 0
            self.data_parallel = config.data_parallel
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # finalize constant init
        self.phase = phase
        self.fold = fold
        self.debug = debug

        self.early_stopping_patience = config.early_stopping_patience
        self.patient_counter = 0
        self.early_stopping_target = config.early_stopping_target
        self.early_stop_signal = torch.tensor(0, dtype=torch.int, device=self.device)

        # class attribute setup
        self.warmup_epochs = config.warmup_epochs
        self.learning_rate = config.learning_rate
        self.grad_clip = config.clip_grad

        self.config = config
        self.logger = TrainingLogger(self.config.log_dir, self.config.trail_id, fold)
        self.best_val_loss = float("inf")
        self.start_epoch = 0

        # augmentations
        self.apply_mixup = config.apply_mixup

        # init setup model
        self.model = model
        self.model_path = os.path.join(config.model_path, f"fold_{fold}")

        if self.early_stopping_target != "loss":
            self.early_stopping_target_metrics_value = 0

        self.metrics_to_monitor = {}
        for metric in config.metrics_to_monitor:
            self.metrics_to_monitor[metric] = 0

        # setup optimizer
        if optimizer == "SGD":
            self.optimizer = SGD(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=config.weight_decay,
                momentum=config.momentum,
            )
        elif optimizer == "AdamW":
            self.optimizer = AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError("Optimizer not supported.")

        # setup scheduler
        if scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.t_max, eta_min=config.eta_min
            )

        # setup criterion
        if criterion is not None:
            self.criterion = criterion
        elif self.config.criterion == "BCEWithLogitsLoss":
            class_weight = [config.positive_weight]
            class_weights_tensor = torch.tensor(class_weight, dtype=torch.float32).to(
                self.device
            )
            self.criterion = BCEWithLogitsLoss(pos_weight=class_weights_tensor)
        else:
            raise ValueError("Criterion not supported.")
        self.criterion = self.criterion.to(self.device)

        load_model_name = None
        if config.resume_training or debug:
            load_model_name = f"last_model_fold{fold}.pth"
        elif phase == "phase2":
            load_model_name = f"best_model_fold{fold}.pth"

        # Model setup continued - Load checkpoint if provided
        if config.resume_training:
            load_checkpoint_path = os.path.join(self.model_path, load_model_name)
            if not os.path.exists(load_checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint {load_checkpoint_path} not found. Cannot resume training. "
                    f"resume_training check: {config.resume_training}, phase: {phase}, debug: {debug}."
                )

            self.start_epoch = self.load_checkpoint(load_checkpoint_path)

            # update scheduler to start from the correct epoch
            self.scheduler.step(self.start_epoch)

            print(
                f"Resuming training phase: {self.config.phase} from epoch {self.start_epoch}."
            )
        elif phase == "phase2":
            load_checkpoint_path = os.path.join(self.model_path, load_model_name)
            if not os.path.exists(load_checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint {load_checkpoint_path} not found. Cannot resume training. "
                    f"resume_training check: {config.resume_training}, phase: {phase}, debug: {debug}."
                )
            _ = self.load_checkpoint(load_checkpoint_path)
            print(f"Phase 2 training: loading phase 1 from epoch {_}.")
        elif hasattr(config, "model_head_path"):
            if config.model_head_path:
                model_head_path = os.path.join(
                    config.model_head_path, f"fold_{fold}", f"best_model_fold{fold}.pth"
                )
                if os.path.exists(model_head_path):
                    model.load_head(model_head_path)
                    print(f"Load head model from {model_head_path}, transfer learning")
                else:
                    print(
                        f"Head model {model_head_path} not found, training from scratch"
                    )
        if config.unfreeze_head_epochs > 0:
            model.freeze_head()

        if self.data_parallel:
            self.model = DataParallel(
                self.model, device_ids=range(torch.cuda.device_count())
            )
        self.model.to(self.device)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # setup dataloader, specific to ddp
        if self.ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                broadcast_buffers=False,
            )

            if self.config.weight_sampler:
                self.train_sampler = get_custom_distributed_weighted_sampler(
                    dataset=train_dataset,
                    weights=train_dataset.get_weights(debug=debug),
                    replacement=True,
                    num_replicas=None,
                    rank=None,
                    shuffle=True,
                    seed=42,
                    drop_last=False,
                )
            else:
                self.train_sampler = DistributedSampler(train_dataset)
            self.val_sampler = DistributedSampler(val_dataset, shuffle=False)

            self.train_loader = DataLoader(
                dataset=train_dataset,
                sampler=self.train_sampler,
                num_workers=config.num_workers,
                batch_size=config.batch_size,
                pin_memory=True,
            )
            self.val_loader = DataLoader(
                dataset=val_dataset,
                sampler=self.val_sampler,
                num_workers=config.num_workers,
                batch_size=config.batch_size,
                pin_memory=True,
            )
        else:
            if self.config.weight_sampler:
                weights = train_dataset.get_weights(debug=debug)
                self.train_sampler = WeightedRandomSampler(weights, len(weights))
            self.train_loader = DataLoader(
                train_dataset,
                sampler=self.train_sampler if self.config.weight_sampler else None,
                num_workers=config.num_workers,
                batch_size=config.batch_size,
                prefetch_factor=2,
                pin_memory=True,
                shuffle=False if self.config.weight_sampler else True,
            )
            self.val_loader = DataLoader(
                val_dataset,
                num_workers=config.num_workers,
                batch_size=1,
                prefetch_factor=2,
                pin_memory=True,
                shuffle=False,
            )

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

        self.optimizer.zero_grad()
        loss_update_freq = self.config.real_batch_size // self.config.batch_size

        for index, inputs_ in enumerate(progress_bar):
            images = inputs_["images"]
            labels = inputs_["target"]
            images, labels = images.to(self.device, non_blocking=True), labels.to(
                self.device, non_blocking=True
            )

            do_mixup = False
            if self.apply_mixup:
                if random() < 0.35:
                    do_mixup = True
                    images, shuffled_labels, lam = mixup(images, labels)

            # TODO: add mask to the inputs or add length mask to the model
            outputs = self.model(images)

            if do_mixup:
                loss = (1 - lam) * self.criterion(
                    outputs, labels
                ) + lam * self.criterion(outputs, shuffled_labels)
            else:
                loss = self.criterion(outputs, labels)

            local_loss = loss.detach().clone()

            loss /= loss_update_freq  # normalize accumulated grad according to https://unsloth.ai/blog/gradient
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            if (index + 1) % loss_update_freq == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

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

            dist.barrier()
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

        with torch.no_grad():
            for inputs_ in progress_bar:
                images = inputs_["images"]
                labels = inputs_["target"]
                images, labels = images.to(self.device, non_blocking=True), labels.to(
                    self.device, non_blocking=True
                )

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                _outputs = outputs.detach().sigmoid()

                outputs_list.append(_outputs)
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
            dist.barrier()
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

    def save_checkpoint(self, epoch, name=None):
        if self.ddp:
            if self.rank != 0:
                return

        if self.data_parallel:
            model_state_dict = self.model.module.state_dict()
        elif self.ddp:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "phase": self.phase,
        }

        for _metrics in self.metrics_to_monitor:
            checkpoint[_metrics] = self.metrics_to_monitor[_metrics]

        if name is not None:
            model_path = os.path.join(self.model_path, f"{name}_fold{self.fold}.pth")
        else:
            model_path = os.path.join(
                self.model_path, f"last_model_fold{self.fold}.pth"
            )

        torch.save(checkpoint, model_path)

    def load_checkpoint(self, load_checkpoint_path):
        checkpoint = torch.load(load_checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]

        for _metrics in self.metrics_to_monitor:
            if _metrics in checkpoint:
                self.metrics_to_monitor[_metrics] = checkpoint[_metrics]

        return checkpoint["epoch"]

    def train(self):
        if self.debug:
            max_epochs = 2
        else:
            max_epochs = self.config.num_epochs

        for epoch in range(self.start_epoch, max_epochs):
            start_time = time.time()
            # Warmup Learning Rate Adjustment
            if epoch < self.config.warmup_epochs:
                lr = (epoch + 1) / self.warmup_epochs * self.learning_rate
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
            elif epoch == self.config.warmup_epochs:  # First epoch after warmup
                # Reset learning rate to the initial value before the main scheduler takes over
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

            if self.config.unfreeze_head_epochs is not None:
                if epoch == self.config.unfreeze_head_epochs:
                    self.logger.log_message(
                        "Unfreezing head at epoch " + str(epoch) + "."
                    )
                    if self.data_parallel:
                        self.model.module.unfreeze_head()
                    elif self.ddp:
                        self.model.module.unfreeze_head()
                    else:
                        self.model.unfreeze_head()

            # Train and Validate
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_metrics = self.validate(epoch)

            # Step the main scheduler only after warmup period is over
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()

            if self.rank == 0:
                # Early Stopping
                if self.config.init_training_epochs < epoch:
                    if self.early_stopping_patience is not None:
                        if self.early_stopping_target == "loss":
                            if val_loss > self.best_val_loss:
                                self.patient_counter += 1
                            else:
                                self.patient_counter = 0
                        else:
                            if (
                                val_metrics[self.early_stopping_target]
                                < self.early_stopping_target_metrics_value
                            ):
                                self.patient_counter += 1
                            else:
                                self.patient_counter = 0
                        if self.patient_counter >= self.early_stopping_patience:
                            early_stop_message = (
                                f"Early stopping at epoch {epoch + 1}; "
                                f"with best validation loss of {self.best_val_loss:.4f} "
                            )
                            if self.early_stopping_target != "loss":
                                early_stop_message += (
                                    f"and best {self.early_stopping_target} of "
                                    f"{self.early_stopping_target_metrics_value:.4f}"
                                )
                            self.logger.log_message(early_stop_message)
                            self.early_stop_signal += 1

                self.save_checkpoint(epoch)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, name="best_model")

                for _metrics in self.metrics_to_monitor:
                    if val_metrics[_metrics] > self.metrics_to_monitor[_metrics]:
                        self.metrics_to_monitor[_metrics] = val_metrics[_metrics]
                        self.save_checkpoint(epoch, name=f"best_{_metrics}".lower())

                    if self.early_stopping_target != "loss":
                        if (
                            val_metrics[self.early_stopping_target]
                            < self.early_stopping_target_metrics_value
                        ):
                            self.early_stopping_target_metrics_value = val_metrics[
                                self.early_stopping_target
                            ]

                # Logging
                self.logger.log_message(
                    f"Epoch [{epoch + 1}/{self.config.num_epochs}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Validation Loss: {val_loss:.4f}, "
                    f"Time: {time.time() - start_time:.2f}s, "
                    f"ETA: {((time.time() - start_time) * (self.config.num_epochs - epoch - 1)) / 60:.2f}m"
                )
                self.logger.update_curves_and_loss(train_loss, val_loss)
                self.logger.update_metrics(val_metrics)

            if self.ddp:
                # Broadcast the early stopping decision from rank 0 to all other ranks
                dist.broadcast(self.early_stop_signal, src=0)
                dist.barrier()
            if self.early_stop_signal > 0:
                break

        if self.ddp:
            if dist.get_rank() == 0:
                self.logger.finalize()
        else:
            self.logger.finalize()
        self.clean()

    def clean(self):
        del self.model
        del self.train_loader
        del self.val_loader
        del self.optimizer
        del self.scheduler
        del self.criterion
        del self.config
        del self.best_val_loss
        del self.start_epoch
        del self.device
        del self.fold
        del self.debug

        self.logger.close()
        del self.logger

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        import gc

        gc.collect()

        print("Trainer Cleaned.")


if __name__ == "__main__":
    # trainer = Trainer()
    pass
