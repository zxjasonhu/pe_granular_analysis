import logging
import os

import matplotlib.pyplot as plt


class TrainingLogger:
    def __init__(self, log_dir, trail_id, fold):
        self.log_dir = log_dir
        self.train_losses = []
        self.val_losses = []
        self.metrics = {}

        # Create directory for logs if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        self.filename = f"{trail_id}.training_log_fold_{fold}.log"

        # Create a unique logger for each fold
        self.logger = logging.getLogger(f"TrainingLogger_Fold{fold}")
        self.logger.setLevel(logging.INFO)

        # Create handlers
        file_handler = logging.FileHandler(os.path.join(log_dir, self.filename))
        # console_handler = logging.StreamHandler()

        # Create formatters and add it to handlers
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        # console_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        # self.logger.addHandler(console_handler)

    def save_train_val_loss(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.log_message(
            f"Epoch: {len(self.train_losses)}, Training Loss: {train_loss}, Validation Loss: {val_loss}"
        )

    def plot_train_val_curve(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss Curves")
        plt.grid(True)

        plot_filename = f"loss_curve_{self.filename}.png"
        plt.savefig(os.path.join(self.log_dir, plot_filename))
        plt.close()  # Close the figure to prevent it from displaying in notebooks

    def update_curves_and_loss(self, train_loss, val_loss):
        self.save_train_val_loss(train_loss, val_loss)
        self.plot_train_val_curve()

    def update_metrics(self, metrics: dict):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def log_message(self, message):
        self.logger.info(message)

    def update_validation_metrics(self, metrics: dict):
        for key, value in metrics.items():
            self.log_message(f"{key}: {value}")

    def finalize(self):
        self.plot_train_val_curve()
        self.log_message("Training completed.")
        self.log_message(
            f"Best Training Loss: {min(self.train_losses)}, Epoch: {self.train_losses.index(min(self.train_losses))}"
        )
        self.log_message(
            f"Best Validation Loss: {min(self.val_losses)}, Epoch: {self.val_losses.index(min(self.val_losses))}"
        )
        for key, value in self.metrics.items():
            self.log_message(
                f"Best {key}: {max(value)}, Epoch: {value.index(max(value))}"
            )

    def close(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
