import logging
import os


class InferenceLogger:
    def __init__(self, log_dir, trail_id, test_id, fold):
        self.log_dir = log_dir
        self.test_id = test_id

        # Create directory for logs if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        self.filename = f"{trail_id}.Inference_log_fold_{fold}.log"

        # Create a unique logger for each fold
        self.logger = logging.getLogger(f"InferenceLogger_Fold{fold}")
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

    def log_message(self, message):
        self.logger.info(message)

    def finalize(self, metrics: dict):
        self.log_message(f"Test {self.test_id} Inference completed.")
        for key, value in metrics.items():
            self.log_message(f"{key}: {value}")

    def close(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
