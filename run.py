import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" # if you have multiple GPUs, you can specify which one to use

from configs.pe.pe_final import PELabelMask
from utils.base import setup_trail

trail_id = "your_trail_id"

cfg = setup_trail(working_dir="./experiments",
                  trail_id=trail_id,
                  config=PELabelMask())

# manual modification of cfg setup:
cfg.model_name = "coatnet"
cfg.masked_label_portion = 0.8

from training.pipeline import pipeline
from training.train_with_aux_mask import TrainerWithAuxLossMask as Trainer
from training.inference_with_aux import InferencerWithAuxLoss as Inferencer

if __name__ == "__main__":
    pipeline(cfg, Trainer, Inferencer, debug=False)