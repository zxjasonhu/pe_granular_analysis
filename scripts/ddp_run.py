import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1" # if you have multiple GPUs, you can specify which one to use
import sys
current_working_dir = os.getcwd()
# get the path to directory "pe_granular_analysis"
_path = current_working_dir[:current_working_dir.find("pe_granular_analysis")+len("pe_granular_analysis")]
if _path not in sys.path:
    sys.path.append(_path)

from configs.pe.pe_final import PELabelMask
from utils.base import setup_trail

from ENVS import NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME_SLICE, TRAINING_DATA_NAME_COMPLETED

trail_id = "dry_run"

pe_cfg = PELabelMask()
pe_cfg.ddp = True # set to True if you want to use distributed data parallel

# manual modification of cfg setup:
pe_cfg.model_name = "coatnet"
pe_cfg.masked_label_portion = 0.9 # portion of labels to be masked for aux loss; 1 - 0.9 = portion of labels to be used for training the model
pe_cfg.batch_size = 2

# pe_cfg.img_depth = 32  # number of slices per input volume

pe_cfg.folds = 1

pe_cfg.aux_loss_dataset_path = os.path.join(NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME_SLICE) # aux loss dataset usually is the full dataset with slice-level labels

# assign data partitions; if not assigned, the k-fold cross validation will be used and random split will be applied
pe_cfg.training_dataset_path = [os.path.join(NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME_COMPLETED)]
pe_cfg.validation_dataset_path = [os.path.join(NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME_COMPLETED)]
pe_cfg.test_dataset_path = [os.path.join(NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME_COMPLETED)]

cfg = setup_trail(working_dir="./experiments",
                  trail_id=trail_id,
                  config=pe_cfg)


from training.pipeline import pipeline
from training.train_with_aux_mask import TrainerWithAuxLossMask as Trainer
from training.inference_with_aux import InferencerWithAuxLoss as Inferencer

if __name__ == "__main__":
    pipeline(cfg, Trainer, Inferencer, debug=True)