import os

current_working_dir = os.getcwd()
# get the path to directory "pe_granular_analysis"
_path = current_working_dir[:current_working_dir.find("pe_granular_analysis")+len("pe_granular_analysis")]

TRAINING_DATA_NAME = "sample_train.csv"
TRAINING_DATA_NAME_SLICE = "sample_train_slices.csv"
TRAINING_DATA_NAME_COMPLETED = "sample_train_completed.csv"
RSNA_DATA_LOCATION = os.path.join(_path, "data")
NIFTI_FORMATTED_DATA_LOCATION = os.path.join(_path, "nifti_data")