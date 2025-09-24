import pandas as pd

import sys
import os
current_working_dir = os.getcwd()
# get the path to directory "pe_granular_analysis"
_path = current_working_dir[:current_working_dir.find("pe_granular_analysis")+len("pe_granular_analysis")]
if _path not in sys.path:
    sys.path.append(_path)

from tqdm.auto import tqdm

from preprocessing.segmentation import segmentation2bbox_batch_process
from preprocessing.dataframe_formatter import add_path2df

from ENVS import TRAINING_DATA_NAME, NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME_COMPLETED

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME))
    df = segmentation2bbox_batch_process(df, task="lung", input_folder=NIFTI_FORMATTED_DATA_LOCATION, save_path=NIFTI_FORMATTED_DATA_LOCATION)
    df = add_path2df(df, path=NIFTI_FORMATTED_DATA_LOCATION)
    df.to_csv(os.path.join(NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME_COMPLETED), index=False)