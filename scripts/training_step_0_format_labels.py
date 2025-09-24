import pandas as pd

import sys
import os
current_working_dir = os.getcwd()
# get the path to directory "pe_granular_analysis"
_path = current_working_dir[:current_working_dir.find("pe_granular_analysis")+len("pe_granular_analysis")]
if _path not in sys.path:
    sys.path.append(_path)

from tqdm.auto import tqdm

from ENVS import TRAINING_DATA_NAME, RSNA_DATA_LOCATION, NIFTI_FORMATTED_DATA_LOCATION

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(RSNA_DATA_LOCATION, "train.csv")) # this will load 
    df = df.drop_duplicates(subset=["SeriesInstanceUID"])
    df = df.reset_index(drop=True)
    df["pe_present_in_exam"] = 1 - df["negative_exam_for_pe"]
    df = df.drop(columns=["negative_exam_for_pe"])

    if not os.path.exists(NIFTI_FORMATTED_DATA_LOCATION):
        os.makedirs(NIFTI_FORMATTED_DATA_LOCATION)
        print(f"Created directory: {NIFTI_FORMATTED_DATA_LOCATION}")
    
    df.to_csv(os.path.join(NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME), index=False)
    print(f"Formatted labels saved to {os.path.join(NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME)}")
