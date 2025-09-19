import pandas as pd

import sys
import os
current_working_dir = os.getcwd()
# get the path to directory "pe_granular_analysis"
_path = current_working_dir[:current_working_dir.find("pe_granular_analysis")+len("pe_granular_analysis")]
if _path not in sys.path:
    sys.path.append(_path)

from preprocessing.segmentation import segmentator_process, CustomSegmentator
from tqdm.auto import tqdm

from ENVS import TRAINING_DATA_NAME, NIFTI_FORMATTED_DATA_LOCATION

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(_path, "data", TRAINING_DATA_NAME))
    cs = CustomSegmentator(task="lung", device="cuda")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        _pid = row["PatientID"] if "PatientID" in row else None
        _study_uid = row["StudyInstanceUID"] if "StudyInstanceUID" in row else None
        _series_uid = row["SeriesInstanceUID"] if "SeriesInstanceUID" in row else None
        error = segmentator_process(
            segmentator_instance=cs,
            input_folder=NIFTI_FORMATTED_DATA_LOCATION,
            pid=_pid,
            study_uid=_study_uid,
            series_uid=_series_uid,
        )
        if error is not None:
            print(f"Error occurred during segmentation for SeriesInstanceUID: {_series_uid}")