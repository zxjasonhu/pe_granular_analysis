import pandas as pd

import sys
import os
current_working_dir = os.getcwd()
# get the path to directory "pe_granular_analysis"
_path = current_working_dir[:current_working_dir.find("pe_granular_analysis")+len("pe_granular_analysis")]
if _path not in sys.path:
    sys.path.append(_path)

from preprocessing.nifti_conversion import batch_convert_dicom_to_volume

from ENVS import TRAINING_DATA_NAME, RSNA_DATA_LOCATION, NIFTI_FORMATTED_DATA_LOCATION


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(RSNA_DATA_LOCATION, TRAINING_DATA_NAME))
    err_suids = batch_convert_dicom_to_volume(
        df=df,
        source_folder=RSNA_DATA_LOCATION,
        output_folder=NIFTI_FORMATTED_DATA_LOCATION,
        require_meta_data=True
    )
    if len(err_suids) > 0:
        print("Errors occurred during DICOM to volume conversion:")
        for error in err_suids:
            print(error)
    else:
        print("DICOM to volume conversion completed successfully.")