import pandas as pd

import sys
import os
current_working_dir = os.getcwd()
# get the path to directory "pe_granular_analysis"
_path = current_working_dir[:current_working_dir.find("pe_granular_analysis")+len("pe_granular_analysis")]
if _path not in sys.path:
    sys.path.append(_path)

from tqdm.auto import tqdm

from ENVS import TRAINING_DATA_NAME, RSNA_DATA_LOCATION, NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME_SLICE


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME))
    # Initialize an empty list to store individual dataframes
    annotations_dfs = []

    # Initialize an empty list to store the study and series IDs without annotations
    missing_annotations = []

    # Loop through each study and series to aggregate annotations
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sop_file_path = os.path.join(NIFTI_FORMATTED_DATA_LOCATION, f"{row['StudyInstanceUID']}/{row['SeriesInstanceUID']}/{row['StudyInstanceUID']}_sop_slice.csv")
       
        # Check if the annotations with path file exists
        if os.path.exists(sop_file_path):
            # Read the annotations and add to the list
            series_annotations_df = pd.read_csv(sop_file_path, dtype={'SOPInstanceUID': str})
            series_annotations_df["StudyInstanceUID"] = row["StudyInstanceUID"]
            series_annotations_df["SeriesInstanceUID"] = row["SeriesInstanceUID"]
            annotations_dfs.append(series_annotations_df)
        else:
            print(f"Warning: Annotations file not found for StudyInstanceUID {row['StudyInstanceUID']} and SeriesInstanceUID {row['SeriesInstanceUID']}")
            # Record the missing annotations
            missing_annotations.append({'StudyInstanceUID': row["StudyInstanceUID"], 'SeriesInstanceUID': row["SeriesInstanceUID"]})

    # Concatenate all the annotations into a single DataFrame
    final_annotations_df = pd.concat(annotations_dfs, ignore_index=True)

    source_slice_labels = pd.read_csv(os.path.join(RSNA_DATA_LOCATION, "train.csv"), dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str, 'SOPInstanceUID': str})[["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "pe_present_on_image"]]
    final_annotations_df = final_annotations_df.merge(source_slice_labels, on=["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"], how="left")
    final_annotations_df.to_csv(os.path.join(NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME_SLICE), index=False)
    print(f"Slice-level labels saved to {os.path.join(NIFTI_FORMATTED_DATA_LOCATION, TRAINING_DATA_NAME_SLICE)}")
   
    # Convert the missing annotations list to a DataFrame
    missing_annotations_df = pd.DataFrame(missing_annotations)
    if not missing_annotations_df.empty:
        print("The following StudyInstanceUID and SeriesInstanceUID combinations are missing annotations:")
        print(missing_annotations_df)
        missing_annotations_df.to_csv(os.path.join(NIFTI_FORMATTED_DATA_LOCATION, "missing_annotations.csv"), index=False)
    else:
        print("All StudyInstanceUID and SeriesInstanceUID combinations have corresponding annotations.")