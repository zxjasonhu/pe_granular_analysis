import os

import pandas as pd

import dicom2nifti
import pydicom

from typing import Optional

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

from utils.base import format_input_path

import logging

log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, "nifti_conversion.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nifti_conversion")


def convert_dicom_to_volume_with_meta_data(
    input_folder: str,
    output_folder: str,
    pid: Optional[str],
    study_uid: Optional[str],
    series_uid: Optional[str],
    require_meta_data: bool = False,
) -> Optional[str]:
    suffix = format_input_path(pid, study_uid, series_uid)

    dicom_folder = os.path.join(input_folder, suffix)

    # Check if DICOM folder exists
    if not os.path.exists(dicom_folder):
        print(f"The folder {dicom_folder} does not exist.")
        logger.error(f"The folder {dicom_folder} does not exist.")
        return

    # Check or create output folder
    output_series_folder = os.path.join(output_folder, suffix)
    if not os.path.exists(output_series_folder):
        # print(f"Create folder: {output_series_folder}")
        os.makedirs(output_series_folder)

    try:
        # DICOM:LPS
        # dicom2nifti -> LAS
        # NIFTI: RAS
        # so here is no need to reorient
        dicom2nifti.convert_directory(
            dicom_folder, output_series_folder, compression=True, reorient=False
        )  # same as dicom for easier mapping back; Total segmentator will handle the reorientation when inferencing.
    except Exception as e:
        print(f"Error converting {dicom_folder}: {e}")
        logger.error(f"Error converting {dicom_folder}: {e}")
        return study_uid

    if not require_meta_data:
        return None

    # Read DICOM files
    slices = [
        pydicom.dcmread(os.path.join(dicom_folder, s)) for s in os.listdir(dicom_folder)
    ]
    slices.sort(key=lambda x: x.ImagePositionPatient[2])
    sop_to_index = {s.SOPInstanceUID: i for i, s in enumerate(slices)}
    # Convert to DataFrame
    sop_index_df = pd.DataFrame(
        list(sop_to_index.items()), columns=["SOPInstanceUID", "slice_index"]
    )
    sop_index_df.to_csv(
        output_series_folder + f"/{study_uid}_sop_slice.csv", index=False
    )
    return None


def batch_convert_dicom_to_volume(
    df: pd.DataFrame, source_folder: str, output_folder: str, require_meta_data: bool = False
) -> Optional[list]:
    """
    Convert DICOM to NIfTI in batch mode.
    :param df: needs to have columns: PatientID, StudyInstanceUID, SeriesInstanceUID
    :param source_folder:
    :param output_folder:
    :return:
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("Create Output folder: ", output_folder)

    with ProcessPoolExecutor(max_workers=16) as executor:
        # Initialize futures list
        futures = []

        unique_series = df.drop_duplicates(subset=["StudyInstanceUID"])  # .iloc[:1]
        # patientID", "StudyInstanceUID", "SeriesInstanceUID
        # Initialize progress bar
        progress_bar = tqdm(
            total=unique_series.shape[0], desc="Total Progress", lock_args=(True,)
        )  # lock_args to ensure thread-safety

        errors = []

        # Submit tasks to the executor
        for _, row in unique_series.iterrows():
            pid = row["PatientID"] if "PatientID" in row else "None"
            study_uid = row["StudyInstanceUID"] if "StudyInstanceUID" in row else "None"
            series_uid = row["SeriesInstanceUID"] if "SeriesInstanceUID" in row else "None"
            # Submit the processing function to the executor
            futures.append(
                executor.submit(
                    convert_dicom_to_volume_with_meta_data,
                    source_folder,
                    output_folder,
                    pid,
                    study_uid,
                    series_uid,
                    require_meta_data,
                )
            )

        # Process as tasks complete
        for future in as_completed(futures):
            # Get result or exception from future if needed
            error = future.result()
            if error is not None:
                errors.append(error)

            # Update the progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

    return errors
