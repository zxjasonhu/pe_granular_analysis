import os

import numpy as np

import nibabel as nib

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm


def nitfti2npz(patient_id, nifti_path, output_path):
    """
    Convert a NIfTI file to an HDF5 file.

    Args:
        patient_id (str): Identifier for the patient.
        study_id (str): Identifier for the study.
        nifti_path (str): Path to the input NIfTI file.
        npz_path (str): Path to the output HDF5 file.
    """
    # Load the NIfTI file
    if not nifti_path.endswith(".gz") and not nifti_path.endswith(".nii"):
        if not os.path.exists(nifti_path):
            return f"Error: The NIfTI file {nifti_path} does not exist."
        
        nifti_file = os.listdir(nifti_path)
        nifti_file = [
            f for f in nifti_file if f.endswith(".nii") or f.endswith(".nii.gz")
        ]
        if len(nifti_file) == 0:
            print("No NIfTI files found in {}".format(nifti_path))
            return f"Error: No NIfTI files found in {nifti_path}."
        elif len(nifti_file) > 1:
            print(
                f"Warning: More than one NIfTI file found in {nifti_path}, using the first one."
            )
        nifti_path = os.path.join(nifti_path, nifti_file[0])

    _nii = nib.load(nifti_path)
    image = (
        _nii.get_fdata().transpose((1, 0, 2)).astype(np.int16)
    )  # Convert to HWL for augmentations
    affine_matrix = _nii.affine

    output_path = os.path.join(output_path, patient_id + ".npz")
    # Save the image and affine matrix to a .npz file
    np.savez(
        output_path,
        image=image,
        affine_matrix=affine_matrix,
    )


def batch_nitfti2npz(df, output_folder, logger=None):
    """
    Batch convert NIfTI files to NPZ format.

    Args:
        df (pd.DataFrame): DataFrame containing patient IDs and NIfTI paths.
        source_folder (str): Path to the folder containing NIfTI files.
        output_folder (str): Path to the folder where NPZ files will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with ProcessPoolExecutor(max_workers=16) as executor:
        # Initialize futures list
        futures = []

        unique_series = df.drop_duplicates(subset=["StudyInstanceUID"])  # .iloc[:1]
        # patientID", "StudyInstanceUID", "SeriesInstanceUID

        # Submit tasks to the executor
        for _, row in unique_series.iterrows():
            pid = row["PatientID"]
            nifti_path = row['image_folder']
            # Submit the processing function to the executor
            futures.append(
                executor.submit(
                    nitfti2npz,
                    pid,
                    nifti_path,
                    output_folder,
                )
            )

        # Process as tasks complete
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Nifti2NPZ Progress"):
            # Get result or exception from future if needed
            error = future.result()
            if error is not None:
                if logger is not None:
                    logger.error(f"Error processing {error}")

        # close the executor
        executor.shutdown(wait=True)