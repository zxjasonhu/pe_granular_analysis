import pydicom
import os


def save_dicom_with_uids(
    dicom_files_list, output_folders, study_uid=None, series_uid=None, patient_id=None
):
    """
    Save a list of DICOM files to a folder.
    :param dicom_files_list: The list of list of DICOM files to save. [[list 1], [list 2], [list 3], ...]
    :param output_folders: The folder to save the DICOM files to.
    """
    for folder in output_folders:
        if not os.path.exists(folder):
            print(f"Creating folder: {folder}")
            os.makedirs(folder)

    if study_uid is None:
        study_uid = pydicom.uid.generate_uid()
    if series_uid is None:
        series_uid = pydicom.uid.generate_uid()[:32]
    if patient_id is None:
        patient_id = "Patient_" + pydicom.uid.generate_uid()[-5:]

    length = len(dicom_files_list[0])
    for dfl in dicom_files_list:
        assert len(dfl) == length, "All dicom_files_list should have the same length"

    for i in range(length):
        sop_instance_uid = pydicom.uid.generate_uid()[:32]
        for j, dfl in enumerate(dicom_files_list):
            dfl[i].SOPInstanceUID = sop_instance_uid
            dfl[i].SeriesInstanceUID = series_uid + f".{10 ** j}"
            dfl[i].StudyInstanceUID = study_uid
            dfl[i].PatientID = patient_id
            dfl[i].SeriesNumber = j + 1
            dfl[i].save_as(os.path.join(output_folders[j], f"{i}.dcm"))

    return {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "patient_id": patient_id,
    }
