import os

import SimpleITK as sitk
import pydicom
import warnings

import numpy as np

from explainer.utils.visualize_cam import overlap_cam_on_voxel


def save_cam(result_output_folder, final_out_cam, series_id):
    """
    Save a class activation map (CAM) to a file.
    :param result_output_folder: The folder to save the CAM to.
    :param final_out_cam: The CAM to save.
    :param series_id: The series ID of the CAM.
    """
    if isinstance(series_id, list):
        series_id = series_id[0]
    series_id = series_id.replace("'", "").replace("[", "").replace("]", "")
    np.save(
        os.path.join(result_output_folder, f"cam_{series_id}.npy"),
        final_out_cam,
    )


def read_arr_from_dicom_dir(DICOM_dir, dtype=sitk.sitkInt16):
    reader = sitk.ImageSeriesReader()
    list_series_ids = reader.GetGDCMSeriesIDs(DICOM_dir)

    sum_series = len(list_series_ids)
    if sum_series > 1:
        warnings.warn("Multiple series ids in this dir, only read one series")

    series_uid = list_series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(DICOM_dir, series_uid)
    image_nii = sitk.ReadImage(file_names, dtype)

    return sitk.GetArrayFromImage(image_nii)[::-1, :]


def load_dicom(image_folder):
    # List all files in the folder
    files = os.listdir(image_folder)

    # Filter only DICOM files (typically they have .dcm extension, but not always)
    dicom_files = [
        pydicom.dcmread(os.path.join(image_folder, f))
        for f in files
        if f.endswith(".dcm")
    ]

    # Sort DICOM files by InstanceNumber
    dicom_files = sorted(
        dicom_files,
        key=lambda x: x.ImagePositionPatient[2],  # x.InstanceNumber
        reverse=False,
    )

    return dicom_files


def save_to_dicom(images, dcms, target_folder, _id_str=".1111", series_number=111):
    assert images.shape[0] == len(
        dcms
    ), f"Shape does not match!! {images.shape[0]} !== f{len(dcms)}"

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    _seriesInstanceUID = pydicom.uid.generate_uid()[:56] + _id_str

    for index, (ds) in enumerate(dcms):
        img = images[index]
        ds.PhotometricInterpretation = "RGB"
        ds.SamplesPerPixel = 3
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.add_new(0x00280006, "US", 0)
        ds.is_little_endian = True
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        ds.WindowCenter = 127
        ds.WindowWidth = 255
        ds.SeriesInstanceUID = _seriesInstanceUID
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        if "SeriesNumber" in ds:
            ds.SeriesNumber = int(ds.SeriesNumber) + series_number
        ds.Rows = img.shape[0]
        ds.Columns = img.shape[1]
        ds.fix_meta_info()

        ds.PixelData = img.tobytes()
        ds["PixelData"].is_undefined_length = False

        filename = os.path.join(target_folder, f"{index}.dcm")
        ds.save_as(filename)


def save_cam_to_dicom(
    cam_npy_path,
    dicom_folder,
    output_folder,
    _id_str=".1111",
    series_number=111,
    smooth_cam=True,
):
    """
    Save a class activation map (CAM) to a DICOM file.
    :param cam_npy_path: The path to the CAM where the CAM is saved.
    :param _id_str: The ID string to append to the SeriesInstanceUID and SOPInstanceUID.
    :param series_number The series number to append to the SeriesNumber.
    :param smooth_cam: Whether to smooth the CAM by squaring it.
    :param cam: The CAM to save.
    :param dicom_folder: The folder containing the DICOM files.
    :param output_folder: The folder to save the CAM to.
    """
    dicom_files = load_dicom(dicom_folder)
    scan_cube = np.stack([ds.pixel_array for ds in dicom_files], axis=0)
    cam = np.load(cam_npy_path)
    if smooth_cam:
        cam = cam**2
    overlapped = overlap_cam_on_voxel(scan_cube, cam, _2rgb=True)

    save_to_dicom(
        overlapped,
        dicom_files,
        output_folder,
        _id_str=_id_str,
        series_number=series_number,
    )
