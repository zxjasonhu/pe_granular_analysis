import os
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import nibabel as nib
from explainer.utils.basic import determine_cam_save_name
from utils.base import format_input_path
from preprocessing.seg_utils.utils import pseudo_pulmonary_artery_region


def convert_to_rgb(image):
    return np.stack((image,) * 3, axis=-1)


def _normalize(v):
    v = np.float32(v)
    v = v - np.min(v)
    v = v / (1e-7 + np.max(v))
    return v


def overlap_cam_on_voxel(
    voxel: np.ndarray,
    cam: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    image_weight: float = 0.5,
    _2rgb: bool = False,
) -> np.ndarray:
    d, h, w = voxel.shape
    overlapped = np.zeros((d, h, w, 3))
    voxel = _normalize(voxel)
    for i in range(d):
        _img = convert_to_rgb(voxel[i])
        _cam = cam[i]
        overlapped[i] = overlap_cam_on_image(_img, _cam, colormap, image_weight, _2rgb)

    overlapped = _normalize(overlapped)
    overlapped = np.uint8(255 * overlapped)
    return overlapped


def overlap_cam_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    image_weight: float = 0.5,
    _2rgb: bool = False,
) -> np.ndarray:
    # convert img and mask to 8UC3
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if _2rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    cam = (1 - image_weight) * heatmap + image_weight * img
    return cam


def convert_cam_to_heatmap(
    cam: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    _2rgb: bool = False,
) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    if _2rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    heatmap /= np.max(heatmap) * 1.15

    return np.uint8(255 * heatmap)


def display(voxel: np.ndarray, cam: np.ndarray):
    print("voxel.shape", voxel.shape)
    print("cam.shape", cam.shape)
    slice_length = voxel.shape[0]

    for i in range(slice_length):
        rgb_img = convert_to_rgb(voxel[i])
        grayscale_cam_ = cam[i]
        rgb_img = rgb_img - np.min(rgb_img)
        rgb_img = rgb_img / np.max(rgb_img)
        cam_image = show_cam_on_image(
            rgb_img, 1 - grayscale_cam_, use_rgb=True
        )  # the blue the higher -> because of the cvtColor BGR to RGB in show_cam_on_image
        cv2.imshow("cam", cam_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

def visualize_suspicious_pe_region(work_dir, pid, study_uid, series_uid, test_name="cam_test"):
    # load nii.gz
    case_intermediate_output = format_input_path(pid, study_uid, series_uid)
    grad_cam_output = os.path.join(work_dir, "grad_cam_outputs", test_name, determine_cam_save_name(pid, study_uid, series_uid))

    cam_data = np.load(grad_cam_output)
    _cam = cam_data["cam"]

    available_nii_gz_files = os.listdir(os.path.join(
        work_dir,
        format_input_path(pid, study_uid, series_uid)
    ))
    first_nii_gz = available_nii_gz_files[0]

    _img = nib.load(os.path.join(
        work_dir,
        f"{case_intermediate_output}{first_nii_gz}"
    ))
    _img_data = _img.get_fdata().transpose(2, 1, 0)

    _lung_mask = nib.load(os.path.join(
        work_dir,
        f"{case_intermediate_output}segmentations/combined_lung.nii.gz"
    ))
    _lung_mask_data = _lung_mask.get_fdata().transpose(2, 1, 0)

    # # Pseudo pulmonary artery region
    # # If you have pulmonary_artery_mask, you can directly use it instead of pseudo
    # # Here we show how to generate pseudo pulmonary artery region using lung and heart masks
    # cs_heart = CustomSegmentator(task="heart", device="cuda")
    # segment_msg_heart = segmentator_process(
    #     segmentator_instance=cs_heart,
    #     input_folder=_output_dir,
    #     pid=_pid,
    #     study_uid=_study_uid,
    #     series_uid=_series_uid,
    # )
    # if segment_msg_heart is not None:
    #     print(f"Segmentation message: {segment_msg_heart}")
    # else:
    #     print("Heart Segmentation successful")
    # _heart_mask = nib.load(os.path.join(
    #     work_dir,
    #     f"{case_intermediate_output}segmentations/heart.nii.gz"
    # ))
    # _heart_mask_data = _heart_mask.get_fdata().transpose(2, 1, 0)
    # pseudo_pulmonary_artery = pseudo_pulmonary_artery_region(lung_mask=_lung_mask_data, heart_mask=_heart_mask_data)

    # If you do not have heart mask, you can also generate pseudo pulmonary artery region using only lung mask via hard-coded rules
    pseudo_pulmonary_artery = pseudo_pulmonary_artery_region(lung_mask=_lung_mask_data)
    _mask_data = (_lung_mask_data > 0) | (pseudo_pulmonary_artery > 0)

    _cam = _cam * _mask_data
    _cam[_cam < 0.3] = 0
    _cam -= np.min(_cam)
    _cam /= (np.max(_cam) - np.min(_cam) + 1e-8)
    _cam = _cam ** 2
    _overlap = overlap_cam_on_voxel(voxel=_img_data, cam=_cam, _2rgb=True)
    return {
        'voxel': _img_data,
        'cam': _cam,
        'overlap': _overlap
    }