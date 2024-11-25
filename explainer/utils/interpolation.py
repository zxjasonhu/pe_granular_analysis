import math
from typing import List, Dict, Tuple, Optional

import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image

import cv2
from scipy.interpolate import interp1d, interpn


def cam_to_intermediate_cam(
    cam: np.ndarray,
    interval: int,
    length: int = 40,
    h: int = 512,
    w: int = 512,
) -> np.ndarray:
    grayscale_cam_resized = np.zeros((length, h, w))
    cam_length = cam.shape[0]
    frame_interval = cam_length / length
    max_index = cam_length - 1

    for i in range(length):
        # Find the indices of the original frames that we'll interpolate between
        idx1 = int(i / length * frame_interval)
        idx2 = min(idx1 + 1, max_index)

        # Find the interpolation weight
        alpha = (i / length * frame_interval) - idx1

        # Interpolate between the original frames
        cam1 = cv2.resize(cam[idx1], (w, h))
        cam2 = cv2.resize(cam[idx2], (w, h))
        grayscale_cam_resized[i] = cv2.addWeighted(cam1, 1 - alpha, cam2, alpha, 0)

    return grayscale_cam_resized[:interval, :, :]


def naive_overlap_cams(
    cam: np.ndarray, target_length, h: int = 512, w: int = 512
) -> np.ndarray:
    assert cam.shape[0] == 2, f"cam.shape[0]={cam.shape[0]} != 2"

    _cam = np.zeros((target_length, h, w))
    start_idx = target_length - 40
    overlap = 40 - start_idx

    _cam[:start_idx] = cam[0][:start_idx]
    _cam[start_idx:40] = (cam[0][start_idx:40] + cam[1][:overlap]) / 2
    _cam[40:] = cam[1][80 - target_length :]

    return _cam


def _interpolate_cam_on_voxel(
    voxel: np.ndarray, cam: np.ndarray, conversion_info: List[Dict]
) -> np.ndarray:
    assert (
        len(conversion_info) == cam.shape[0]
    ), f"len(conversion_info)={len(conversion_info)} != cam.shape[0]={cam.shape[0]}"

    slice_indicator = np.ones(voxel.shape[0])
    final_mask = np.zeros(voxel.shape)

    # parse conversion info
    for i in range(len(conversion_info)):
        info = conversion_info[i]
        _type = info["type"]
        _total_slice = info["total_slice"]
        _current_index = info["current_index"]
        _z1 = info["z1"]
        _z2 = info["z2"]
        _x1 = info["x1"]
        _x2 = info["x2"]
        _y1 = info["y1"]
        _y2 = info["y2"]
        real_length = _z2 - _z1
        real_height = _y2 - _y1
        real_width = _x2 - _x1

        if _type == "overlap":
            i += 1
            former = cam_to_intermediate_cam(cam[i], 40, 40, real_height, real_width)
            latter = cam_to_intermediate_cam(
                cam[i + 1], 40, 40, real_height, real_width
            )
            current_cam = naive_overlap_cams(
                np.stack([former, latter]), _total_slice, real_height, real_width
            )
            current_cam = cam_to_intermediate_cam(
                current_cam, real_length, real_length, real_height, real_width
            )
        elif _type == "padding":
            slice_indicator[_z1:_z2] = 1
            current_cam = cam_to_intermediate_cam(
                cam[i],
                int(40 * real_length / _total_slice),
                real_length,
                real_height,
                real_width,
            )
        elif _type == "normal":
            slice_indicator[_z1:_z2] = 1
            current_cam = cam_to_intermediate_cam(
                cam[i], real_length, real_length, real_height, real_width
            )
        else:
            continue

        slice_indicator[_z1:_z2] += 1
        # DEBUG use
        # if np.max(current_cam) < 0.5:
        #     current_cam = np.ones_like(current_cam)
        final_mask[_z1:_z2, _y1:_y2, _x1:_x2] += current_cam

    # No need of normalization
    return final_mask


def remove_black_borders(
    cam: np.ndarray, xy_boxes: np.ndarray, input_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Remove black borders from the CAM.
    :param cam:
    :param xy_boxes:
    :param input_shape:
    :return:
    """
    _, ih, iw = input_shape
    xmin, ymin, xmax, ymax = xy_boxes
    crop_width = xmax - xmin + 1
    crop_height = ymax - ymin + 1

    if ih / crop_height - iw / crop_width > 0.03:
        # remove top and bottom black strips
        real_ih = int(iw / crop_width * crop_height)
        real_iymin = int((ih - real_ih) / 2)
        real_iymax = real_iymin + real_ih
        ih = real_ih
        cam = cam[:, real_iymin:real_iymax, :]
    elif iw / crop_width - ih / crop_height > 0.03:
        # remove left and right black strips
        real_iw = int(ih / crop_height * crop_width)
        real_ixmin = int((iw - real_iw) / 2)
        real_ixmax = real_ixmin + real_iw
        iw = real_iw
        cam = cam[:, :, real_ixmin:real_ixmax]

    return cam


def cam_direct_alignment(
    cam: np.ndarray,
    xy_plane_box: np.ndarray,
    z_slices: np.ndarray,
    source_voxel_shape: np.ndarray,
) -> np.ndarray:
    """
    Align the CAM to the source voxel.
    :param cam: The CAM to align.
    :param input_shape: The shape of the input voxel to the model, (id, ih, iw).
    :param z_slices: The z indices of the slices used in input_voxel from the source voxel.
    :param xy_plane_box: The bounding box of the xy plane for the source voxel. (xmin, ymin, xmax, ymax)
    :return:
    """
    sh, sw, sd = source_voxel_shape

    sxmin, symin, sxmax, symax = xy_plane_box
    # off by one because the indices on boundary are inclusive
    if sxmax == sw:
        sxmax -= 1
    if symax == sh:
        symax -= 1
    box_h, box_w = int(symax - symin + 1), int(sxmax - sxmin + 1)

    szmin, szmax = z_slices[0], z_slices[-1]
    if szmax == sd:
        szmax -= 1
    box_d = int(szmax - szmin + 1)  # +1 because indices are inclusive

    full_cam = np.zeros((box_d, box_h, box_w))
    norm_cam = np.ones((box_d, box_h, box_w))
    initialized = []
    for ind in range(len(z_slices)):
        full_cam_ind = z_slices[ind] - szmin
        full_cam[full_cam_ind] += cv2.resize(cam[ind], (box_w, box_h))

        if full_cam_ind not in initialized:
            initialized.append(full_cam_ind)
        else:
            norm_cam[full_cam_ind] += 1

    full_cam /= norm_cam

    missing_slices = set(range(szmin, szmax + 1)) - set(z_slices)
    for ms in missing_slices:
        # find left and right indices
        left_idx = None
        right_idx = None
        # find the left index
        ls = ms - 1
        while left_idx is None:
            if ls in z_slices:
                left_idx = ls
            elif ls < szmin:
                raise ValueError(
                    f"Cannot find the left index for the missing slice {ms}"
                )
            ls -= 1
        # find the right index
        rs = ms + 1
        while right_idx is None:
            if rs in z_slices:
                right_idx = rs
            elif rs > szmax:
                raise ValueError(
                    f"Cannot find the right index for the missing slice {ms}"
                )
            rs += 1
        # interpolate
        left_cam = full_cam[left_idx - szmin]
        right_cam = full_cam[right_idx - szmin]
        alpha = (ms - left_idx) / (right_idx - left_idx)
        full_cam[ms - szmin] = left_cam * (1 - alpha) + right_cam * alpha

    return full_cam


def cam_alignment(
    cam: np.ndarray,
    input_shape: Tuple[int, int, int],
    xy_plane_box: np.ndarray,
    z_slices: np.ndarray,
    source_voxel_shape: np.ndarray,
) -> np.ndarray:
    """
    Align the CAM to the source voxel.
    :param cam: The CAM to align.
    :param input_shape: The shape of the input voxel to the model, (id, ih, iw).
    :param z_slices: The z indices of the slices used in input_voxel from the source voxel.
    :param xy_plane_box: The bounding box of the xy plane for the source voxel. (xmin, ymin, xmax, ymax)
    :return:
    """
    idepth, ih, iw = input_shape
    cam_length, cam_height, cam_width = cam.shape
    sh, sw, sd = source_voxel_shape

    padding = idepth - len(z_slices)
    assert (
        padding >= 0
    ), f"padding={padding} < 0. The number of slices in the source voxel is less than the input voxel."

    sxmin, symin, sxmax, symax = xy_plane_box
    # off by one because the indices on boundary are inclusive
    if sxmax == sw:
        sxmax -= 1
    if symax == sh:
        symax -= 1
    box_h, box_w = int(symax - symin + 1), int(sxmax - sxmin + 1)

    szmin, szmax = z_slices[0], z_slices[-1]
    if szmax == sd:
        szmax -= 1
    box_d = int(szmax - szmin + 1)  # +1 because indices are inclusive

    num_intervals = cam_length  # number of whole intervals. usually will pad front and back. 1/2, 1, ... 1, 1/2
    size_intervals = idepth / num_intervals
    interval_pivots_indices = [
        int(size_intervals / 2 + i * size_intervals - 1) for i in range(num_intervals)
    ]

    # cam to input interpolation
    input_cam = np.zeros((idepth, box_h, box_w))
    # front and back
    for i in range(interval_pivots_indices[0]):
        alpha = 1 - (interval_pivots_indices[0] - i) / size_intervals
        input_cam[i] = cv2.resize(cam[0], (box_w, box_h)) * alpha
    for i in range(interval_pivots_indices[-1], idepth):
        alpha = 1 - (i - interval_pivots_indices[-1]) / size_intervals
        input_cam[i] = cv2.resize(cam[-1], (box_w, box_h)) * alpha

    # pivot
    for i in range(num_intervals):
        input_cam[interval_pivots_indices[i]] = cv2.resize(cam[i], (box_w, box_h))

    # between pivots
    for i in range(num_intervals - 1):
        start = interval_pivots_indices[i]
        end = interval_pivots_indices[i + 1]
        start_cam = input_cam[start]
        end_cam = input_cam[end]
        for j in range(start + 1, end):
            alpha = (j - start) / size_intervals
            input_cam[j] = start_cam * (1 - alpha) + end_cam * alpha

    # input to source interpolation
    if padding > 0:
        input_cam = input_cam[:-padding, :, :]
    full_cam = np.zeros((box_d, box_h, box_w))
    cam_normalizer = np.ones((box_d, box_h, box_w))
    _initialized = []
    for i in range(input_cam.shape[0]):
        full_cam_ind = z_slices[i] - szmin
        full_cam[full_cam_ind] = np.maximum(input_cam[i], full_cam[full_cam_ind])
        # if full_cam_ind not in _initialized:
        #     _initialized.append(full_cam_ind)
        # else:
        #     cam_normalizer[full_cam_ind, :, :] += 1

    # full_cam /= cam_normalizer

    missing_slices = set(range(szmin, szmax + 1)) - set(z_slices)
    for ms in missing_slices:
        # find left and right indices
        left_idx = None
        right_idx = None
        # find the left index
        ls = ms - 1
        while left_idx is None:
            if ls in z_slices:
                left_idx = ls
            elif ls < szmin:
                raise ValueError(
                    f"Cannot find the left index for the missing slice {ms}"
                )
            ls -= 1
        # find the right index
        rs = ms + 1
        while right_idx is None:
            if rs in z_slices:
                right_idx = rs
            elif rs > szmax:
                raise ValueError(
                    f"Cannot find the right index for the missing slice {ms}"
                )
            rs += 1
        # interpolate
        left_cam = full_cam[left_idx - szmin]
        right_cam = full_cam[right_idx - szmin]
        alpha = (ms - left_idx) / (right_idx - left_idx)
        full_cam[ms - szmin] = left_cam * (1 - alpha) + right_cam * alpha

    return full_cam


def cam_alignment_3d(
    cam: np.ndarray,
    input_shape: Tuple[int, int, int],
    xy_plane_box: np.ndarray,
    z_slices: np.ndarray,
) -> np.ndarray:
    """
    Align the CAM to the source voxel.
    :param cam: The CAM to align.
    :param input_shape: The shape of the input voxel to the model, (id, ih, iw).
    :param z_slices: The z indices of the slices used in input_voxel from the source voxel.
    :param xy_plane_box: The bounding box of the xy plane for the source voxel. (xmin, ymin, xmax, ymax)
    :return:
    """
    ### 3D interpolation: Interpolating the CAM to the source directly
    idepth, ih, iw = input_shape
    cam_length, cam_height, cam_width = cam.shape

    padding = idepth - len(z_slices)
    assert (
        padding >= 0
    ), f"padding={padding} < 0. The number of slices in the source voxel is less than the input voxel."

    sxmin, symin, sxmax, symax = xy_plane_box
    box_h, box_w = int(symax - symin + 1), int(sxmax - sxmin + 1)
    szmin, szmax = z_slices[0], z_slices[-1]
    box_d = int(szmax - szmin + 1)  # +1 because indices are inclusive

    if box_w < cam_width or box_h < cam_height:
        for cl in range(cam_length):
            cam[cl] = cv2.resize(cam[cl], (box_w, box_h))
        cam_height, cam_width = cam.shape[1], cam.shape[2]

    sh_to_ch = box_h / cam_height
    sw_to_cw = box_w / cam_width

    num_intervals = cam_length
    size_intervals = idepth / num_intervals
    interval_pivots_indices = [
        int(size_intervals / 2 + i * size_intervals - 1) for i in range(num_intervals)
    ]

    # cam to input interpolation
    # Prepare for 3D interpolation
    x_points = (np.arange(0, cam_width) * sw_to_cw).astype(int)
    y_points = (np.arange(0, cam_height) * sh_to_ch).astype(int)
    z_points = np.array(interval_pivots_indices)
    values = cam

    # Define grid for interpolation
    _x = np.arange(iw)
    _y = np.arange(ih)
    _z = np.arange(idepth)

    z, y, x = np.meshgrid(_z, _y, _x, indexing="ij")

    # Interpolate using `interpn`
    input_cam = interpn(
        points=(z_points, y_points, x_points),  # (z, y, x)
        values=values,
        xi=(z, y, x),
        method="cubic",
        bounds_error=False,
        fill_value=0,
    )

    # input to source interpolation
    input_cam_padding_removed = input_cam[:-padding, :, :]
    full_cam = np.zeros((box_d, box_h, box_w))
    cam_normalizer = np.ones((box_d, box_h, box_w))
    _initialized = []
    for i in range(input_cam_padding_removed.shape[0]):
        full_cam_ind = z_slices[i] - szmin
        full_cam[full_cam_ind] += input_cam_padding_removed[i]
        if full_cam_ind not in _initialized:
            _initialized.append(full_cam_ind)
        else:
            cam_normalizer[full_cam_ind, :, :] += 1

    full_cam /= cam_normalizer

    missing_slices = set(range(szmin, szmax + 1)) - set(z_slices)
    for ms in missing_slices:
        # find left and right indices
        left_idx = None
        right_idx = None
        # find the left index
        ls = ms - 1
        while left_idx is None:
            if ls in z_slices:
                left_idx = ls
            elif ls < szmin:
                raise ValueError(
                    f"Cannot find the left index for the missing slice {ms}"
                )
            ls -= 1
        # find the right index
        rs = ms + 1
        while right_idx is None:
            if rs in z_slices:
                right_idx = rs
            elif rs > szmax:
                raise ValueError(
                    f"Cannot find the right index for the missing slice {ms}"
                )
            rs += 1
        # interpolate
        left_cam = full_cam[left_idx - szmin]
        right_cam = full_cam[right_idx - szmin]
        alpha = (ms - left_idx) / (right_idx - left_idx)
        full_cam[ms - szmin] = left_cam * (1 - alpha) + right_cam * alpha

    return full_cam


def interpolate_cam_on_voxel(
    cam: np.ndarray,
    source_voxel_shape: np.ndarray,
    input_voxel_shape: Tuple[int, int, int],
    z_slices: np.ndarray,
    xy_plane_boxes: np.ndarray,
) -> np.ndarray:
    """
    Interpolate the CAM on the source voxel.
    :param cam: The CAM to interpolate.
    :param source_voxel_shape: The shape of the source voxel. (source means the original voxel shape)
    :param input_voxel_shape: The shape of the input voxel. which is the voxel fed into the model.
    :param z_slices: The z indices of the slices on source voxel for each input voxel.
    :param xy_plane_boxes: The bounding boxes of the xy plane on source voxel for each input voxel.
    :return:
    """
    sh, sw, sd = source_voxel_shape
    idepth, ih, iw = input_voxel_shape
    xmin, ymin, xmax, ymax = xy_plane_boxes

    if len(z_slices.shape) > 1:
        z_slices = z_slices.squeeze(0)
    zmin, zmax = z_slices[0], z_slices[-1]

    cam = remove_black_borders(cam, xy_plane_boxes, input_voxel_shape)

    if cam.shape[0] == idepth:
        aligned_cam = cam_direct_alignment(
            cam, xy_plane_boxes, z_slices, source_voxel_shape
        )
    else:
        aligned_cam = cam_alignment(
            cam, (idepth, ih, iw), xy_plane_boxes, z_slices, source_voxel_shape
        )

    final_cam = np.zeros((sd, sh, sw))
    final_cam[zmin : zmax + 1, ymin : ymax + 1, xmin : xmax + 1] = aligned_cam

    return final_cam
