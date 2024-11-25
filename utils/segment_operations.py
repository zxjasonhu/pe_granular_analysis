from typing import Optional

import cv2
import numpy as np

import albumentations as A

from utils.image_operations import crop_and_pad


def process_one_segment_multi_channel(
    coordinates: tuple,
    ct_image: np.ndarray,
    image_channels: int = 1,
    channel_strip: int = 1,
    xy_plane_enlarge_factor: Optional[float] = 0.0,
    z_axis_enlarge_factor: Optional[float] = 0.0,
    mask: Optional[np.ndarray] = None,
    img_depth: Optional[int] = 40,
    pad_seq_when_shorter: Optional[bool] = False,
    augmentations: Optional[A.Compose] = None,
    return_dict: Optional[bool] = False,
):
    """
    Process a single segment of a 3D image.
    :param coordinates:
    :param ct_image:
    :param image_channels:
    :param xy_plane_enlarge_factor:
    :param z_axis_enlarge_factor:
    :param mask:
    :param img_depth: target length of the image sequence
    :param pad_seq_when_shorter:
    :param augmentations:
    :param return_dict:
    :return:
    """
    h, w, length = ct_image.shape
    # Z-axis VOI span: could be lung, cervical, lumbar whatever
    xmin, ymin, zmin, xmax, ymax, zmax = coordinates
    # enlarge zmax and zmin by (zmax - zmin) * 0.2
    zmin = max(0, zmin - int((zmax - zmin) * z_axis_enlarge_factor))
    zmax = min(length, zmax + int((zmax - zmin) * z_axis_enlarge_factor))

    one_side_channels = image_channels // 2

    lowest, highest = max(0, zmin - one_side_channels * channel_strip), min(
        length, zmax + one_side_channels * channel_strip
    )
    if highest - lowest > 512:
        diff = highest - lowest - 512
        lowest += diff // 2
        highest -= diff - diff // 2

    # bound the zmin and zmax to the most_left and most_right
    zmin, zmax = max(zmin, lowest), min(zmax, highest - 1)
    length = zmax - zmin + 1

    # Determine the center slice indices
    if length < img_depth and pad_seq_when_shorter:
        center_slice_indices = np.arange(zmin, zmax + 1)
    else:
        center_slice_indices = np.linspace(zmin, zmax, img_depth, dtype=np.int16)

    multichannel_center_slice_indices = []
    for s in center_slice_indices:
        start = max(lowest, s - channel_strip)
        end = min(highest, s + channel_strip + 1)
        multichannel_center_slice_indices.append(np.arange(start, end, channel_strip))

    bbox = np.array([xmin, ymin, xmax, ymax])
    # set bbox to int:
    bbox = bbox.astype(int)
    image = crop_and_pad(
        ct_image,
        bbox,
        z_indicies=multichannel_center_slice_indices,
        channel_per_slice=image_channels,
        img_depth=img_depth,
        crop_to_square=True,
        bbox_enlarge_factor=xy_plane_enlarge_factor,
        return_dict=return_dict,
    )  # LHWC

    # Load masks if required
    m = None
    if mask is not None:
        m = crop_and_pad(
            mask,
            bbox,
            z_indicies=center_slice_indices,
            img_depth=img_depth,
            crop_to_square=True,
            bbox_enlarge_factor=xy_plane_enlarge_factor,
            return_dict=False,
        ).transpose(
            2, 0, 1
        )  # LHW
        # attach mask to the image
        image = np.concatenate([image, np.expand_dims(m, axis=-1)], axis=-1)

    if return_dict:
        bbox = image["bbox"]
        image = image["cropped_img"]

    # Apply augmentations if required
    if augmentations:
        augmented_images = []
        _l, _h, _w, _c = image.shape
        for i in range(_l):
            _img = image[i]  # HWC
            augmented_image = augmentations(image=_img)["image"]
            augmented_images.append(augmented_image)

        image = np.stack(augmented_images, axis=0)  # LHWC

    # LHWC -> LCHW
    image = image.transpose(0, 3, 1, 2)

    if return_dict:
        # return image, center_slice_indices, bbox
        # Will be used in GradCAM
        return {
            "image": image,
            "center_slice_indices": center_slice_indices,
            "bbox": bbox,
        }

    return image
