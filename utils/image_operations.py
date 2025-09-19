from typing import Optional

import cv2
import numpy as np

import albumentations as A


def window_normalization(img, window_center, window_width):
    """
    Normalize medical image data with windowing operation.

    Args:
    - img (numpy.ndarray): The input image data.
    - window_center (int or float): The center of the window.
    - window_width (int or float): The width of the window.

    Returns:
    - numpy.ndarray: The windowed image.
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2

    # Clip the intensity values to the specified window
    windowed_img = np.clip(img, img_min, img_max)

    # Scale the intensity values to [0, 1]
    normalized_img = (windowed_img - img_min) / (img_max - img_min)

    return normalized_img


def boundary_validation(img_shape, bbox):
    """
    Validate a bounding box against the image boundaries.

    Args:
    - img_shape (tuple): The shape of the image.
    - bbox (tuple): The bounding box to validate.

    Returns:
    - corrected bbox (tuple): The corrected bounding box.
    """
    # Get the bounding box coordinates
    xmin, ymin, xmax, ymax = bbox

    # offset 1 to avoid out of boundary
    xmax += 1
    ymax += 1

    # Correct the bounding box if it exceeds the image boundaries
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > img_shape[0]:
        xmax = img_shape[0]
    if ymax > img_shape[1]:
        ymax = img_shape[1]

    return (xmin, ymin, xmax, ymax)


def enlarge_bbox(bbox, bbox_enlarge_factor=0.0):
    """
    Enlarge a bounding box by a factor.

    Args:
    - bbox (tuple): The bounding box to enlarge.
    - bbox_enlarge_factor (float): The factor to enlarge the bounding box by. one dimension enlarge factor

    Returns:
    - enlarged bbox (tuple): The enlarged bounding box.
    """
    bbox_enlarge_factor = 1.0 + bbox_enlarge_factor
    # Get the bounding box coordinates
    xmin, ymin, xmax, ymax = bbox

    # Calculate the center of the bounding box
    x_center = (xmin + xmax) // 2
    y_center = (ymin + ymax) // 2

    # Calculate the new bounding box coordinates
    xmin = int(x_center - (x_center - xmin) * bbox_enlarge_factor)
    xmax = int(x_center + (xmax - x_center) * bbox_enlarge_factor)
    ymin = int(y_center - (y_center - ymin) * bbox_enlarge_factor)
    ymax = int(y_center + (ymax - y_center) * bbox_enlarge_factor)

    return (xmin, ymin, xmax, ymax)


def crop_and_pad(
    img: np.ndarray,
    bbox: np.ndarray | list,
    z_indicies: Optional[np.ndarray | list] = None,
    channel_per_slice: Optional[int] = 1,
    img_depth: Optional[int] = None,
    bbox_enlarge_factor=0.0,
    crop_to_square=False,
    return_dict: Optional[bool] = False,
):
    """
    Crop and pad a 3D image to a bounding box.

    Args:
    - img (numpy.ndarray): The input image data. h, w, c
    - bbox (tuple): The bounding box to crop to. xmin, ymin, xmax, ymax
    - center_slice_indices (numpy.ndarray | list): The center slice indices to crop. Z-direction slice indices
    - img_depth (int): The target length of the image sequence.
    - pad_value (int or float): The value to pad the image with.
    - bbox_enlarge_factor (float): The factor to enlarge the bounding box by.
    - crop_to_square (bool): Whether to crop the image to a square. Centered on the bounding box.

    Returns:
    - numpy.ndarray: The cropped and padded image. HWL if channel_per_slice > 1, LHWC if channel_per_slice = 1
    """

    if crop_to_square:
        crop_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        center_x = (bbox[2] + bbox[0]) // 2
        center_y = (bbox[3] + bbox[1]) // 2
        bbox = (
            center_x - crop_size // 2,
            center_y - crop_size // 2,
            center_x + crop_size // 2,
            center_y + crop_size // 2,
        )

    # Get the bounding box coordinates
    if bbox_enlarge_factor != 0.0:
        bbox = enlarge_bbox(bbox, bbox_enlarge_factor)

    (
        xmin,
        ymin,
        xmax,
        ymax,
    ) = boundary_validation(img.shape, bbox)

    width = xmax - xmin
    height = ymax - ymin

    if z_indicies is None or img_depth is None:
        z_indicies = np.arange(img.shape[2])
        img_depth = img.shape[2]

    if channel_per_slice and channel_per_slice > 1:
        cropped_img = np.zeros(
            (img_depth, height, width, channel_per_slice), dtype=np.float32
        )

        for i, z in enumerate(z_indicies):
            cropped_img[i, :, :, : z.shape[0]] = img[ymin:ymax, xmin:xmax, z]
    else:
        if len(z_indicies) < img_depth:
            cropped_img = np.zeros((height, width, img_depth), dtype=np.float32)
            cropped_img[:, :, : len(z_indicies)] = img[ymin:ymax, xmin:xmax, z_indicies]
        else:
            cropped_img = img[ymin:ymax, xmin:xmax, z_indicies]

    if return_dict:
        return {
            "cropped_img": cropped_img,
            "bbox": (
                xmin,
                ymin,
                xmax,
                ymax,
            ),
        }

    return cropped_img


def draw_bounding_box(img, bbox):
    """
    Draw a bounding box on an image.

    Args:
    - img (numpy.ndarray): The input image data.
    - bbox (tuple): The bounding box to draw.

    Returns:
    - numpy.ndarray: The image with the bounding box drawn on it.
    """
    # Get the bounding box coordinates
    xmin, ymin, xmax, ymax = bbox

    # Draw the bounding box
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    return img


def process_one_segment(
    coordinates: tuple,
    ct_image: np.ndarray,
    image_channels: int = 1,
    xy_plane_enlarge_factor: Optional[float] = 0.0,
    z_axis_enlarge_factor: Optional[float] = 0.0,
    mask: Optional[np.ndarray] = None,
    usage: str = "train",
    img_depth: Optional[int] = 40,
    pad_seq_when_shorter: Optional[bool] = False,
    augmentations: Optional[A.Compose] = None,
    on_slice_augmentation: Optional[bool] = False,
    return_dict: Optional[bool] = False,
):
    """
    Process a single segment of a 3D image.
    :param coordinates:
    :param ct_image:
    :param image_channels:
    :param xy_plane_enlarge_factor:
    :param z_axis_enlarge_factor:
    :param usage: train or eval
    :param mask:
    :param img_depth: target length of the image sequence
    :param pad_seq_when_shorter:
    :param augmentations:
    :param on_slice_augmentation:
    :param return_dict: bbox or slice indices will be removed if False
    :return: a single segment of a 3D image with shape (L, H, W) or (C, L, H, W)
    """
    h, w, length = ct_image.shape
    # Z-axis VOI span: could be lung, cervical, lumbar whatever
    xmin, ymin, zmin, xmax, ymax, zmax = coordinates
    # enlarge zmax and zmin by (zmax - zmin) * 0.2
    zmin = max(0, zmin - int((zmax - zmin) * z_axis_enlarge_factor))
    zmax = min(length, zmax + int((zmax - zmin) * z_axis_enlarge_factor))

    channels_per_slice = 1  # image_channels
    one_side_channels = channels_per_slice // 2

    lowest, highest = max(0, zmin - one_side_channels), min(
        length, zmax + one_side_channels
    )
    if highest - lowest > 512:
        diff = highest - lowest - 512
        lowest += diff // 2
        highest -= diff - diff // 2

    # bound the zmin and zmax to the most_left and most_right
    zmin, zmax = max(zmin, lowest), min(zmax, highest - 1)
    length = zmax - zmin + 1

    if usage == "train":
        # Determine the center slice indices
        if length < img_depth and pad_seq_when_shorter:
            center_slice_indices = np.arange(zmin, zmax + 1)
        else:
            center_slice_indices = np.linspace(zmin, zmax, img_depth, dtype=np.int16)
    elif usage == "cam":
        # hard code downsampling for cam; Need to implement per-slice Grad-Cam for future adoption
        if length > 150:
            _n = length // 75
            center_slice_indices = np.arange(zmin, zmax + 1)[::_n]
        else:
            center_slice_indices = np.arange(zmin, zmax + 1)
        img_depth = len(center_slice_indices)
    else:
        # override all the settings in eval mode
        center_slice_indices = np.arange(zmin, zmax + 1)
        img_depth = length

    bbox = np.array([xmin, ymin, xmax, ymax])
    # set bbox to int:
    bbox = bbox.astype(int)
    image = crop_and_pad(
        ct_image,
        bbox,
        z_indicies=center_slice_indices,
        img_depth=img_depth,
        crop_to_square=True,
        bbox_enlarge_factor=xy_plane_enlarge_factor,
        return_dict=return_dict,
    )  # HWL

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
        )

    if return_dict:
        # refresh image and bbox
        bbox = image["bbox"]
        image = image["cropped_img"]

    # Apply augmentations if required
    if augmentations:
        if not on_slice_augmentation:
            if mask is not None:
                transformed = augmentations(image=image, mask=m)
                augmented_image = transformed["image"]
                augmented_mask = transformed["mask"]
                mask = augmented_mask.transpose((2, 0, 1))  # LHW
            else:
                augmented_image = augmentations(image=image)["image"]

            image = augmented_image.transpose((2, 0, 1))  # LHW
        else:
            augmented_images = []
            _h, _w, _l = image.shape
            for i in range(_l):
                _img = image[:, :, i]
                if mask is not None:
                    _mask = mask[:, :, i]
                    transformed = augmentations(image=_img, mask=_mask)
                    augmented_image = transformed["image"]
                    augmented_mask = transformed["mask"]
                    mask[i] = augmented_mask
                else:
                    augmented_image = augmentations(image=_img)["image"]

                augmented_images.append(augmented_image)

            image = np.stack(augmented_images, axis=0)  # LHW

    # concat image_sequence and mask_sequence
    if image_channels > 1:
        image = np.stack([image] * image_channels, axis=0)

    if mask is not None:
        if mask.ndim == 3:
            mask = np.expand_dims(mask, axis=0)

        # handle edge case when image_channels == 1, the first dimension is missing
        if image_channels == 1:
            image = np.expand_dims(image, axis=0)

        image = np.concatenate([image, mask], axis=0)

    if return_dict:
        # return image, center_slice_indices, bbox
        # Will be used in GradCAM
        return {
            "image": image,
            "center_slice_indices": center_slice_indices,
            "bbox": bbox,
        }

    return image


if __name__ == "__main__":
    test_img = np.random.rand(256, 256, 208)  # h, w, c
    test_bbox = (50, 50, 150, 150)
    test_bbox_enlarge_factor = 0.1
    test_crop_to_square = True
    test_pad_value = 0

    test_img = crop_and_pad(
        test_img,
        test_bbox,
        bbox_enlarge_factor=test_bbox_enlarge_factor,
        crop_to_square=test_crop_to_square,
    )
    print(test_img.shape)
