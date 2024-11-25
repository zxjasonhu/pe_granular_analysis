import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image

import cv2


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
