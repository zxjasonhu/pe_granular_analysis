import albumentations as A
import cv2

from dataset.augmentations.plain_cutout import Cutout


def get_pe_transforms(config, target: str = "train"):
    assert config is not None, "config is None"

    if config.img_size is None:
        config.img_size = 256

    img_size = config.img_size
    img_depth = config.img_depth

    if target == "train":
        transform = A.Compose(
            [
                A.LongestMaxSize(img_size),
                A.PadIfNeeded(
                    img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0
                ),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=25,
                    border_mode=4,
                    p=0.4,
                ),
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=(3, 9), always_apply=True),
                        A.MedianBlur(blur_limit=(3, 5), always_apply=True),
                        A.GaussianBlur(blur_limit=(3, 9), always_apply=True),
                    ],
                    p=0.5,
                ),
                A.GaussNoise(var_limit=(0.0005, 0.008), p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.OneOf(
                    [
                        A.OpticalDistortion(
                            distort_limit=(-0.5, 0.5), always_apply=True
                        ),
                        A.GridDistortion(
                            num_steps=5, distort_limit=0.1, always_apply=True
                        ),
                    ],
                    p=0.3,
                ),
                Cutout(
                    max_h_size=int(256 * 0.2),
                    max_w_size=int(256 * 0.2),
                    num_holes=2,
                    p=0.3,
                ),
            ]
        )

    else:
        transform = A.Compose(
            [
                A.LongestMaxSize(img_size),
                A.PadIfNeeded(
                    img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0
                ),
            ]
        )

    return transform

if __name__ == "__main__":
    pass
