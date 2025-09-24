import os
from typing import Optional, List, Tuple
import types

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from pytorch_grad_cam.base_cam import BaseCAM

from configs.base_cfg import Config
from explainer.cam_generator.base_generator import BaseGenerator
from explainer.utils.interpolation import interpolate_cam_on_voxel
from explainer.utils.save_cam import save_cam


class GradCam2DSeqGenerator(BaseGenerator):
    def __init__(
        self,
        model_list: List[nn.Module],
        cam: BaseCAM.__class__,
        dataset: Dataset,
        config: Config,
        grad_cam_targets: Optional[list] = None,
        test_name: str = "cam_test",
        debug: bool = False,
    ):
        def custom_get_target_width_height(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
            # print("calling custom get_target_width_height")
            if len(input_tensor.shape) == 4:
                width, height = input_tensor.size(-1), input_tensor.size(-2)
                return width, height
            elif len(input_tensor.shape) == 5:
                width, height = input_tensor.size(-1), input_tensor.size(-2)
                return width, height
            else:
                raise ValueError("Invalid input_tensor shape. Only 2D or 3D images are supported.")
        cam.get_target_width_height = types.MethodType(custom_get_target_width_height, cam)
        
        super().__init__(
            model_list=model_list,
            cam=cam,
            dataset=dataset,
            config=config,
            grad_cam_targets=grad_cam_targets,
            test_name=test_name,
            debug=debug,
        )

    def process(self):
        progress_bar = tqdm(self.loader, desc="GradCamGeneration", leave=False)

        for inputs_ in progress_bar:
            images = inputs_["images"]
            print(f"Input images shape: {images.shape}")
            xy_boxes = inputs_["xy_boxes"]  # [xmin, ymin, xmax, ymax]
            z_slices = inputs_["z_slices"]
            pid = inputs_["pid"]
            study_uid = inputs_["study_uid"]
            series_uid = inputs_["series_uid"]
            input_folder = inputs_["input_folder"]
            source_voxel_shape = torch.cat(inputs_["source_voxel_shape"], dim=0).numpy()

            batch, depth, channel, height, width = images.shape
            (
                source_height,
                source_width,
                source_depth,
            ) = source_voxel_shape  # HWL from dataset

            if self.debug:
                print(f"source_voxel_shape: {source_voxel_shape}")
                print(f"input_folder: {input_folder}")
                print(f"_id: {_id}")
                print(f"xy_boxes: {type(xy_boxes)}, {xy_boxes}")
                print(f"z_slices: {type(z_slices)}, {z_slices}")
                print(f"input images shape: {images.shape}")
                debug_img = images.detach().cpu().numpy()
                np.save(
                    os.path.join(
                        self.result_output_folder,
                        f"debug_{_id}_input_img.npy".replace("'", "")
                        .replace("[", "")
                        .replace("]", ""),
                    ),
                    debug_img,
                )

            assert (
                batch == 1
            ), f"batch={batch} != 1. input_voxel_shape should have 1 batch dimension. does not support batch processing."

            final_out_cam = None
            cam_normalizer = np.ones((source_depth, source_height, source_width))
            images = images.to(self.device, non_blocking=True)

            for model, grad_cam_target_layers in zip(
                self.model_list, self.grad_cam_target_layers_list
            ):
                with self.grad_cam(
                    model=model,
                    target_layers=grad_cam_target_layers,
                ) as cam:
                    model_cam = None
                    _image = images
                    _z_slices = z_slices.numpy()
                    _xy_boxes = torch.cat(xy_boxes, dim=0).numpy()

                    cam_images = cam(
                        input_tensor=_image, targets=self.grad_cam_targets
                    )  # DHW
                    if self.debug:
                        print(f"cam_images shape: {cam_images.shape}")

                    _out_cam = (
                        interpolate_cam_on_voxel(
                            cam=cam_images,
                            source_voxel_shape=source_voxel_shape,
                            input_voxel_shape=(depth, height, width),
                            z_slices=_z_slices,
                            xy_plane_boxes=_xy_boxes,
                        )
                    )

                    if self.config.require_flip_inference:
                        flip_cam = cam(
                            input_tensor=torch.flip(_image, dims=(-1,)),
                            targets=self.grad_cam_targets,
                        )
                        flip_interpolated_cam = (
                            interpolate_cam_on_voxel(
                                cam=flip_cam,
                                source_voxel_shape=source_voxel_shape,
                                input_voxel_shape=(depth, height, width),
                                z_slices=_z_slices,
                                xy_plane_boxes=_xy_boxes,
                            )
                        )
                        _out_cam += np.flip(flip_interpolated_cam, axis=-1)
                        _out_cam /= 2

                    if model_cam is None:
                        model_cam = _out_cam
                    else:
                        model_cam = np.maximum(model_cam, _out_cam)

                if final_out_cam is None:
                    final_out_cam = model_cam
                else:
                    final_out_cam += model_cam
                    # _out_cam > 0 means the voxel is activated
                    cam_normalizer += (model_cam > 0).astype(np.float32)

            final_out_cam /= cam_normalizer
            if self.debug:
                print(f"cam_image shape: {cam_images.shape}")
                print(f"interpolated_cam shape: {_out_cam.shape}")
                np.savez_compressed(
                    os.path.join(
                        self.result_output_folder,
                        f"debug_{_id}_cam.npz".replace("'", "")
                        .replace("[", "")
                        .replace("]", ""),
                    ),
                    final_out_cam,
                )
                break
            save_cam(result_output_folder=self.result_output_folder, 
                     final_out_cam=final_out_cam, 
                     pid=pid, 
                     study_uid=study_uid, 
                     series_uid=series_uid)
        self.clean()

if __name__ == "__main__":
    pass
