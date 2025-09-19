import warnings

from nibabel import as_closest_canonical
from totalsegmentator.alignment import undo_canonical
from totalsegmentator.cropping import crop_to_mask, undo_crop
from totalsegmentator.dicom_io import dcm_to_nifti, save_mask_as_rtstruct
from totalsegmentator.nifti_ext_header import add_label_map_to_nifti
from totalsegmentator.nnunet import save_segmentation_nifti
from totalsegmentator.postprocessing import (
    remove_auxiliary_labels,
    remove_small_blobs_multilabel,
)
from totalsegmentator.python_api import convert_device_to_cuda
from totalsegmentator.resampling import change_spacing

from .targets import (
    class_map,
    class_map_lung_part,
    class_map_heart_part,
    map_taskid_to_partname_ct,
)


def recursive_find_python_class_custom(
    folder: str, class_name: str, current_module: str
):
    if class_name == "nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring":
        return nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring

    return recursive_find_python_class(folder, class_name, current_module)


# monkey-patch
import nnunetv2

nnunetv2.inference.predict_from_raw_data.recursive_find_python_class = (
    recursive_find_python_class_custom
)
# --- now we have included custom trainers into the nnUNetv2 basic package --- #

# nnUNet 2.1
# with nostdout():
#     from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
# nnUNet 2.2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Hide nnunetv2 warning: Detected old nnU-Net plans format. Attempting to reconstruct network architecture...
warnings.filterwarnings("ignore", category=UserWarning, module="nnunetv2")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="nnunetv2"
)  # ignore torch.load warning
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import Union
from multiprocessing import Pool
import tempfile

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image

# from p_tqdm import p_map
import torch

from totalsegmentator.libs import (
    nostdout,
    combine_masks,
    check_if_shape_and_affine_identical,
)

# --- monkey-patch snippet (custom trainers) --- #
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from totalsegmentator.custom_trainers import (
    nnUNetTrainer_MOSAIC_1k_QuarterLR_NoMirroring,
)


def nnUNetv2_predict(
    dir_in,
    dir_out,
    num_threads_preprocessing=3,
    num_threads_nifti_save=2,
    predictor=None,
):
    """
    Identical to bash function nnUNetv2_predict
    """
    dir_in = str(dir_in)
    dir_out = str(dir_out)

    save_probabilities = False
    continue_prediction = False

    npp = num_threads_preprocessing
    nps = num_threads_nifti_save
    prev_stage_predictions = None
    num_parts = 1
    part_id = 0

    # new nnunetv2 feature: keep dir_out empty to return predictions as return value
    predictor.predict_from_files(
        dir_in,
        dir_out,
        save_probabilities=save_probabilities,
        overwrite=not continue_prediction,
        num_processes_preprocessing=npp,
        num_processes_segmentation_export=nps,
        folder_with_segs_from_prev_stage=prev_stage_predictions,
        num_parts=num_parts,
        part_id=part_id,
    )


def nnUNet_predict_image(
    file_in: Union[str, Path, Nifti1Image],
    file_out,
    task_id,
    model="3d_fullres",
    folds=None,
    trainer="nnUNetTrainerV2",
    tta=False,
    multilabel_image=True,
    resample=None,
    crop=None,
    crop_path=None,
    task_name="total",
    nora_tag="None",
    preview=False,
    save_binary=False,
    nr_threads_resampling=1,
    nr_threads_saving=6,
    force_split=False,
    crop_addon=[3, 3, 3],
    roi_subset=None,
    output_type="nifti",
    statistics=False,
    quiet=False,
    verbose=False,
    test=0,
    skip_saving=False,
    device="cuda",
    exclude_masks_at_border=True,
    no_derived_masks=False,
    v1_order=False,
    stats_aggregation="mean",
    remove_small_blobs=False,
    normalized_intensities=False,
    predictor=None,
):
    """
    crop: string or a nibabel image
    resample: None or float (target spacing for all dimensions) or list of floats
    """
    if not isinstance(file_in, Nifti1Image):
        file_in = Path(file_in)
        if str(file_in).endswith(".nii") or str(file_in).endswith(".nii.gz"):
            img_type = "nifti"
        else:
            img_type = "dicom"
        if not file_in.exists():
            sys.exit("ERROR: The input file or directory does not exist.")
    else:
        img_type = "nifti"
    if file_out is not None:
        file_out = Path(file_out)
    multimodel = type(task_id) is list

    if img_type == "nifti" and output_type == "dicom":
        raise ValueError(
            "To use output type dicom you also have to use a Dicom image as input."
        )

    if task_name == "lung":
        class_map_parts = class_map_lung_part
    elif task_name == "heart":
        class_map_parts = class_map_heart_part
    map_taskid_to_partname = map_taskid_to_partname_ct

    if type(resample) is float:
        resample = [resample, resample, resample]

    label_map = class_map[task_name]

    # Keep only voxel values corresponding to the roi_subset
    if roi_subset is not None:
        label_map = {k: v for k, v in label_map.items() if v in roi_subset}

    # for debugging
    # tmp_dir = file_in.parent / ("nnunet_tmp_" + ''.join(random.Random().choices(string.ascii_uppercase + string.digits, k=8)))
    # (tmp_dir).mkdir(exist_ok=True)
    # with tmp_dir as tmp_folder:
    with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
        tmp_dir = Path(tmp_folder)
        if verbose:
            print(f"tmp_dir: {tmp_dir}")

        if img_type == "dicom":
            if not quiet:
                print("Converting dicom to nifti...")
            (
                tmp_dir / "dcm"
            ).mkdir()  # make subdir otherwise this file would be included by nnUNet_predict
            dcm_to_nifti(
                file_in,
                tmp_dir / "dcm" / "converted_dcm.nii.gz",
                tmp_dir,
                verbose=verbose,
            )
            file_in_dcm = file_in
            file_in = tmp_dir / "dcm" / "converted_dcm.nii.gz"

            # for debugging
            # shutil.copy(file_in, file_in_dcm.parent / "converted_dcm_TMP.nii.gz")

            # Workaround to be able to access file_in on windows (see issue #106)
            # if platform.system() == "Windows":
            #     file_in = file_in.NamedTemporaryFile(delete = False)
            #     file_in.close()

            # if not multilabel_image:
            #     shutil.copy(file_in, file_out / "input_file.nii.gz")
            if not quiet:
                print(f"  found image with shape {nib.load(file_in).shape}")

        if isinstance(file_in, Nifti1Image):
            img_in_orig = file_in
        else:
            img_in_orig = nib.load(file_in)
        if len(img_in_orig.shape) == 2:
            raise ValueError(
                "TotalSegmentator does not work for 2D images. Use a 3D image."
            )
        if len(img_in_orig.shape) > 3:
            print(
                f"WARNING: Input image has {len(img_in_orig.shape)} dimensions. Only using first three dimensions."
            )
            img_in_orig = nib.Nifti1Image(
                img_in_orig.get_fdata()[:, :, :, 0], img_in_orig.affine
            )

        img_dtype = img_in_orig.get_data_dtype()
        if img_dtype.fields is not None:
            raise TypeError(
                f"Invalid dtype {img_dtype}. Expected a simple dtype, not a structured one."
            )

        # takes ~0.9s for medium image
        img_in = nib.Nifti1Image(
            img_in_orig.get_fdata(), img_in_orig.affine
        )  # copy img_in_orig

        if crop is not None:
            if type(crop) is str:
                if crop == "lung" or crop == "pelvis":
                    crop_mask_img = combine_masks(crop_path, crop)
                else:
                    crop_mask_img = nib.load(crop_path / f"{crop}.nii.gz")
            else:
                crop_mask_img = crop

            if crop_mask_img.get_fdata().sum() == 0:
                if not quiet:
                    print("INFO: Crop is empty. Returning empty segmentation.")
                img_out = nib.Nifti1Image(
                    np.zeros(img_in.shape, dtype=np.uint8), img_in.affine
                )
                img_out = add_label_map_to_nifti(img_out, label_map)
                if file_out is not None:
                    nib.save(img_out, file_out)
                    if nora_tag != "None":
                        subprocess.call(
                            f"/opt/nora/src/node/nora -p {nora_tag} --add {file_out} --addtag atlas",
                            shell=True,
                        )
                return img_out, img_in_orig, None

            img_in, bbox = crop_to_mask(
                img_in, crop_mask_img, addon=crop_addon, dtype=np.int32, verbose=verbose
            )
            if not quiet:
                print(f"  cropping from {crop_mask_img.shape} to {img_in.shape}")

        img_in = as_closest_canonical(img_in)

        if resample is not None:
            if not quiet:
                print("Resampling...")
            st = time.time()
            img_in_shape = img_in.shape
            img_in_zooms = img_in.header.get_zooms()
            img_in_rsp = change_spacing(
                img_in, resample, order=3, dtype=np.int32, nr_cpus=nr_threads_resampling
            )  # 4 cpus instead of 1 makes it a bit slower
            if verbose:
                print(f"  from shape {img_in.shape} to shape {img_in_rsp.shape}")
            if not quiet:
                print(f"  Resampled in {time.time() - st:.2f}s")
        else:
            img_in_rsp = img_in

        nib.save(img_in_rsp, tmp_dir / "s01_0000.nii.gz")

        # todo important: change
        nr_voxels_thr = 512 * 512 * 900
        # nr_voxels_thr = 256*256*900
        img_parts = ["s01"]
        ss = img_in_rsp.shape
        # If image to big then split into 3 parts along z axis. Also make sure that z-axis is at least 200px otherwise
        # splitting along it does not really make sense.
        do_triple_split = np.prod(ss) > nr_voxels_thr and ss[2] > 200 and multimodel
        if force_split:
            do_triple_split = True
        if do_triple_split:
            if not quiet:
                print("Splitting into subparts...")
            img_parts = ["s01", "s02", "s03"]
            third = img_in_rsp.shape[2] // 3
            margin = 20  # set margin with fixed values to avoid rounding problem if using percentage of third
            img_in_rsp_data = img_in_rsp.get_fdata()
            nib.save(
                nib.Nifti1Image(
                    img_in_rsp_data[:, :, : third + margin], img_in_rsp.affine
                ),
                tmp_dir / "s01_0000.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(
                    img_in_rsp_data[:, :, third + 1 - margin : third * 2 + margin],
                    img_in_rsp.affine,
                ),
                tmp_dir / "s02_0000.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(
                    img_in_rsp_data[:, :, third * 2 + 1 - margin :], img_in_rsp.affine
                ),
                tmp_dir / "s03_0000.nii.gz",
            )

        if task_name == "total" and resample is not None and resample[0] < 3.0:
            # overall speedup for 15mm model roughly 11% (GPU) and 100% (CPU)
            # overall speedup for  3mm model roughly  0% (GPU) and  10% (CPU)
            # (dice 0.001 worse on test set -> ok)
            # (for lung_trachea_bronchia somehow a lot lower dice)
            step_size = 0.8
        else:
            step_size = 0.5

        st = time.time()
        if multimodel:  # if running multiple models
            # only compute model parts containing the roi subset
            if roi_subset is not None:
                part_names = []
                new_task_id = []
                for part_name, part_map in class_map_parts.items():
                    if any(organ in roi_subset for organ in part_map.values()):
                        # get taskid associated to model part_name
                        map_partname_to_taskid = {
                            v: k for k, v in map_taskid_to_partname.items()
                        }
                        new_task_id.append(map_partname_to_taskid[part_name])
                        part_names.append(part_name)
                task_id = new_task_id
                if verbose:
                    print(
                        f"Computing parts: {part_names} based on the provided roi_subset"
                    )

            if test == 0:
                class_map_inv = {v: k for k, v in class_map[task_name].items()}
                (tmp_dir / "parts").mkdir(exist_ok=True)
                seg_combined = {}
                # iterate over subparts of image
                for img_part in img_parts:
                    img_shape = nib.load(tmp_dir / f"{img_part}_0000.nii.gz").shape
                    seg_combined[img_part] = np.zeros(img_shape, dtype=np.uint8)
                # Run several tasks and combine results into one segmentation
                for idx, tid in enumerate(task_id):
                    if not quiet:
                        print(f"Predicting part {idx+1} of {len(task_id)} ...")
                    with nostdout(verbose):
                        # nnUNet_predict(tmp_dir, tmp_dir, tid, model, folds, trainer, tta,
                        #                nr_threads_resampling, nr_threads_saving)
                        nnUNetv2_predict(
                            tmp_dir,
                            tmp_dir,
                            nr_threads_resampling,
                            nr_threads_saving,
                            predictor=predictor,
                        )
                    # iterate over models (different sets of classes)
                    for img_part in img_parts:
                        (tmp_dir / f"{img_part}.nii.gz").rename(
                            tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz"
                        )
                        seg = nib.load(
                            tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz"
                        ).get_fdata()
                        for jdx, class_name in class_map_parts[
                            map_taskid_to_partname[tid]
                        ].items():
                            seg_combined[img_part][seg == jdx] = class_map_inv[
                                class_name
                            ]
                # iterate over subparts of image
                for img_part in img_parts:
                    nib.save(
                        nib.Nifti1Image(seg_combined[img_part], img_in_rsp.affine),
                        tmp_dir / f"{img_part}.nii.gz",
                    )
            elif test == 1:
                print("WARNING: Using reference seg instead of prediction for testing.")
                shutil.copy(
                    Path("tests") / "reference_files" / "example_seg.nii.gz",
                    tmp_dir / "s01.nii.gz",
                )
        else:
            if not quiet:
                print("Predicting...")
            if test == 0:
                with nostdout(verbose):
                    # nnUNet_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta,
                    #                nr_threads_resampling, nr_threads_saving)
                    nnUNetv2_predict(
                        tmp_dir,
                        tmp_dir,
                        nr_threads_resampling,
                        nr_threads_saving,
                        predictor=predictor,
                    )
            # elif test == 2:
            #     print("WARNING: Using reference seg instead of prediction for testing.")
            #     shutil.copy(Path("tests") / "reference_files" / "example_seg_fast.nii.gz", tmp_dir / f"s01.nii.gz")
            elif test == 3:
                print("WARNING: Using reference seg instead of prediction for testing.")
                shutil.copy(
                    Path("tests")
                    / "reference_files"
                    / "example_seg_lung_vessels.nii.gz",
                    tmp_dir / "s01.nii.gz",
                )
        if not quiet:
            print(f"  Predicted in {time.time() - st:.2f}s")

        # Combine image subparts back to one image
        if do_triple_split:
            combined_img = np.zeros(img_in_rsp.shape, dtype=np.uint8)
            combined_img[:, :, :third] = nib.load(tmp_dir / "s01.nii.gz").get_fdata()[
                :, :, :-margin
            ]
            combined_img[:, :, third : third * 2] = nib.load(
                tmp_dir / "s02.nii.gz"
            ).get_fdata()[:, :, margin - 1 : -margin]
            combined_img[:, :, third * 2 :] = nib.load(
                tmp_dir / "s03.nii.gz"
            ).get_fdata()[:, :, margin - 1 :]
            nib.save(
                nib.Nifti1Image(combined_img, img_in_rsp.affine), tmp_dir / "s01.nii.gz"
            )

        img_pred = nib.load(tmp_dir / "s01.nii.gz")

        # Currently only relevant for T304 (appendicular bones)
        img_pred = remove_auxiliary_labels(img_pred, task_name)

        # General postprocessing
        if remove_small_blobs:
            if not quiet:
                print("Removing small blobs...")
            st = time.time()
            vox_vol = np.prod(img_pred.header.get_zooms())
            size_thr_mm3 = 200
            img_pred_pp = remove_small_blobs_multilabel(
                img_pred.get_fdata().astype(np.uint8),
                class_map[task_name],
                list(class_map[task_name].values()),
                interval=[size_thr_mm3 / vox_vol, 1e10],
                debug=False,
                quiet=quiet,
            )  # ~24s
            img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)
            if not quiet:
                print(f"  Removed in {time.time() - st:.2f}s")

        if preview:
            from totalsegmentator.preview import generate_preview

            # Generate preview before upsampling so it is faster and still in canonical space
            # for better orientation.
            if not quiet:
                print("Generating preview...")
            if file_out is None:
                print(
                    "WARNING: No output directory specified. Skipping preview generation."
                )
            else:
                st = time.time()
                smoothing = 20
                preview_dir = file_out.parent if multilabel_image else file_out
                generate_preview(
                    img_in_rsp,
                    preview_dir / f"preview_{task_name}.png",
                    img_pred.get_fdata(),
                    smoothing,
                    task_name,
                )
                if not quiet:
                    print(f"  Generated in {time.time() - st:.2f}s")

        # Statistics calculated on the 3mm downsampled image are very similar to statistics
        # calculated on the original image. Volume often completely identical. For intensity
        # some more change but still minor.
        #
        # Speed:
        # stats on 1.5mm: 37s
        # stats on 3.0mm: 4s    -> great improvement
        stats = None

        if resample is not None:
            if not quiet:
                print("Resampling...")
            if verbose:
                print(f"  back to original shape: {img_in_shape}")
            # Use force_affine otherwise output affine sometimes slightly off (which then is even increased
            # by undo_canonical)
            img_pred = change_spacing(
                img_pred,
                resample,
                img_in_shape,
                order=0,
                dtype=np.uint8,
                nr_cpus=nr_threads_resampling,
                force_affine=img_in.affine,
            )

        if verbose:
            print("Undoing canonical...")
        img_pred = undo_canonical(img_pred, img_in_orig)

        if crop is not None:
            if verbose:
                print("Undoing cropping...")
            img_pred = undo_crop(img_pred, img_in_orig, bbox)

        check_if_shape_and_affine_identical(img_in_orig, img_pred)

        img_data = img_pred.get_fdata().astype(np.uint8)
        if save_binary:
            img_data = (img_data > 0).astype(np.uint8)

        # Keep only voxel values corresponding to the roi_subset
        if roi_subset is not None:
            img_data *= np.isin(img_data, list(label_map.keys()))

        # Prepare output nifti
        # Copy header to make output header exactly the same as input. But change dtype otherwise it will be
        # float or int and therefore the masks will need a lot more space.
        # (infos on header: https://nipy.org/nibabel/nifti_images.html)
        new_header = img_in_orig.header.copy()
        new_header.set_data_dtype(np.uint8)
        img_out = nib.Nifti1Image(img_data, img_pred.affine, new_header)
        img_out = add_label_map_to_nifti(img_out, label_map)

        if file_out is not None and skip_saving is False:
            if not quiet:
                print("Saving segmentations...")

            # Select subset of classes if required
            selected_classes = class_map[task_name]
            if roi_subset is not None:
                selected_classes = {
                    k: v for k, v in selected_classes.items() if v in roi_subset
                }

            if output_type == "dicom":
                file_out.mkdir(exist_ok=True, parents=True)
                save_mask_as_rtstruct(
                    img_data,
                    selected_classes,
                    file_in_dcm,
                    file_out / "segmentations.dcm",
                )
            else:
                st = time.time()
                if multilabel_image:
                    file_out.parent.mkdir(exist_ok=True, parents=True)
                else:
                    file_out.mkdir(exist_ok=True, parents=True)
                if multilabel_image:
                    nib.save(img_out, file_out)
                    if nora_tag != "None":
                        subprocess.call(
                            f"/opt/nora/src/node/nora -p {nora_tag} --add {file_out} --addtag atlas",
                            shell=True,
                        )
                else:  # save each class as a separate binary image
                    file_out.mkdir(exist_ok=True, parents=True)

                    if np.prod(img_data.shape) > 512 * 512 * 1000:
                        print(
                            "Shape of output image is very big. Setting nr_threads_saving=1 to save memory."
                        )
                        nr_threads_saving = 1

                    # Code for single threaded execution  (runtime:24s)
                    if nr_threads_saving == 1:
                        for k, v in selected_classes.items():
                            binary_img = img_data == k
                            output_path = str(file_out / f"{v}.nii.gz")
                            nib.save(
                                nib.Nifti1Image(
                                    binary_img.astype(np.uint8),
                                    img_pred.affine,
                                    new_header,
                                ),
                                output_path,
                            )
                            if nora_tag != "None":
                                subprocess.call(
                                    f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask",
                                    shell=True,
                                )
                    else:
                        nib.save(
                            img_pred, tmp_dir / "s01.nii.gz"
                        )  # needed inside of threads

                        # Code for multithreaded execution
                        #   Speed with different number of threads:
                        #   1: 46s, 2: 24s, 6: 11s, 10: 8s, 14: 8s
                        # _ = p_map(partial(save_segmentation_nifti, tmp_dir=tmp_dir, file_out=file_out, nora_tag=nora_tag, header=new_header, task_name=task_name, quiet=quiet),
                        #         selected_classes.items(), num_cpus=nr_threads_saving, disable=quiet)

                        # Multihreaded saving with same functions as in nnUNet -> same speed as p_map
                        pool = Pool(nr_threads_saving)
                        results = []
                        for k, v in selected_classes.items():
                            results.append(
                                pool.starmap_async(
                                    save_segmentation_nifti,
                                    [
                                        (
                                            (k, v),
                                            tmp_dir,
                                            file_out,
                                            nora_tag,
                                            new_header,
                                            task_name,
                                            quiet,
                                        )
                                    ],
                                )
                            )
                        _ = [
                            i.get() for i in results
                        ]  # this actually starts the execution of the async functions
                        pool.close()
                        pool.join()
            if not quiet:
                print(f"  Saved in {time.time() - st:.2f}s")

            # Postprocessing single files
            #    (these not directly transferable to multilabel)

            # Lung mask does not exist since I use 6mm model. Would have to save lung mask from 6mm seg.
            # if task_name == "lung_vessels":
            #     remove_outside_of_mask(file_out / "lung_vessels.nii.gz", file_out / "lung.nii.gz")

            # if task_name == "heartchambers_test":
            #     remove_outside_of_mask(file_out / "heart_myocardium.nii.gz", file_out / "heart.nii.gz", addon=5)
            #     remove_outside_of_mask(file_out / "heart_atrium_left.nii.gz", file_out / "heart.nii.gz", addon=5)
            #     remove_outside_of_mask(file_out / "heart_ventricle_left.nii.gz", file_out / "heart.nii.gz", addon=5)
            #     remove_outside_of_mask(file_out / "heart_atrium_right.nii.gz", file_out / "heart.nii.gz", addon=5)
            #     remove_outside_of_mask(file_out / "heart_ventricle_right.nii.gz", file_out / "heart.nii.gz", addon=5)
            #     remove_outside_of_mask(file_out / "aorta.nii.gz", file_out / "heart.nii.gz", addon=5)
            #     remove_outside_of_mask(file_out / "pulmonary_artery.nii.gz", file_out / "heart.nii.gz", addon=5)

    return img_out, img_in_orig, stats


# def select_device(device):
#     device = convert_device_to_cuda(device)
#
#     # available devices: gpu | cpu | mps | gpu:1, gpu:2, etc.
#     if device == "gpu":
#         device = "cuda"
#     if device.startswith("cuda"):
#         if device == "cuda":
#             device = "cuda:0"
#         if not torch.cuda.is_available():
#             print(
#                 "No GPU detected. Running on CPU. This can be very slow. The '--fast' or the `--roi_subset` option can help to reduce runtime."
#             )
#             device = "cpu"
#         else:
#             device_id = int(device[5:])
#             if device_id < torch.cuda.device_count():
#                 device = torch.device(device)
#             else:
#                 print("Invalid GPU config, running on the CPU")
#                 device = "cpu"
#     return device

def pseudo_pulmonary_artery_region(lung_mask:np.ndarray, heart_mask:np.ndarray=None) -> np.ndarray:
    """
    Create a mask that includes the pulmonary artery region based on lung and heart masks.
    The region is defined using anatomical heuristics.
    Parameters:
    - lung_mask: 3D numpy array representing the lung segmentation mask.
    - heart_mask: 3D numpy array representing the heart segmentation mask.
    Returns:
    - roi_mask: 3D numpy array (boolean) representing the region of interest mask.
    """
    if heart_mask is None:
        # find lung center
        lung_center = np.mean(np.argwhere(lung_mask > 0), axis=0)
        # find lung bbox coordinates
        lung_coords = np.argwhere(lung_mask > 0)
        # get xyz ranges
        lung_z_min, lung_y_min, lung_x_min = np.min(lung_coords, axis=0)
        lung_z_max, lung_y_max, lung_x_max = np.max(lung_coords, axis=0)

        # z_bound by heart center z and lung_center + 1/4 lung length
        lung_length_z = lung_z_max - lung_z_min
        z_lower_bound = max(lung_z_min, int(lung_center[0] - lung_length_z * 0.1))
        z_upper_bound = min(lung_z_max, int(lung_center[0] + lung_length_z * 0.3))
        # y_bound by y_ratio * lung bbox height around lung center y
        lung_height_y = lung_y_max - lung_y_min
        y_lower_bound = max(lung_y_min, int(lung_center[1] - lung_height_y * 0.3))
        y_upper_bound = min(lung_y_max, int(lung_center[1] + lung_height_y * 0.15))
        # x_bound by any non-zero value between 0.6 * lung bbox width around lung center x
        lung_width_x = lung_x_max - lung_x_min
        x_lower_bound = max(lung_x_min, int(lung_center[2] - lung_width_x * 0.3))
        x_upper_bound = min(lung_x_max, int(lung_center[2] + lung_width_x * 0.3))
        # create a mask for the region of interest
        roi_mask = np.zeros_like(lung_mask, dtype=bool)
        roi_mask[z_lower_bound:z_upper_bound, y_lower_bound:y_upper_bound, x_lower_bound:x_upper_bound] = True
        return roi_mask

    assert lung_mask.shape == heart_mask.shape, "Lung and heart masks must have the same shape."

    # find heart center
    heart_center = np.mean(np.argwhere(heart_mask > 0), axis=0)
    # find lung center
    lung_center = np.mean(np.argwhere(lung_mask > 0), axis=0)
    # find lung bbox coordinates
    lung_coords = np.argwhere(lung_mask > 0)
    # get xyz ranges
    lung_z_min, lung_y_min, lung_x_min = np.min(lung_coords, axis=0)
    lung_z_max, lung_y_max, lung_x_max = np.max(lung_coords, axis=0)

    # z_bound by heart center z and lung_center + 1/4 lung length
    lung_length_z = lung_z_max - lung_z_min
    z_lower_bound = max(lung_z_min, int(heart_center[0]))
    z_upper_bound = min(lung_z_max, int(lung_center[0] + lung_length_z * 0.3))
    # y_bound by 0.5 * lung bbox height around lung center y
    lung_height_y = lung_y_max - lung_y_min
    y_lower_bound = max(lung_y_min, int(lung_center[1] - lung_height_y * 0.3))
    y_upper_bound = min(lung_y_max, int(lung_center[1] + lung_height_y * 0.15))
    # x_bound by any non-zero value between 0.6 * lung bbox width around lung center x
    lung_width_x = lung_x_max - lung_x_min
    x_lower_bound = max(lung_x_min, int(lung_center[2] - lung_width_x * 0.3))
    x_upper_bound = min(lung_x_max, int(lung_center[2] + lung_width_x * 0.3))

    # create a mask for the region of interest
    roi_mask = np.zeros_like(lung_mask, dtype=bool)
    roi_mask[z_lower_bound:z_upper_bound, y_lower_bound:y_upper_bound, x_lower_bound:x_upper_bound] = True
    return roi_mask