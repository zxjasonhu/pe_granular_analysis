from totalsegmentator.python_api import (
    setup_totalseg,
    setup_nnunet,
)

setup_nnunet()
setup_totalseg()

import os

from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import torch

import nibabel as nib
from nibabel import Nifti1Image

from totalsegmentator.libs import download_pretrained_weights, combine_masks
from totalsegmentator.nnunet import supports_keyword_argument

from nnunetv2.utilities.file_path_utilities import get_output_folder
from preprocessing.seg_utils import nnUNet_predict_image
from nnlog.utils import get_logger
from utils.base import format_input_path

import logging

class CustomSegmentator:
    def __init__(
        self,
        output_type: str = "nifti",
        multilabel: bool = False,
        nr_thr_resamp: int = 1,
        nr_thr_saving: int = 6,
        fast: bool = False,
        fastest: bool = False,
        nora_tag: str = "None",
        preview: bool = False,
        task: str = "lung",
        roi_subset: Optional[List[str]] = None,
        roi_subset_robust: Optional[List[str]] = None,
        robust_crop: bool = False,
        higher_order_resampling: bool = False,
        statistics: bool = False,
        radiomics: bool = False,
        stats_include_incomplete: bool = False,
        crop_path: Optional[Path] = None,
        body_seg: bool = False,
        force_split: bool = False,
        skip_saving: bool = False,
        no_derived_masks: bool = False,
        v1_order: bool = False,
        remove_small_blobs: bool = False,
        device: str = "gpu",
        quiet: bool = True,
        verbose: bool = False,
        license_number: Optional[str] = None,
        test: int = 0,
        save_probabilities: Optional[Path] = None,
    ):
        self.output_type = output_type
        self.multilabel = multilabel
        self.nr_thr_resamp = nr_thr_resamp
        self.nr_thr_saving = nr_thr_saving
        self.fast = fast
        self.fastest = fastest
        self.nora_tag = nora_tag
        self.preview = preview
        self.task = task
        self.roi_subset = roi_subset
        self.roi_subset_robust = roi_subset_robust
        self.robust_crop = robust_crop
        self.higher_order_resampling = higher_order_resampling
        self.statistics = statistics
        self.radiomics = radiomics
        self.stats_include_incomplete = stats_include_incomplete
        self.crop_path = crop_path
        self.body_seg = body_seg
        self.force_split = force_split
        self.skip_saving = skip_saving
        self.no_derived_masks = no_derived_masks
        self.v1_order = v1_order
        self.remove_small_blobs = remove_small_blobs
        device = torch.device(device)
        self.device = device
        self.quiet = quiet
        self.verbose = verbose
        self.license_number = license_number
        self.test = test
        self.save_probabilities = save_probabilities

        self.nora_tag = "None" if nora_tag is None else nora_tag

        # Store initial torch settings
        self.initial_cudnn_benchmark = torch.backends.cudnn.benchmark
        self.initial_num_threads = torch.get_num_threads()

        if verbose:
            print(f"Using Device: {device}")

        if output_type == "dicom":
            try:
                from rt_utils import RTStructBuilder
            except ImportError:
                raise ImportError(
                    "rt_utils is required for output_type='dicom'. Please install it with 'pip install rt_utils'."
                )

        if not quiet:
            print(
                "\nIf you use this tool please cite: https://pubs.rsna.org/doi/10.1148/ryai.230024\n"
            )

        setup_nnunet()
        setup_totalseg()

        self.crop_addon = [3, 3, 3]  # default value
        self.cascade = None
        if task == "lung":
            self.task_id = [291]
        elif task == "heart":
            self.task_id = [293]
        else:
            raise ValueError(f"Unsupported task: {task}")
        # elif task == "total": # not supported yet
        #     self.task_id = [291, 292, 293, 294, 295]

        self.resample = 1.5
        self.step_size = 0.8

        self.trainer = "nnUNetTrainerNoMirroring"
        self.crop = None
        self.model = "3d_fullres"
        self.folds = [0]

        if type(self.task_id) is list:
            for tid in self.task_id:
                download_pretrained_weights(tid)
        else:
            download_pretrained_weights(self.task_id)

        if self.statistics and self.fast:
            self.statistics_fast = True
            self.statistics = False
        else:
            self.statistics_fast = False

        self.predictor = self.initalize_model_instance(task_id=self.task_id[0])

    def initalize_model_instance(self, task_id: int):
        model_folder = get_output_folder(
            task_id, self.trainer, "nnUNetPlans", self.model
        )

        print("Initializing model instance... model_folder:", model_folder)

        # assert self.device in ["cpu", "cuda", "mps"] or isinstance(
        #     self.device, torch.device
        # ), f"-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {self.device}."
        if self.device == "cpu":
            # let's allow torch to use hella threads
            import multiprocessing

            torch.set_num_threads(multiprocessing.cpu_count())
            self.device = torch.device("cpu")
        elif self.device == "cuda":
            # multithreading in torch doesn't help nnU-Net if run on GPU
            torch.set_num_threads(1)
            # torch.set_num_interop_threads(1)  # throws error if setting the second time
            self.device = torch.device("cuda")
        elif isinstance(self.device, torch.device):
            torch.set_num_threads(1)
            self.device = self.device
        else:
            self.device = torch.device("mps")

        tta = False
        disable_tta = not tta
        verbose = False
        chk = "checkpoint_final.pth"
        allow_tqdm = not self.quiet

        # nnUNet 2.1
        # predict_from_raw_data(dir_in,
        #                       dir_out,
        #                       model_folder,
        #                       folds,
        #                       step_size,
        #                       use_gaussian=True,
        #                       use_mirroring=not disable_tta,
        #                       perform_everything_on_gpu=True,
        #                       verbose=verbose,
        #                       save_probabilities=save_probabilities,
        #                       overwrite=not continue_prediction,
        #                       checkpoint_name=chk,
        #                       num_processes_preprocessing=npp,
        #                       num_processes_segmentation_export=nps,
        #                       folder_with_segs_from_prev_stage=prev_stage_predictions,
        #                       num_parts=num_parts,
        #                       part_id=part_id,
        #                       device=device)

        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        # nnUNet 2.2.1
        if supports_keyword_argument(nnUNetPredictor, "perform_everything_on_gpu"):
            predictor = nnUNetPredictor(
                tile_step_size=self.step_size,
                use_gaussian=True,
                use_mirroring=not disable_tta,
                perform_everything_on_gpu=True,  # for nnunetv2<=2.2.1
                device=self.device,
                verbose=verbose,
                verbose_preprocessing=verbose,
                allow_tqdm=allow_tqdm,
            )
        # nnUNet >= 2.2.2
        else:
            predictor = nnUNetPredictor(
                tile_step_size=self.step_size,
                use_gaussian=True,
                use_mirroring=not disable_tta,
                perform_everything_on_device=True,  # for nnunetv2>=2.2.2
                device=self.device,
                verbose=verbose,
                verbose_preprocessing=verbose,
                allow_tqdm=allow_tqdm,
            )
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=self.folds,
            checkpoint_name=chk,
        )
        return predictor

    def process(
        self,
        _input: Union[str, Path, Nifti1Image],
        _output: Union[str, Path],
        crop_path: Optional[Path] = None,
    ):
        if not isinstance(_input, Nifti1Image):
            _input = Path(_input)

        if _output is not None:
            _output = Path(_output)
        else:
            if self.radiomics:
                raise ValueError("Output path is required for radiomics.")

        crop_path = _output if crop_path is None else crop_path

        nnUNet_predict_image(
            _input,
            _output,
            self.task_id,
            model=self.model,
            folds=self.folds,
            trainer=self.trainer,
            tta=False,
            multilabel_image=self.multilabel,
            resample=self.resample,
            crop=self.crop,
            crop_path=crop_path,
            task_name=self.task,
            nora_tag=self.nora_tag,
            preview=self.preview,
            nr_threads_resampling=self.nr_thr_resamp,
            nr_threads_saving=self.nr_thr_saving,
            force_split=self.force_split,
            crop_addon=self.crop_addon,
            roi_subset=self.roi_subset,
            output_type=self.output_type,
            statistics=self.statistics_fast,
            quiet=self.quiet,
            verbose=self.verbose,
            test=self.test,
            skip_saving=self.skip_saving,
            device=self.device,
            exclude_masks_at_border=not self.stats_include_incomplete,
            no_derived_masks=self.no_derived_masks,
            v1_order=self.v1_order,
            stats_aggregation="mean",
            remove_small_blobs=self.remove_small_blobs,
            normalized_intensities=False,
            predictor=self.predictor,
        )

        if self.task == "lung":
            # merge lung masks
            combined_mask = combine_masks(_output, "lung")
            nib.save(combined_mask, _output / "combined_lung.nii.gz")

            # remove original masks:
            for m in [
                "lung_upper_lobe_left",
                "lung_lower_lobe_left",
                "lung_upper_lobe_right",
                "lung_middle_lobe_right",
                "lung_lower_lobe_right",
            ]:
                mask_path = _output / f"{m}.nii.gz"
                if mask_path.exists():
                    os.remove(mask_path)

    def cleanup(self):
        """
        Clean up the environment by resetting torch settings.
        """
        torch.backends.cudnn.benchmark = self.initial_cudnn_benchmark
        torch.set_num_threads(self.initial_num_threads)


def segmentator_process(
    segmentator_instance: CustomSegmentator,
    input_folder: str,
    pid: Optional[str],
    study_uid: Optional[str],
    series_uid: Optional[str],
) -> Optional[str]:
    suffix = format_input_path(pid, study_uid, series_uid)

    suid_path = os.path.join(input_folder, suffix)
    if not os.path.exists(suid_path):
        return f"Path {suid_path} does not exist, skipping..."
    nii_file = [
        f for f in os.listdir(suid_path) if f.endswith(".nii") or f.endswith(".nii.gz")
    ]
    if len(nii_file) == 0:
        return f"No nii file in {suid_path}, skipping..."
    elif len(nii_file) > 1:
        print(f"Warning: more than one nii file in {suid_path}, using the first one.")
    nii_path = os.path.join(suid_path, nii_file[0])

    try:
        print(f"Processing {suid_path}... saving to {os.path.join(suid_path, 'segmentations')}")
        segmentator_instance.process(
            _input=nii_path, _output=os.path.join(suid_path, "segmentations")
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error processing {suid_path}: {e}")

    return None


def segmentator_batch_process(df, task, source_folder, max_worker=4, logger=None):
    if logger is None:
        logger = get_logger("./logs", name=f"segmentator_{task}")
    segmentator_pool = []

    available_gpus = torch.cuda.device_count()

    for i in range(max_worker):
        _ = CustomSegmentator(task=task, device=f"cuda:{i % available_gpus}")
        segmentator_pool.append(_)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=max_worker) as executor:
        # Initialize futures list
        futures = []

        # Initialize progress bar
        from tqdm import tqdm

        progress_bar = tqdm(total=df.shape[0], desc="Total Progress", lock_args=(True,))

        for index, row in df.iterrows():
            # Submit the task to the executor
            pid = row["PatientID"] if "PatientID" in row else None
            study_uid = row["StudyInstanceUID"] if "StudyInstanceUID" in row else None
            series_uid = row["SeriesInstanceUID"] if "SeriesInstanceUID" in row else None
            future = executor.submit(segmentator_process, segmentator_pool[index], source_folder, pid, study_uid, series_uid)
            futures.append(future)

        # Process as tasks complete
        for future in as_completed(futures):
            # Get result or exception from future if needed
            error = future.result()
            if error is not None:
                logger.error(error)

            # Update the progress bar
            progress_bar.update(1)


def load_nifti(file_path):
    return nib.load(file_path).get_fdata()


# overwrite annotation_with_path
def segmentation2bbox_series_process(    
        input_folder: str,
        pid: Optional[str],
        study_uid: Optional[str],
        series_uid: Optional[str],
        task="lung"
    ):
    
    suffix = format_input_path(pid, study_uid, series_uid)

    segmentation_path = os.path.join(input_folder, suffix, "segmentations")

    # Check if segmentations folder and annotations file exist
    if not os.path.exists(segmentation_path):
        # Save StudyInstanceUID and SeriesInstanceUID to DataFrame
        logging.info(f"Ignore mask unavailable case: {segmentation_path} does not exist.")
        return None

    if task == "lung":
        task_masks = [
            "combined_lung.nii.gz",
        ]
    else:
        raise ValueError(f"Unsupported task: {task}")

    # segmentation
    # (xmin, ymin, zmin), (xmax, ymax, zmax)
    rois = {}
    for mask in task_masks:
        name = mask.split(".")[0].split("_")[-1]
        p = os.path.join(segmentation_path, mask)
        if not os.path.isfile(p):
            logging.error(f"unavailable mask {name} path {p}")
            rois[name] = [-1, -1, -1, -1, -1, -1]
        try:
            mask_data = load_nifti(p)
            # Get bounding box for lung segmentation
            where = np.array(np.where(mask_data))
            (xmin, ymin, zmin), (xmax, ymax, zmax) = np.min(where, axis=1), np.max(
                where, axis=1
            )
            rois[name] = [xmin, ymin, zmin, xmax, ymax, zmax]
        except:
            print(f"error with mask {name} path {p}")

    return_dict = {
        "PatientID": pid,
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
    }

    return_dict.update(rois)
    return return_dict

def segmentation2bbox_batch_process(df, task, input_folder, save_path=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    df = df.copy()

    results = []
    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = []

        # Submit tasks to the executor
        for _, row in df.iterrows():
            pid = row["PatientID"] if "PatientID" in row else None
            study_uid = row["StudyInstanceUID"] if "StudyInstanceUID" in row else None
            series_uid = row["SeriesInstanceUID"] if "SeriesInstanceUID" in row else None
            # Submit the processing function to the executor
            futures.append(
                executor.submit(
                    segmentation2bbox_series_process, input_folder, pid, study_uid, series_uid, task
                )
            )

        # Process as tasks complete
        for future in tqdm(as_completed(futures), 
                          total=len(futures), 
                          desc="Segmentation2BBOX Progress"):
            # Properly handle return value
            result = future.result()
            if result is not None:  # Check if result is not None
                results.append(result)

    for _r in results:
        _series_uid = _r["SeriesInstanceUID"]
        segments = _r.keys() - {"PatientID", "StudyInstanceUID", "SeriesInstanceUID"}

        for seg in segments:
            df.loc[df["SeriesInstanceUID"] == _series_uid, seg] = str(_r[seg])

    # Save the DataFrame to a CSV file
    if save_path is not None:
        output_csv_path = save_path if save_path.endswith(".csv") else os.path.join(save_path, f"{task}_labels_bbox.csv")
    else:
        output_csv_path = f"{input_folder}/{task}_labels_bbox.csv"
        
    if not os.path.exists(os.path.dirname(output_csv_path)):
        os.makedirs(os.path.dirname(output_csv_path))
    print(f"Saving segmentation bounding boxes to {output_csv_path}")
    df.to_csv(output_csv_path, index=False)
    return df
