# Pulmonary Embolism Detection with Semi‑Weakly Supervised Learning

![Pipeline Overview](/docs/pipeline.jpg?raw=true "Pipeline Overview")

## Overview
This repository contains code and pretrained models for **pulmonary embolism (PE) detection** from CTPA scans using a **semi‑weakly supervised** training strategy that reduces the need for dense slice‑level annotations.

- **Datasets**: RSNA Pulmonary Embolism CT Dataset (RSPECT) for development; evaluated on RSNA Public/Private test sets, AIDA & FUMPE, and SMH cohorts.
- **Goals**: High performance with substantially fewer labels; robust generalization across sites.

> If you use this repository, please cite the paper listed in **Citation**.

---

## Features
- End‑to‑end training scripts (stepwise pipeline).
- Mixed slice‑level and exam‑level supervision with configurable loss weights.
- Flexible 2D/2.5D model architectures (ResNet, ViT, etc backbones).
- Optional lung masking and PE candidate region visualization.

---

## Repository Structure
```
.
├── scripts/
│   ├── ENVS.py # environment variable helper
│   ├── training_step_0_format_labels.py
│   ├── training_step_1_convert_to_nifti.py
│   ├── training_step_2_prepare_slice_level_labels.py
│   ├── training_step_3_lung_segment_on_nifti.py
│   ├── training_step_4_save_mask_bbox2df.py
│   └── training_step_5_run.py
├── notebooks/
│   ├── 1.inference_single_ct_scan_to_pe_pred.ipynb
│   └── 2.inference_batch_processing.ipynb
├── data/
│   └── patient_1
|       └── DICOM files...
├── models/
│   └── pe/
│       └── pretrained_weights.pth # example weights
├── docs/
│   └── pipeline.jpg
├── requirements.txt
└── README.md
```
> File layout may vary slightly; adjust paths in configs accordingly.

---

## Quickstart
### 1) Environment
Use conda or venv. Python ≥ 3.10 recommended.
```bash
# with conda
conda create -n pe-semiweak python=3.10 -y
conda activate pe-semiweak
pip install -r requirements.txt 
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

### 2) Data Format
Download and extract the RSNA Pulmonary Embolism CT Dataset from [Kaggle](https://www.kaggle.com/competitions/rsna-str-pulmonary-embolism-detection/data). Arrange files as:
```
data/
├── patient_1/               # DICOM files (organized by patient/study)
├── patient_2/
├── ...
└── train.csv            # Labels and metadata
```

The csv file should contain columns like:
- `StudyInstanceUID`: DICOM study UID
- `SeriesInstanceUID`: DICOM series UID
- `SOPInstanceUID`: DICOM slice UID
- `pe_present_on_image`: binary exam-level PE label
- `negative_exam_for_pe`: binary exam-level negative label (will be converted to `pe_present_in_exam` internally)

### 3) Configure Paths
Edit `scripts/ENVS.py` to set:


```python
# scripts/ENVS.py
# _path is the root directory of this repository
TRAINING_DATA_NAME = "sample_train.csv"
TRAINING_DATA_NAME_SLICE = "sample_train_slices.csv"
TRAINING_DATA_NAME_COMPLETED = "sample_train_completed.csv"
RSNA_DATA_LOCATION = os.path.join(_path, "data")
NIFTI_FORMATTED_DATA_LOCATION = os.path.join(_path, "nifti_data")
```

## Training Pipeline (Step‑by‑Step)
> Run steps in order; each step writes outputs used by the next.

**Step 0 — Format labels**
```bash
python scripts/training_step_0_format_labels.py
```
Produces a normalized label file (exam/slice indices, targets, splits).

**Step 1 — Convert DICOM → NIfTI**
```bash
python scripts/training_step_1_convert_to_nifti.py
```
Ensures consistent orientation, spacing, and metadata.

**Step 2 — Prepare slice‑level labels**
```bash
python scripts/training_step_2_prepare_slice_level_labels.py
```
Materializes slice supervision used by the semi‑weak loss.

**Step 3 — Lung segmentation **
```bash
python scripts/training_step_3_lung_segment_on_nifti.py
```
Generates lung masks to focus the model on relevant anatomy. Segmentation is based on the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) tool. Core code of the segmentator is splited and modified to fit fast preprocessing of the PE dataset. Please also cite it if you use this code in your work.

**Step 4 — Save mask‑bounded boxes to DataFrame**
```bash
python scripts/training_step_4_save_mask_bbox2df.py
```
Computes per‑slice bounding boxes/crops for efficient training.

**Step 5 — Train**
```bash
python scripts/training_step_5_run.py
```
ddp training can be enabled with
```bash
sh scripts/ddp_run.sh
```
Change [Config]([pe_final.py](configs/pe/pe_final.py)) options in `training_step_5_run` before running.

Monitors validation metrics and saves checkpoints to `WORKDIR/experiment_name/`.

---

## Inference
> Example command; adjust to match your inference script/signature if different.
> 
**Outputs** may include exam‑level PE probability, and optional heatmaps/crops.

Please refer to the notebooks in `notebooks/` for inference examples:
- `1.inference_single_ct_scan_to_pe_pred.ipynb`: single study inference
- `2.inference_batch_processing.ipynb`: batch inference on multiple studies

An example model checkpoint is provided in this [link](https://drive.google.com/file/d/1jyzuPnvaJFH5Ee4xPzEpOQl9NIA8iADW/view?usp=sharing)

---

## Configuration
Most options are set in configs;
Override examples:
```python
pe_cfg = PELabelMask()
pe_cfg.ddp = False # set to True if you want to use distributed data parallel

# manual modification of cfg setup:
pe_cfg.model_name = "coatnet"
pe_cfg.masked_label_portion = 0.9
pe_cfg.batch_size = 2

pe_cfg.img_size = 96
pe_cfg.img_depth = 32  # number of slices per input volume

pe_cfg.folds = 1

# then start training pipeline
# pipeline(cfg)
```

---

## How to Use (TL;DR)
1. **Install** dependencies and set up environment.
2. **Prepare** data (`Step 0–2`).
3. **Segmentations** run lung masking (`Step 3`) and bbox extraction (`Step 4`).
4. **Train** (`Step 5`) and **select** the best checkpoint.
5. **Infer** on new studies with the chosen checkpoint.

---

## Acknowledgements
This work builds on the RSNA Pulmonary Embolism CT Dataset and community tools for medical image processing (DICOM/NIfTI ecosystems). We thank collaborators and clinical partners for their support.

---

## License
See `LICENSE` for terms. By default, models and code are intended for **research use**. For clinical/commercial use, obtain appropriate approvals and review licensing.

---

## Citation
If you use this code or models in your research, please cite the following paper:
```
@article{hu_high_2025,
	title = {High performance with fewer labels using semi-weakly supervised learning for pulmonary embolism diagnosis},
	volume = {8},
	rights = {2025 The Author(s)},
	issn = {2398-6352},
	url = {https://www.nature.com/articles/s41746-025-01594-2},
	doi = {10.1038/s41746-025-01594-2},
	number = {1},
	journaltitle = {npj Digital Medicine},
	shortjournal = {npj Digit. Med.},
	author = {Hu, Zixuan and Lin, Hui Ming and Mathur, Shobhit and Moreland, Robert and Witiw, Christopher D. and Jimenez-Juan, Laura and Callejas, Matias F. and Deva, Djeven P. and Sejdić, Ervin and Colak, Errol},
	langid = {english},
	note = {Publisher: Nature Publishing Group},
	keywords = {Computed tomography, Cardiovascular diseases, Computer science, Medical research},
}
```