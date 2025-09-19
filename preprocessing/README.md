### Preprocessing Module

This module is responsible for preprocessing the data before it is fed into the model. It includes functions for:

- Converting DICOM files to NIfTI format
- Running TotalSegmentator for segmentation

Segmentation is based on the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) tool. Core code of the segmentator is splited and modified to fit fast preprocessing of the PE dataset.

Please cite also cite it if you use this code in your work.