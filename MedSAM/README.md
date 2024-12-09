****Prostate MRI Segmentation with MedSAM****
This branch focuses on using the pre-trained MedSAM model to segment the prostate on the Prostate-MRI-US-Biopsy dataset and fine-tuning MedSAM to improve segmentation performance.

**Repository Structure**
**- MedSAM Folder**
The MedSAM folder contains the files for the MedSAM model. This model is sourced from: https://github.com/bowang-lab/MedSAM

**- Key Scripts**
Mask_generation_Medsam.py
This script applies the MedSAM model to the Prostate-MRI-US-Biopsy dataset for prostate segmentation.

FineTune_MedSAM_Prostate.py
This script fine-tunes the MedSAM model on the Prostate-MRI-US-Biopsy dataset to enhance segmentation accuracy.

Compare_mask.py
This script compares the MedSAM segmentation results with the ground truth masks to evaluate consistency and performance.

Preprocessing Folder
The Preprocessing folder contains scripts for dataset preprocessing:
1.DCM_to_NRRD.py
  Converts prostate MR images from DICOM format to NRRD format. This script is intended to run in 3D Slicer.
2.convert_STL_labelmap_slicer_script.py
  Uses the NRRD files generated in the previous step as reference images to convert STL format mask files into NIfTI (nii.gz) format. This ensures spatial correspondence between images and masks at each slice.