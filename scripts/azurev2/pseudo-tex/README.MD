# Pseudo TeX

- [Pseudo TeX](#pseudo-teX)
  - [Introduction](#introduction)
  - [Instructions](#instructions)

![Pseudo TeX](../../../media/pseudo-tex.png)

## Introduction

Pseudo-TeX is based on the [HADAR](https://www.nature.com/articles/s41586-023-06174-6) paper. 
This new format combines temperature, material emissivity, and texture in one image. 
The Pseudo-TeX image created here follows the Pseudo-TeX description discussed in the supplementary material of the paper, it is an approximated version of the TeX image.
The Pseudo-TeX image uses the thermal image as an approximation of temperature, an equalisation of the thermal image is used to appoximate texture, and a semantic segmentation of the equivalent RGB image using Mask2former approximates the material emissivity. 


## Instructions

To output Pseudo-TeX images of your dataset on Azure ML, follow these instructions.

- In your dataset's root folder, create two separate folders for your thermal and RGB images. Call the subfolders `thermal` and `rgb`. 
Make sure the filenames are exactly the same for paired images.

- Upload your dataset on azure using the following command

    `python scripts/azurev2/upload_dataset.py --dataset-name <dataset_name> --version <version> --dataset-path <dataset_path> --description <description>`

- Run the following command to log your Pseudo-TeX images of the dataset on Azure

    `python scripts/azurev2/pseudo-tex/pseudo-tex-azure.py --dataset-name <dataset_name> --version <version>`



