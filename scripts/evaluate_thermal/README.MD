# Evaluate Thermal Images

- [Evaluate Thermal Images](#evaluate-thermal-images)
  - [Introduction](#introduction)
  - [Instructions](#instructions)


## Introduction

This folder evaluates thermal images by plotting a graph showing the temperature values across the images.
There's typically a baseline method and another method that are both plotted on the graph for comparison. 

There are two methods to do the evaluation: by tracking or by masking.
Tracking uses a saved tensor file with tracked pixel location across the images. 
Masking uses a set of input masks images and the average temperature values of the mask is plotted on the graph. 

## Instructions

To evaluate your thermal images against a baseline, make sure to save the images of each method in a separate folder. 

If you have tracked pixels, run the file `evaluate.py` and assign the parameter `mask_available` to False. 

If you have masked images, run the file `evaluate.py` and assign the parameter `mask_available` to True. 

If you want to generate your own masks, run the file `mask_images.py` and annotate the images by clicking on the pixels you're interested to evaluate. 
The masked pixels should be coloured in purple and the mask generate is a binary image,

