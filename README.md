# Rebel Nerf

Rebel-NeRF allows running [NeRFStudio](https://docs.nerf.studio/en/latest/), a simple API for an end-to-end NeRF pipeline, on Azure.

![synthetic video](media/synthesized_video_rgb_facade.gif)

- [Rebel Nerf](#rebel-nerf)
  - [Install for development](#install-for-development)
  - [Data preparation](#data-preparation)
  - [NeRF Training](#nerf-training)
  - [Fly-through video rendering](#fly-through-video-rendering)
  - [Environment](#environment)
  - [Compute Requirements](#compute-requirements)

## Install for development

For a simple install for development, install the editable version `pip install -e .`.

For developing on nerfstudio, we recommend that you install the library in the local root folder of rebel_nerf, so upload of the modifications to Azure is streamlined.
Clone nerfstudio in the root folder and install with `pip install -e . --config-settings editable_mode=compat` (the compat mode is needed for pylance to bind to the editable version installed with setuptools).
If you plan to do modifications, use [our fork of nerfstudio](git@ssh.dev.azure.com:v3/devsdb/CRD-NT_Insulated/nerfstudio).

## Data preparation

A NeRF model requires a set of images and their corresponding poses as input for training.

This stage takes as input a set of images, assumed to be saved in the default blob store of the building-mind-workspace on [Microsoft Azure Machine Learning Studio](https://ml.azure.com/data/datastore/workspaceblobstore/).

The following creates from the set of images a `transforms.json` file containing the relative pose of each image, alongside a folder `images` containing a downscaled version of the original images, which can make [NeRF training](#nerf-training) go faster.
The `--downscale-factor` is defaulted to one and can be changed as a CLI argument.

```bash
python3 scripts/azurev2/images_to_nerf_dataset_azure.py --scene-name <scene_name> --version <version_number>
```

the `--scene-name` argument specifes the name of the scene being to reconstruct through NeRF, and should be consistent throughout the 3 stages.

At the end of this stage, it is necessary to register the image data asset manually in Azure ML Studio.
Name the data asset `<scene_name>-images-to-nerf-dataset` where `scene_name` is the argument chosen before.

## NeRF Training

The following will train a nerfacto model, the recommended model for real world scenes.

```bash
python3 scripts/azurev2/train_on_azure.py --scene-name <scene_name> --version <version_number>
```

Make sure to have the same `--scene-name` argument to that of the previous stage. 
By default, the maximum number of iterations is 30000. 
Change it using `--max-num-iterations`.

To view the scene reconstruction/training happen in real-time, check in the outputs+logs tab, click on the file std_log.txt (which should appear ~10 mn after you start training) and visit the URL link.
Since you are running on a remote machine, you will need to port forward the websocket port (defaults to 7007).

To do so, you must first make sure that the training job is running on a compute cluster with SSH access enabled.
After training starts, in the overview of your job, go in the compute section and click on your compute instance. Check in the Nodes tab and copy the connection string.

The following should appear:

![webviewer port tunnel](media/ssh_pipeline.png)

The connection string has the shape ```ssh username@xx.xxx.xxx.xxx -p <connection_port>```, where `x` are ints.
In our case, the username is `insulated` and the connection port will be 50000 or 50001.
Finally, in a terminal, write the following command using the informations from your connection string (replace all `x` by the good values):

```bash
ssh -L 7007:localhost:7007 insulated@xx.xxx.xxx.xxx -p 5000x
```

Keep the terminal running after connecting, then navigate back to the link in your terminal and access it.
The webviewer should now work (if not, open the link again).

Once training is done, register the model if you want to use it to render a video, using the command printed at the end of the terminal from which you launched the job on azure.

## Fly-through video rendering

The trained NeRF model can now render a flythrough video of the scene.
You must first create a trajectory for the camera to follow.
This can be done in the viewer under the "RENDER" tab.
Orient your 3D view to the location where you wish the video to start, then press "ADD CAMERA".
This will set the first camera key frame.
Continue to new viewpoints adding additional cameras to create the camera path.

Once satisfied with the path, click on the `EXPORT PATH` button to download a `camera_path.json` file with the interpolated poses following your trajectory.
Rename and save this file in the scripts/trajectories/ folder.

The following will then render a video of your custom trajectory:

```bash
python3 scripts/azurev2/render_video_on_azure.py --scene-name <scene_name> --data-version <data_version> --model-version <model_version> --camera-path <path_to_camera_path_json>
```

## Environment

The base docker image used can be found [on github](https://github.com/Azure/AzureML-Containers/tree/master/base/gpu).
Navigate to the `openmpi4.1.0-cuda11.3-cudnn8-ubuntu20.04` folder and you can find the differnet versions available of the image `mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.3-cudnn8-ubuntu20.04` through the `release-notes.md` file provided within.

## Compute Requirements

To run this Repository on Azure, make sure to have a compute cluster with a GPU of architecture = 75+ (i.e. Turing GPU or later).
This is needed to utilize fused MLPs for fast training.

The below figure shows how to create a suitable compute cluster on Azure:

![compute cluster](media/compute_cluster.PNG)
