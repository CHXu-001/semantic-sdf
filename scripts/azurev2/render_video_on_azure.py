from dataclasses import dataclass
from datetime import datetime
from typing import Any

import tyro
from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes
from azure_ml_utils.azure_connect import get_client

from scripts.azurev2.nerf_dataset import get_dataset_from_scene_name


@dataclass
class RenderingParameters:
    scene_name: str
    """Name of the scene/object to be assigned to the registered output of the \
       script on Azure"""
    data_version: str
    """The version of dataset corresponding to `scene_name`"""
    model_version: str
    """The version of registered model used to do rendering"""
    camera_path: str = "./scripts/trajectories/camera_path.json"
    """Path to the file camera_path.json"""
    save_images: bool = True
    """Save rendered images for evaluation."""
    save_video: bool = True
    """Save rendered video."""
    downscale_factor: int = 4
    """Downscale factor of the resolution of rendered views with respect to the \
        original trianing images."""
    seconds: int = 0.05
    """Time spent on one image in the GIF"""
    environment: str = "rebel-nerf-backbone"
    environment_version: str = "@latest"
    debug: bool = False

    def get_inputs(self, ml_client: MLClient) -> tuple[dict[str, Any], str]:
        job_inputs = {
            "model": Input(
                type=AssetTypes.CUSTOM_MODEL,  # type: ignore
                path=f"azureml:{self.scene_name}-nerf:{self.model_version}",
            ),
            "save_video": self.save_video,
            "save_images": self.save_images,
            "downscale_factor": self.downscale_factor,
            "seconds": self.seconds,
            "scene_name": self.scene_name,
        }
        command = "python3.10 "

        if self.debug:
            command += "-m debugpy --listen localhost:5678 --wait-for-client "

        command += (
            "scripts/render_video_script.py "
            "--model-uri ${{inputs.model}} "
            "--downscale-factor ${{inputs.downscale_factor}} "
            "--seconds ${{inputs.seconds}} "
        )
        if self.save_images:
            command += "--save-images "
        if not self.save_video:
            command += "--no-save-video "

        dataset_path = get_dataset_from_scene_name(
            ml_client=ml_client, scene_name=self.scene_name, version=self.data_version
        ).path
        assert dataset_path is not None
        job_inputs["dataset_path"] = Input(
            type=AssetTypes.URI_FOLDER, path=dataset_path  # type: ignore
        )

        command += "--dataset_path ${{inputs.dataset_path}} "

        job_inputs["camera_path"] = Input(
            type=AssetTypes.URI_FILE, path=self.camera_path
        )
        command += "--camera-path-filename ${{inputs.camera_path}} "

        return job_inputs, command


def main() -> None:
    parameters = tyro.cli(RenderingParameters)

    ml_client = get_client()

    job_inputs, cmd = parameters.get_inputs(ml_client)

    job_name = parameters.scene_name + "-" + datetime.now().strftime("%d-%m-%Y-%H%M%S")

    job = command(
        code=".",
        inputs=job_inputs,
        environment=parameters.environment + parameters.environment_version,
        compute="nerf-T4-ssh-gpu",
        command=cmd,
        experiment_name="nerfstudio-render-experiment",
        display_name=job_name,
        name=job_name,
    )

    returned_job = ml_client.jobs.create_or_update(job)

    assert returned_job.services is not None
    returned_job.services["Studio"].endpoint


if __name__ == "__main__":
    main()