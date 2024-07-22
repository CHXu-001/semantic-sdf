from dataclasses import dataclass
from datetime import datetime

import tyro
from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes
from azure_ml_utils.azure_connect import get_client

from scripts.azurev2.nerf_dataset import get_dataset_from_scene_name


@dataclass
class AccessParameters:
    scene_name: str
    """Name of the scene/object to be assigned to the registered output of the \
       script on Azure"""
    data_version: str
    """The version of dataset corresponding to `scene_name`"""
    model_version: str
    """The version of registered model used to do rendering"""
    environment: str = "rebel-nerf-backbone"
    environment_version: str = "@latest"


def main() -> None:
    param = tyro.cli(AccessParameters)

    ml_client = get_client()

    job_inputs = {
        "model": Input(
            type=AssetTypes.CUSTOM_MODEL,  # type: ignore
            path=f"azureml:{param.scene_name}-nerf:{param.model_version}",
        ),
    }

    cmd = "python3.10 scripts/access_sdf_model.py --model-uri ${{inputs.model}} "

    dataset_path = get_dataset_from_scene_name(
        ml_client=ml_client, scene_name=param.scene_name, version=param.data_version
    ).path
    assert dataset_path is not None
    job_inputs["dataset_path"] = Input(
        type=AssetTypes.URI_FOLDER, path=dataset_path  # type: ignore
    )

    cmd += "--dataset_path ${{inputs.dataset_path}} "

    job_name = param.scene_name + "-" + datetime.now().strftime("%d-%m-%Y-%H%M%S")

    job = command(
        code=".",
        inputs=job_inputs,
        environment=param.environment + param.environment_version,
        # compute="nerf-T4-ssh-gpu",
        compute="nerf-a100",
        command=cmd,
        experiment_name="nerfstudio-access-experiment",
        display_name=job_name,
        name=job_name,
    )

    returned_job = ml_client.jobs.create_or_update(job)

    assert returned_job.services is not None
    returned_job.services["Studio"].endpoint


if __name__ == "__main__":
    main()
