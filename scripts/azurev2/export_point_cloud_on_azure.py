from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tyro
from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure_ml_utils.azure_connect import get_client

from scripts.azurev2.nerf_dataset import get_dataset_from_scene_name


@dataclass
class ExportParameters:
    scene_name: str
    """Name of the scene/object to be assigned to the registered output of the \
       script on Azure"""
    data_version: str
    """Version of the dataset"""
    model_version: str
    """Version of the model"""
    model_type: str
    """Type of the model"""
    output_dir: Path = Path("./outputs")
    environment: str = "rebel-nerf-backbone"
    """Environment"""
    environment_version: str = "latest"
    experiment_name: str = "nerfstudio-export-experiment"
    """Experiment name in azure"""


def main() -> None:
    parameters = tyro.cli(ExportParameters)

    ml_client = get_client()

    dataset = get_dataset_from_scene_name(
        ml_client=ml_client,
        scene_name=parameters.scene_name,
        version=parameters.data_version,
    )

    job_inputs = {
        "dataset_path": Input(
            type=AssetTypes.URI_FOLDER,  # type: ignore
            path=dataset.path,
        ),
        "model": Input(
            type=AssetTypes.CUSTOM_MODEL,  # type: ignore
            path=f"azureml:{parameters.scene_name}-nerf:{parameters.model_version}",
        ),
        "model_type": parameters.model_type,
    }

    job_outputs = {"export_output": Output(type=AssetTypes.URI_FOLDER)}  # type: ignore

    cmd = "python3.10 "

    cmd += (
        "scripts/export_point_cloud.py "
        "--dataset ${{inputs.dataset_path}} "
        "--model-uri ${{inputs.model}} "
        "--model-type ${{inputs.model_type}} "
        "--output-dir ${{outputs.export_output}} "
    )

    job_name = parameters.scene_name + "-" + datetime.now().strftime("%d-%m-%Y-%H%M%S")

    job = command(
        inputs=job_inputs,
        outputs=job_outputs,
        code=".",  # location of source code
        command=cmd,
        environment=parameters.environment + "@" + parameters.environment_version,
        compute="nerf-a100",
        experiment_name=parameters.experiment_name,
        display_name=job_name,
        name=job_name,
    )

    job = ml_client.create_or_update(job)


if __name__ == "__main__":
    main()
