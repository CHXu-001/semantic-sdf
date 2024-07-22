from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tyro
from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure_ml_utils.azure_connect import get_client


@dataclass
class RenderingParameters:
    dataset_name: str
    "Test dataset"
    data_version: str
    model_name: str
    """Name of the scene/object to be assigned to the registered output of the \
       script on Azure"""
    model_version: str
    environment: str = "rebel-nerf-backbone"
    environment_version: str = "@latest"


def main() -> None:
    parameters = tyro.cli(RenderingParameters)
    ml_client = get_client()
    dataset = ml_client.data.get(
        name=parameters.dataset_name, version=parameters.data_version
    )

    job_inputs = dict(
        model=Input(
            type=AssetTypes.MLFLOW_MODEL,  # type: ignore
            path=f"azureml:{parameters.model_name}:{parameters.model_version}",
        ),
        data=Input(
            type=AssetTypes.URI_FOLDER,  # type: ignore
            path=dataset.path,
        ),
    )

    output_path = "azureml://datastores/workspaceblobstore/paths/" + str(
        Path("image-translator", parameters.dataset_name + "-translated")
    )
    job_outputs = {
        "output_dataset": Output(type=AssetTypes.URI_FOLDER, path=output_path)
    }

    job_name = (
        "load-"
        + parameters.model_name
        + "-"
        + datetime.now().strftime("%d-%m-%Y-%H%M%S")
    )

    job = command(
        code=".",
        inputs=job_inputs,
        outputs=job_outputs,
        environment=parameters.environment + parameters.environment_version,
        compute="nerf-T4-ssh-gpu",
        command=(
            "python3.10 scripts/test_image_translator.py "
            "--model-uri ${{inputs.model}} "
            "--data-root ${{inputs.data}} "
            "--output-dir ${{outputs.output_dataset}} "
        ),
        experiment_name="test_mlflow_loading",
        display_name=job_name,
        name=job_name,
    )

    returned_job = ml_client.jobs.create_or_update(job)

    assert returned_job.services is not None
    returned_job.services["Studio"].endpoint


if __name__ == "__main__":
    main()
