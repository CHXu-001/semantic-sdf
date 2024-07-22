import logging
from dataclasses import dataclass
from datetime import datetime

import tyro
from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure_ml_utils.azure_connect import get_client


@dataclass
class TrainingParameters:
    dataset_name: str
    """Name of the scene/object to be assigned to the registered output of the \
       script on Azure"""
    version: str = "1"
    """Version of dataset on Azure"""
    environment: str = "rebel-nerf-backbone"
    """Environment"""
    environment_version: str = "latest"
    compute_node: str = "nerf-T4-gpu-aml"


def main() -> None:
    logging.basicConfig()
    rgb2thermal_translator_logger = logging.getLogger("rebel_nerf")
    rgb2thermal_translator_logger.setLevel(logging.INFO)

    parameters = tyro.cli(TrainingParameters)

    ml_client = get_client()
    dataset = ml_client.data.get(
        name=parameters.dataset_name, version=parameters.version
    )

    job_inputs = dict(
        dataset_name=parameters.dataset_name,
        data=Input(
            type=AssetTypes.URI_FOLDER,  # type: ignore
            path=dataset.path,
        ),
    )

    job_name = (
        parameters.dataset_name + "-" + datetime.now().strftime("%d-%m-%Y-%H%M%S")
    )

    my_job_outputs = {
        "custom_model_output": Output(type=AssetTypes.CUSTOM_MODEL)  # type: ignore
    }

    job = command(
        inputs=job_inputs,
        outputs=my_job_outputs,
        code=".",  # location of source code
        command=(
            "python3.10 scripts/train_image_translator_script.py "
            "--dataset-name ${{inputs.dataset_name}} "
            "--data-root ${{inputs.data}} "
            "--output-dir ${{outputs.custom_model_output}} "
        ),
        environment=parameters.environment + "@" + parameters.environment_version,
        compute=parameters.compute_node,
        experiment_name="rgb2thermal-translator-train-experiment",
        display_name=job_name,
        name=job_name,
    )

    job = ml_client.create_or_update(job)


if __name__ == "__main__":
    main()
