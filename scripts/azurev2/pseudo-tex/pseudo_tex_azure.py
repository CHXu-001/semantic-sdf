import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tyro
from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes

from azure_ml_utils.azure_connect import get_client


@dataclass
class PseudoTexParameters:
    dataset_name: str
    """Name of the scene/object to be assigned to the registered output of the
       script on Azure"""
    version: str = "1"
    """Version of dataset on Azure"""
    environment: str = "pseudo-tex"
    """Environment"""
    environment_version: str = "latest"
    compute_node: str = "rebel-nerf-t4-ssh"


def main() -> None:
    logging.basicConfig()
    pseudo_tex_logger = logging.getLogger("rebel_nerf")
    pseudo_tex_logger.setLevel(logging.INFO)

    parameters = tyro.cli(PseudoTexParameters)

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

    job_name = "pseudo_tex_" + datetime.now().strftime("%d-%m-%Y-%H%M%S")

    output_path = "azureml://datastores/workspaceblobstore/paths/" + str(
        Path(str(parameters.dataset_name), "pseudo_tex_images")
    )

    job_outputs = {
        "output_dataset": Output(type=AssetTypes.URI_FOLDER, path=output_path)
    }

    job = command(
        inputs=job_inputs,
        outputs=job_outputs,
        code=".",  # location of source code
        command=(
            "python3.10  scripts/create_pseudo_tex.py "
            "--input_data_file  ${{inputs.data}} "
            "--output_file ${{outputs.output_dataset}}"
        ),
        environment=parameters.environment + "@" + parameters.environment_version,
        compute=parameters.compute_node,
        experiment_name="pseudo-tex-experiment",
        display_name=job_name,
        name=job_name,
    )

    job = ml_client.create_or_update(job)


if __name__ == "__main__":
    main()
