import logging
from dataclasses import dataclass
from pathlib import Path

import tyro
from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure_ml_utils.azure_connect import get_client

from scripts.azurev2.nerf_dataset import get_eval_dataset_from_scene_name


@dataclass
class ImagesToNerfDatasetParameters:
    scene_name: str
    """Name of the scene/object to be assigned to the registered output of the \
       script on Azure"""
    version: str
    environment: str = "rebel-nerf-backbone"
    environment_version: str = "@latest"
    eval: bool = False
    eval_data_version: str = "1"


def main():
    parameters = tyro.cli(ImagesToNerfDatasetParameters)

    # Get a handle to the workspace
    ml_client = get_client()
    dataset = ml_client.data.get(name=parameters.scene_name, version=parameters.version)

    job_inputs = dict(
        scene_name=parameters.scene_name,
        data=Input(
            type=AssetTypes.URI_FOLDER,  # type: ignore
            path=dataset.path,
        ),
    )
    if parameters.eval:
        eval_dataset_path = get_eval_dataset_from_scene_name(
            ml_client=ml_client,
            scene_name=parameters.scene_name,
            version=parameters.eval_data_version,
        ).path
        job_inputs["eval_data_path"] = Input(
            type=AssetTypes.URI_FOLDER, path=eval_dataset_path  # type: ignore
        )

    output_path = "azureml://datastores/workspaceblobstore/paths/" + str(
        Path("rebel-nerf", parameters.scene_name + "-images-to-nerf-dataset")
    )

    job_outputs = {
        "output_dataset": Output(type=AssetTypes.URI_FOLDER, path=output_path)
    }
    cmd = (
        "python3.10 scripts/images_to_nerf_dataset.py "
        "--matching_method exhaustive "
        "--data ${{inputs.data}} "
        "--output-dir ${{outputs.output_dataset}} "
        "--num-downscales 0"
    )
    cmd += " --eval-data ${{inputs.eval_data_path}}" if parameters.eval else ""

    job = command(
        inputs=job_inputs,
        outputs=job_outputs,
        code=".",  # location of source code
        command=cmd,
        environment=parameters.environment + parameters.environment_version,
        compute="nerf-T4-gpu-aml",
        experiment_name="nerfstudio-colmap-experiment",
    )

    returned_job = ml_client.create_or_update(job)

    aml_url = returned_job.studio_url
    logging.info("job link:", aml_url)

    logging.info(
        "Once the job is finished, verify you data and register the data asset either"
        "in python or in the web interface: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-data-assets?tabs=Python-SDKsing"  # noqa
        "See: https://github.com/Azure/azure-sdk-for-python/issues/26618 for updates"
        " on when we will be able to do it programmatically."
    )


if __name__ == "__main__":
    main()
