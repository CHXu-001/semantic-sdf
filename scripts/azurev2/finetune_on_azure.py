import logging
from dataclasses import dataclass
from datetime import datetime

import tyro
from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes
from azure_ml_utils.azure_connect import get_client

from scripts.azurev2.nerf_dataset import get_dataset_from_scene_name


@dataclass
class TrainingParameters:
    scene_name: str
    """Name of the scene/object to be assigned to the registered output of the \
       script on Azure"""
    data_version: str
    model_name: str
    """Name of the registered model on Azure to resume before finetuning"""
    model_version: str
    model_relative_path: str
    """The relative path of the pre-trained checkpoint for the registered model
    on azureml, e.g.,
    /Building2-Images-training-job/nerfacto/2023-08-11_123540/nerfstudio_models"""
    model_type: str = "nerfacto"
    """What NeRF model to train. Defaults to Nerfacto"""
    environment: str = "rebel-nerf-backbone"
    """Environment"""
    environment_version: str = "latest"
    max_num_iterations: int = 30000


def main() -> None:
    logging.basicConfig()
    rebel_nerf_logger = logging.getLogger("rebel_nerf")
    rebel_nerf_logger.setLevel(logging.INFO)

    parameters = tyro.cli(TrainingParameters)

    ml_client = get_client()
    dataset = get_dataset_from_scene_name(
        ml_client=ml_client,
        scene_name=parameters.scene_name,
        version=parameters.data_version,
    )

    job_inputs = dict(
        scene_name=parameters.scene_name,
        data=Input(
            type=AssetTypes.URI_FOLDER,  # type: ignore
            path=dataset.path,
        ),
        model_type=parameters.model_type,
        model=Input(
            type=AssetTypes.CUSTOM_MODEL,  # type: ignore
            path=f"azureml:{parameters.model_name}-nerf:{parameters.model_version}",
        ),
        model_relative_path=parameters.model_relative_path,
        max_num_iterations=parameters.max_num_iterations,
    )

    job_name = (
        parameters.scene_name
        + "-"
        + parameters.model_type
        + "-"
        + datetime.now().strftime("%d-%m-%Y-%H%M%S")
    )

    my_job_outputs = {
        "custom_model_output": Output(type=AssetTypes.CUSTOM_MODEL)  # type: ignore
    }

    job = command(
        inputs=job_inputs,
        outputs=my_job_outputs,
        code=".",  # location of source code
        command=(
            "python3.10 scripts/train_script.py ${{inputs.model_type}} "
            "--load-dir ${{inputs.model}}${{inputs.model_relative_path}} "
            "--viewer.quit_on_train_completion True "
            "--experiment-name ${{inputs.scene_name}}-training-job "
            "--output-dir ${{outputs.custom_model_output}} "
            "--max-num-iterations ${{inputs.max_num_iterations}} "
            "--vis viewer nerfstudio-data "
            "--data ${{inputs.data}} "
        ),
        environment=parameters.environment + "@" + parameters.environment_version,
        compute="nerf-T4-ssh-gpu",
        experiment_name="nerfstudio-train-experiment",
        display_name=job_name,
        name=job_name,
    )

    job = ml_client.create_or_update(job)

    rebel_nerf_logger.info(
        "Once the job is finished, verify you model and register it using the script"
        " `python3 scripts/azurev2/register_model.py --scene-name "
        + parameters.scene_name
        + " --job-name "
        + job.name
        + "`"
    )


if __name__ == "__main__":
    main()
