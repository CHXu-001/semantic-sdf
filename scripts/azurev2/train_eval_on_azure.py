import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
    version: str
    model_type: str = "uncertainty-nerf"
    """What NeRF model to train. Defaults to Nerfacto"""
    environment: str = "rebel-nerf-backbone"
    """Environment"""
    environment_version: str = "latest"
    experiment_name: str = "nerfstudio-train-eval-experiment"
    """Experiment name in azure"""
    max_num_iterations: int = 30000
    """Number of iterations to train for"""
    metrics_output_folder: str = "./outputs/"
    """Name of the output folder to save metrics"""
    use_uncertainty_loss: bool = False
    """flag to use uncertainty loss"""
    job_param_identifier: str = "None"
    """identifier saved in metrics json to identify the job param"""
    blurred_data_path: Path = Path("./outputs/data/")
    """Path to save the blurred images"""
    percentage_blur_images: float = 0.0
    """Percentage of images to blur"""
    seed: int = 0
    """Seed for the random number generator"""


def main() -> None:
    logging.basicConfig()
    rebel_nerf_logger = logging.getLogger("rebel_nerf")
    rebel_nerf_logger.setLevel(logging.INFO)

    parameters = tyro.cli(TrainingParameters)

    ml_client = get_client()
    dataset = get_dataset_from_scene_name(
        ml_client=ml_client,
        scene_name=parameters.scene_name,
        version=parameters.version,
    )

    job_inputs = dict(
        scene_name=parameters.scene_name,
        data=Input(
            type=AssetTypes.URI_FOLDER,  # type: ignore
            path=dataset.path,
        ),
        max_num_iterations=parameters.max_num_iterations,
    )

    job_name = (
        parameters.scene_name
        + "-"
        + parameters.model_type
        + "-"
        + datetime.now().strftime("%d-%m-%Y-%H%M%S")
    )

    blurred_data_asset_path = "azureml://datastores/workspaceblobstore/paths/"

    my_job_outputs = {
        "custom_model_output": Output(type=AssetTypes.CUSTOM_MODEL),  # type: ignore
        "output_dataset": Output(
            type=AssetTypes.URI_FOLDER, path=blurred_data_asset_path
        ),  # type: ignore
    }

    cmd = (
        "python3.10 rebel_nerf/uncertainty_nerf/scripts/train_eval_script.py "
        "--experiment-name ${{inputs.scene_name}}-training-job "
        "--model_output_folder ${{outputs.custom_model_output}} "
        "--max-num-iterations ${{inputs.max_num_iterations}} "
        "--data ${{inputs.data}} "
        f"--metrics_output_folder {parameters.metrics_output_folder} "
        f"--job_param_identifier {parameters.job_param_identifier} "
        f"--blurred_data_path {parameters.blurred_data_path} "
        f"--percentage_blur_images {parameters.percentage_blur_images} "
        f"--seed {parameters.seed} "
        "--data_asset_path ${{outputs.output_dataset}} "
    )

    if parameters.use_uncertainty_loss:
        cmd += "--use_uncertainty_loss"
    job = command(
        inputs=job_inputs,
        outputs=my_job_outputs,
        code=".",  # location of source code
        command=cmd,
        environment=parameters.environment + "@" + parameters.environment_version,
        compute="nerf-T4-ssh-gpu",
        experiment_name=parameters.experiment_name,
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
