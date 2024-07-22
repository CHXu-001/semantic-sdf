from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tyro
from azure.ai.ml import Input, MLClient, command
from azure.ai.ml.constants import AssetTypes
from azure_ml_utils.azure_connect import get_client

from scripts.azurev2.nerf_dataset import get_dataset_from_scene_name


@dataclass
class EvalParameters:
    scene_name: str
    """Name of the scene/object to be assigned to the registered output of the \
       script on Azure"""
    data_version: str
    """The version of dataset corresponding to `scene_name`"""
    model_version: str
    """The version of registered model"""
    output_folder: Path = Path("./outputs")
    rendered_output_dir: Path = Path("./outputs")
    environment: str = "rebel-nerf-backbone"
    environment_version: str = "@latest"
    debug: bool = False

    def get_inputs(self, ml_client: MLClient) -> tuple[dict[str, Any], str]:
        job_inputs = {
            "model": Input(
                type=AssetTypes.CUSTOM_MODEL,  # type: ignore
                path=Path(f"azureml:{self.scene_name}-nerf:{self.model_version}"),
            ),
            "scene_name": self.scene_name,
        }
        command = "python3.10 "

        if self.debug:
            command += "-m debugpy --listen localhost:5678 --wait-for-client "

        command += "scripts/eval_script.py " "--model-uri ${{inputs.model}} "

        dataset_path = get_dataset_from_scene_name(
            ml_client=ml_client, scene_name=self.scene_name, version=self.data_version
        ).path
        assert dataset_path is not None
        job_inputs["dataset_path"] = Input(
            type=AssetTypes.URI_FOLDER, path=dataset_path  # type: ignore
        )

        command += "--dataset_path ${{inputs.dataset_path}} "
        command += f"--output_folder {self.output_folder} "

        return job_inputs, command


def main() -> None:
    parameters = tyro.cli(EvalParameters)

    ml_client = get_client()

    job_inputs, cmd = parameters.get_inputs(ml_client)

    job_name = parameters.scene_name + "-" + datetime.now().strftime("%d-%m-%Y-%H%M%S")

    job = command(
        code=".",
        inputs=job_inputs,
        environment=parameters.environment + parameters.environment_version,
        compute="rebel-nerf-t4-ssh",
        command=cmd,
        experiment_name="nerfstudio-eval-experiment",
        display_name=job_name,
        name=job_name,
    )

    returned_job = ml_client.jobs.create_or_update(job)

    assert returned_job.services is not None
    returned_job.services["Studio"].endpoint


if __name__ == "__main__":
    main()
