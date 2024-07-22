import argparse

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure_ml_utils.azure_connect import get_client


def get_parameters() -> tuple[str, str]:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene-name",
        type=str,
        help="Name of the scene/object to be assigned to the registered output of the \
       script on Azure",
        required=True,
    )
    parser.add_argument(
        "--job-name",
        type=str,
        help="Job name that created the model",
        required=True,
    )

    args = parser.parse_args()
    return args.scene_name, args.job_name


def main() -> None:
    scene_name, job_name = get_parameters()

    model = Model(
        path=f"azureml://jobs/{job_name}/outputs/custom_model_output",
        name=scene_name + "-nerf",
        description="Model created from run " + job_name,
        type=AssetTypes.CUSTOM_MODEL,
    )

    ml_client = get_client()
    ml_client.models.create_or_update(model)


if __name__ == "__main__":
    main()
