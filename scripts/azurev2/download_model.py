import argparse

from azure_ml_utils.azure_connect import get_client
from azure_ml_utils.download_artifacts import download_model


def get_parameters() -> tuple[str, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        help="Name of the dataset in Azure",
        required=True,
    )

    parser.add_argument(
        "--version",
        type=int,
        help="Version of the dataset in Azure",
        required=True,
    )

    args = parser.parse_args()
    return args.model_name, args.version


def main() -> None:
    model_name, model_version = get_parameters()
    ml_client = get_client()

    download_model(
        ml_client=ml_client,
        model_name=model_name,
        model_version=model_version,
    )


if __name__ == "__main__":
    main()
