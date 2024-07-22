import argparse

from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure_ml_utils.azure_connect import get_client


def get_parameters() -> tuple[str, str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of the dataset in Azure",
        required=True,
    )

    parser.add_argument(
        "--version",
        type=str,
        help="Version of the dataset in Azure",
        required=True,
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to dataset on local machine",
        required=True,
    )

    parser.add_argument(
        "--description",
        type=str,
        help="Description of the dataset",
        required=True,
    )

    args = parser.parse_args()
    return args.dataset_name, args.version, args.dataset_path, args.description


def main() -> None:
    dataset_name, dataset_version, dataset_path, description = get_parameters()

    my_data = Data(
        path=dataset_path,
        type=AssetTypes.URI_FOLDER,
        description=description,
        name=dataset_name,
        version=dataset_version,
    )

    # Get a handle to the workspace
    ml_client = get_client()
    ml_client.data.create_or_update(my_data)


if __name__ == "__main__":
    main()
