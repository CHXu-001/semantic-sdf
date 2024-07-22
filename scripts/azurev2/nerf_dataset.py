from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data


def get_dataset_from_scene_name(
    ml_client: MLClient, scene_name: str, version: str = "1"
) -> Data:
    dataset_name = scene_name + "-images-to-nerf-dataset"
    return ml_client.data.get(name=dataset_name, version=version)


def get_eval_dataset_from_scene_name(
    ml_client: MLClient, scene_name: str, version: str = "1"
) -> Data:
    dataset_name = scene_name + "-eval"
    return ml_client.data.get(name=dataset_name, version=version)
