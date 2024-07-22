import sys
from pathlib import Path
from matplotlib import pyplot as plt
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
import numpy as np

import torch
import yaml
from nerfstudio.engine.trainer import TrainerConfig

sys.path
sys.path.append(".")
sys.path.append("./nerfstudio")


def get_trainer(model_path: Path) -> TrainerConfig:
    """Get the trainer config associated to the model in `model_path`.

    `eval_num_rays_per_chunk` represents the number of rays to render per forward
    pass and a default value should exist in the loaded config file. Only change
    from `None` if the PC's memory can't handle rendering the default chunck / batch
    value per one forward pass.

    :raises RuntimeError: if more than one config can be found (recursively) in the
    path `model_path`
    :return: the trainer config that correspond to the model at `model_path`
    """
    render_config_paths = list(model_path.rglob("config.yml"))
    if len(render_config_paths) > 1:
        raise RuntimeError(
            "Try to load a model from a path where multiple models "
            "can be (recursively) found. Limit the path to a single "
            "model."
        )

    if len(render_config_paths) == 0:
        raise RuntimeError("No model found at path", model_path)

    with open(render_config_paths[0], "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    return config


def load_model(
    model_path: Path,
    data_path: Path,
) -> tuple[VanillaPipeline, TrainerConfig]:
    config = get_trainer(model_path)

    # Get model state
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = list(model_path.rglob("*.ckpt"))[-1]
    loaded_state = torch.load(checkpoint_path, map_location=device)

    config.pipeline.datamanager.data = data_path

    # create pipeline from the config file content
    pipeline = config.pipeline.setup(device=device, test_mode="inference")
    pipeline.eval()
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])

    return pipeline, config


def main():
    pipeline, config = load_model(
        Path("./tests/data/semantic_sdf_model"),
        Path("./../datasets/mini-synthetic-building2"),
    )
    model = pipeline.model

    nbr_points = 5
    x = np.linspace(-1, 1, nbr_points)
    y = np.linspace(-1, 1, nbr_points)

    x_mesh, y_mesh = np.meshgrid(x, y)
    x_list = x_mesh.reshape((nbr_points**2, 1))
    y_list = y_mesh.reshape((nbr_points**2, 1))

    positions = np.concatenate([x_list, y_list, np.ones((nbr_points**2, 1))], axis=1)
    positions = torch.Tensor(positions)

    positions = positions.reshape((1, nbr_points**2, 3))
    print(positions.size())
    print(positions.type())

    model.field.eval()

    hidden_output = model.field.forward_geonetwork(positions)
    print("hidden_output : ", hidden_output.type())
    print(hidden_output.size())
    print("model.field.config.geo_feat_dim : ", model.field.config.geo_feat_dim)
    sdf, geo_feature = torch.split(
        hidden_output, [1, model.field.config.geo_feat_dim], dim=-1
    )
    print("geo_feature : ", geo_feature.type())
    print(geo_feature.size())


if __name__ == "__main__":
    main()
