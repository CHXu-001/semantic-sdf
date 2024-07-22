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
    # seed = 0
    # torch.manual_seed(seed)

    pipeline, config = load_model(
        Path("./tests/data/semantic_nerf_model"),
        Path("./../datasets/mini-synthetic-building2"),
    )
    model = pipeline.model

    res = 5
    positions = torch.Tensor([[i / res, i / res, 1] for i in range(res)])
    positions = positions.reshape((1, res, 3))
    print(positions.size())
    print(positions.dtype)
    print(positions)
    # positions = torch.Tensor([[0, 0, i / res] for i in range(res)])

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
    print(positions.dtype)
    print(positions)
    model.field.eval()
    model_output = model.field.mlp_base(positions)

    density_before_activation, density_embedding = torch.split(
        model_output, [1, model.field.geo_feat_dim], dim=-1
    )

    density = trunc_exp(density_before_activation.to(positions))
    print(density)
    semantics_input = density_embedding.view(-1, model.field.geo_feat_dim)
    x = model.field.mlp_semantics(semantics_input)
    semantic = model.field.field_head_semantics(x)

    semantic_labels = torch.argmax(
        torch.nn.functional.softmax(semantic, dim=-1), dim=-1
    )

    print("semantic_labels : ", semantic_labels)

    fig, ax = plt.subplots()
    # ax.pcolor(
    #     x_list.reshape((nbr_points**2,)),
    #     y_list.reshape((nbr_points**2,)),
    #     semantic_labels,
    # )
    semantic_labels = semantic_labels.reshape((nbr_points, nbr_points))
    semantic_labels[0][0] = 1
    print(semantic_labels)
    density_plot = ax.pcolormesh(
        density.detach().numpy().reshape((nbr_points, nbr_points)),
    )
    fig.colorbar(density_plot)
    plt.show()


if __name__ == "__main__":
    main()

# nbr_points = 2
# x = np.linspace(-1, 1, nbr_points)
# y = np.linspace(-1, 1, nbr_points)

# x_mesh, y_mesh = np.meshgrid(x, y)
# x_list = torch.Tensor(x_mesh.tolist()).reshape((nbr_points**2, 1))
# y_list = torch.Tensor(y_mesh.tolist()).reshape((nbr_points**2, 1))

# positions = torch.concatenate(
#     (x_list, y_list, torch.ones(nbr_points**2, 1)), dim=1
# )
# positions = positions.reshape((1, nbr_points**2, 3))