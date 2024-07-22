import sys
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
import tyro
import yaml
from matplotlib import pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.colors import ListedColormap

sys.path
sys.path.append(".")
sys.path.append("./nerfstudio")

from nerfstudio.engine.trainer import TrainerConfig  # noqa: E402
from nerfstudio.field_components.activations import trunc_exp  # noqa: E402
from nerfstudio.pipelines.base_pipeline import VanillaPipeline  # noqa: E402
from nerfstudio.utils.rich_utils import CONSOLE  # noqa: E402


@dataclass
class AccessCLIArgs:
    """Stores input arguments used for accessing a model."""

    model_uri: str
    """Path to model."""
    dataset_path: Path = None
    """Path to a dataset containing a transforms.json file generated by COLMAP.
    Used to query the dataset first pose to generate a spiral trajectory"""


def get_trainer(model_path: Path) -> TrainerConfig:
    """Get the trainer config associated to the model in `model_path`.

    :return: the trainer config that correspond to the model at `model_path`
    """
    render_config_paths = list(model_path.rglob("config.yml"))

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


def set_colorbar(colorbar: Colorbar) -> None:
    colorbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(["Wall", "Background", "Roof", "Window"]):
        colorbar.ax.text(
            1.3,
            (2 * j + 1) / 2.70,
            lab,
            ha="center",
            va="center",
            rotation=270,
        )


def plot_colormap(
    metric: torch.Tensor,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    nbr_points: int,
    cMap: ListedColormap,
):
    fig, ax = plt.subplots()
    metric = metric.reshape((nbr_points, nbr_points))
    plot_colormap = ax.pcolormesh(
        x_mesh,
        y_mesh,
        metric.cpu(),
        cmap=cMap,
    )
    return fig, plot_colormap


def main() -> None:
    tyro.extras.set_accent_color("bright_yellow")
    param = tyro.cli(AccessCLIArgs)

    pipeline, config = load_model(
        Path(param.model_uri),
        param.dataset_path,
    )
    model = pipeline.model

    nbr_points = 768
    # x_list = np.linspace(-5, 5, nbr_points)
    # y_list = np.linspace(-5, 5, nbr_points)
    # x_list = np.linspace(-4.1, -3, nbr_points)
    # y_list = np.linspace(-2.5, 4, nbr_points)
    x_list = np.linspace(-7, 7, nbr_points)
    y_list = np.linspace(-7, 7, nbr_points)

    x_mesh, y_mesh = np.meshgrid(x_list, y_list)
    x_mesh = x_mesh.reshape((nbr_points**2, 1))
    y_mesh = y_mesh.reshape((nbr_points**2, 1))

    z_list = np.linspace(0, 22, 15)
    # building_labels = {0: 0, 3: 0}  # 0: wall, 3: window
    for z in z_list:
        # Get a meshgrid in x-y plain
        positions = np.concatenate(
            [x_mesh, y_mesh, z * np.ones((nbr_points**2, 1))], axis=1
        )
        # positions = positions * 0.0569121
        positions = torch.Tensor(positions)

        # Infer density
        positions = model.field.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0

        model.field.eval()
        model_output = model.field.mlp_base(positions).reshape((1, nbr_points**2, 16))

        density_before_activation, density_embedding = torch.split(
            model_output, [1, model.field.geo_feat_dim], dim=-1
        )
        density = trunc_exp(density_before_activation.to(positions)).detach()

        # Infer the semantic label
        semantics_input = density_embedding.view(-1, model.field.geo_feat_dim)
        semantic_field_output = model.field.mlp_semantics(semantics_input).reshape(
            (1, nbr_points**2, 64)
        )
        semantic_field_output = semantic_field_output.type(torch.float32)
        semantic = model.field.field_head_semantics(semantic_field_output)

        semantic_labels = torch.argmax(
            torch.nn.functional.softmax(semantic, dim=-1), dim=-1
        )

        # Correct the segmentation according to the density
        # density = density.reshape((1, nbr_points**2))
        # semantic_labels[density < 20] = 1
        semantic_labels = (
            semantic_labels.reshape((nbr_points, nbr_points)).cpu().numpy()
        )

        #     # Detect walls and windows in the x-y plain
        #     for side in range(4):
        #         side_labels = {0: 0, 3: 0}
        #         for i in range(nbr_points):
        #             line_labels = {0: 0, 2: 0, 3: 0}
        #             for j in range(nbr_points):
        #                 if side == 0:
        #                     label = semantic_labels[i, j]
        #                 elif side == 1:
        #                     label = semantic_labels[j, i]
        #                 elif side == 2:
        #                     label = semantic_labels[i, nbr_points - j - 1]
        #                 elif side == 3:
        #                     label = semantic_labels[nbr_points - j - 1, i]

        #                 if label == 1:
        #                     continue
        #                 line_labels[label] += 1
        #                 if sum(line_labels.values()) == 10:
        #                     line_label = max(line_labels, key=line_labels.get)
        #                     if line_label == 2:
        #                         continue
        #                     building_labels[line_label] += 1
        #                     side_labels[line_label] += 1
        #                     break

        #         if side_labels[3] != 0:
        #             CONSOLE.print(side_labels)
        #             CONSOLE.print(
        #                 "WWR = ",
        #                 side_labels[3] / (side_labels[0] + side_labels[3]),
        #             )

        # # Detect roof
        # x_mesh, z_mesh = np.meshgrid(x_list, z_list)
        # x_mesh = x_mesh.reshape((nbr_points**2, 1))
        # z_mesh = z_mesh.reshape((nbr_points**2, 1))

        # nbr_plot_inf = 0
        # nbr_plot_sup = 0

        # for y in y_list:
        #     # Get a meshgrid in x-z plain
        #     positions = np.concatenate(
        #         [x_mesh, y * np.ones((nbr_points**2, 1)), z_mesh], axis=1
        #     )
        #     positions = torch.Tensor(positions)

        #     # Infer density
        #     model.field.eval()
        #     model_output = model.field.mlp_base(positions).reshape((1, nbr_points**2, 16))

        #     density_before_activation, density_embedding = torch.split(
        #         model_output, [1, model.field.geo_feat_dim], dim=-1
        #     )
        #     density = trunc_exp(density_before_activation.to(positions)).detach()

        #     # Infer the semantic label
        #     semantics_input = density_embedding.view(-1, model.field.geo_feat_dim)
        #     semantic_field_output = model.field.mlp_semantics(semantics_input).reshape(
        #         (1, nbr_points**2, 64)
        #     )
        #     semantic_field_output = semantic_field_output.type(torch.float32)
        #     semantic = model.field.field_head_semantics(semantic_field_output)

        #     semantic_labels = torch.argmax(
        #         torch.nn.functional.softmax(semantic, dim=-1), dim=-1
        #     )

        #     # Correct the segmentation according to the density
        #     density = density.reshape((1, nbr_points**2))
        #     semantic_labels[density < 50] = 1
        #     semantic_labels = (
        #         semantic_labels.reshape((nbr_points, nbr_points)).cpu().numpy()
        #     )

        #     # Find start of the wall
        #     roof_start = 0
        #     for i in range(nbr_points):
        #         line_labels = {0: 0, 2: 0, 3: 0}
        #         got_roof: bool = False
        #         for k in range(nbr_points):
        #             label = semantic_labels[nbr_points - k - 1, i]
        #             if label == 1:
        #                 continue
        #             line_labels[label] += 1

        #             if got_roof is False and line_labels[2] >= 5:
        #                 got_roof = True
        #                 line_labels = {0: 0, 2: 0, 3: 0}

        #             if got_roof and line_labels[0] >= 20:
        #                 roof_start = i
        #                 break
        #         if roof_start != 0:
        #             break

        #     # Find end of the wall
        #     roof_end = 0
        #     for i in range(nbr_points):
        #         line_labels = {0: 0, 2: 0, 3: 0}
        #         got_roof: bool = False
        #         for k in range(nbr_points):
        #             label = semantic_labels[nbr_points - k - 1, nbr_points - i - 1]
        #             if label == 1:
        #                 continue
        #             line_labels[label] += 1

        #             if got_roof is False and line_labels[2] >= 5:
        #                 got_roof = True
        #                 line_labels = {0: 0, 2: 0, 3: 0}

        #             if got_roof and line_labels[0] >= 20:
        #                 roof_end = nbr_points - i
        #                 break
        #         if roof_end != 0:
        #             break

        #     if roof_start != 0 and roof_end != 0:
        #         building_labels[0] += roof_end - roof_start
        #         CONSOLE.print("roof : ", roof_start, roof_end)

        #         if (nbr_plot_inf < 5 and roof_start < 100) or (
        #             nbr_plot_sup < 5 and roof_start > 300
        #         ):
        #             if roof_start < 100:
        #                 nbr_plot_inf += 1
        #             if roof_start > 300:
        #                 nbr_plot_sup += 1

        #             CONSOLE.print("line_labels : ", line_labels)

        #             CONSOLE.print("Data plot : ", y, roof_start, roof_end)

        #             colors = ["indigo", "darkviolet", "mediumvioletred", "pink"]
        #             cMap = ListedColormap(colors)

        #             product_fig, plot_product = plot_colormap(
        #                 torch.Tensor(semantic_labels),
        #                 x_mesh.reshape((nbr_points, nbr_points)),
        #                 z_mesh.reshape((nbr_points, nbr_points)),
        #                 nbr_points,
        #                 cMap,
        #             )
        #             plot_product.set_clim(0, 3)
        #             colorbar_product = product_fig.colorbar(plot_product)
        #             set_colorbar(colorbar_product)

        #             mlflow.log_figure(product_fig, "product_z_" + str(y) + ".png")

        # CONSOLE.print(building_labels)
        # CONSOLE.print("WWR = ", building_labels[3] / building_labels[0])
        # CONSOLE.print(
        #     "WWR = ", building_labels[3] / (building_labels[0] + building_labels[3])
        # )

        # colors = ["indigo", "darkviolet", "mediumvioletred", "pink"]
        colors = np.array(
            [[81, 92, 47], [17, 46, 86], [255, 247, 252], [215, 151, 244]]
        )
        cMap = ListedColormap(colors / 255)

        fig, plot_segmentation = plot_colormap(
            torch.Tensor(semantic_labels),
            x_mesh.reshape((nbr_points, nbr_points)),
            y_mesh.reshape((nbr_points, nbr_points)),
            nbr_points,
            cMap,
        )
        plot_segmentation.set_clim(0, 3)
        colorbar_segmentation = fig.colorbar(plot_segmentation)
        set_colorbar(colorbar_segmentation)

        density[density > 1000] = 1000
        density_fig, plot_density = plot_colormap(
            density,
            x_mesh.reshape((nbr_points, nbr_points)),
            y_mesh.reshape((nbr_points, nbr_points)),
            nbr_points,
            cMap=None,
        )
        density_fig.colorbar(plot_density)

        density = density.reshape((1, nbr_points**2))
        semantic_labels = semantic_labels.reshape((1, nbr_points**2))
        semantic_labels[density < 20] = 1
        product_fig, plot_product = plot_colormap(
            torch.Tensor(semantic_labels),
            x_mesh.reshape((nbr_points, nbr_points)),
            y_mesh.reshape((nbr_points, nbr_points)),
            nbr_points,
            cMap,
        )
        plot_product.set_clim(0, 3)
        colorbar_product = fig.colorbar(plot_product)
        set_colorbar(colorbar_product)

        mlflow.log_figure(fig, "z_" + str(z) + ".png")
        mlflow.log_figure(density_fig, "density_z_" + str(z) + ".png")
        mlflow.log_figure(product_fig, "product_z_" + str(z) + ".png")


if __name__ == "__main__":
    main()
