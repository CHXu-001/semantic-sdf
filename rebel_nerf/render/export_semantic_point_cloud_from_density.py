from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import open3d as o3d
import torch
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.io import load_from_json

from rebel_nerf.render.marching_cubes_for_density import (
    generate_mesh_with_marching_cubes,
)
from rebel_nerf.render.renderer import Renderer


@dataclass
class PointcloudFromNeRFExporter(Renderer):
    def __init__(
        self,
        pipeline: Pipeline,
        config: TrainerConfig,
        color_mapping: dict[str, list[int, int, int]],
    ) -> None:
        self.pipeline = pipeline
        self.config = config
        self.color_mapping = color_mapping

    @classmethod
    def from_pipeline_path(
        cls,
        model_path: Path,
        dataset_path: Path,
        eval_num_rays_per_chunk: Optional[int] = None,
    ) -> "Renderer":
        """Creates a point cloud exporter loading the model saved at `model_path`.

        `dataset_path` is a useless artifact from loading the pipeline instead of
        the model; it needs to point toward the dataset.
        `eval_num_rays_per_chunk` represents the number of rays to render per forward
        pass and a default value should exist in the loaded config file. Only change
        from `None` if the PC's memory can't handle rendering the default chunck / batch
        value per one forward pass.

        :returns: object of class PointcloudFromNeRFExporter
        """
        pipeline, config = Renderer.extract_pipeline(
            model_path=model_path,
            transforms_path=dataset_path,
            eval_num_rays_per_chunk=eval_num_rays_per_chunk,
        )

        # Get class-color mapping for segmentation
        panoptic_classes = load_from_json(dataset_path / "segmentation.json")

        return cls(pipeline, config, panoptic_classes)

    @staticmethod
    def _infer_density(
        positions: torch.Tensor,
        model: Model,
    ) -> torch.Tensor:
        """Infers the density at `positions` using a `model`"""

        positions = model.field.spatial_distortion(positions)
        positions = (positions.cpu() + 2.0) / 4.0

        model.field.eval()
        model_output = model.field.mlp_base(positions)

        density_before_activation, _ = torch.split(
            model_output, [1, model.field.geo_feat_dim], dim=-1
        )
        density = trunc_exp(density_before_activation.to(positions)).detach()

        # Activation function form Semantic-NeRF
        density[density < 0] = 0
        density = 1.0 - torch.exp(-density * (20 / 512))

        return density.squeeze()

    @staticmethod
    def _infer_semantic_label(positions: torch.Tensor, model: Model) -> torch.Tensor:
        """Infers the labels at `positions` using a `model`"""
        model.field.eval()

        # Apply spatial distortion
        positions = positions.to(torch.device("cuda:0"))
        positions = model.field.spatial_distortion(positions)
        positions = (positions.cpu() + 2.0) / 4.0

        # Get geometric features
        model_output = model.field.mlp_base(positions)

        _, density_embedding = torch.split(
            model_output, [1, model.field.geo_feat_dim], dim=-1
        )

        # Get the labels
        semantics_input = density_embedding.view(-1, model.field.geo_feat_dim)
        semantic_field_output = model.field.mlp_semantics(semantics_input)

        semantic_field_output = semantic_field_output.type(torch.float32)
        semantic = model.field.field_head_semantics(semantic_field_output)

        semantic_labels = torch.argmax(
            torch.nn.functional.softmax(semantic, dim=-1), dim=-1
        )

        return semantic_labels

    def generate_point_cloud(
        self,
        extract_semantic: bool = True,
    ) -> None:
        """
        Generates a semantic point cloud.

        Set `extract_semantic` to false to have a segmentation-free point cloud.
        """
        self.resolution = 512
        assert (
            self.resolution % 512 == 0
        ), f"""resolution must be divisible by 512, got {self.resolution}.
        This is important because the algorithm uses a multi-resolution approach
        to evaluate the SDF where the minimum resolution is 512."""

        self.bounding_box_min = (-5, -5, -5)
        self.bounding_box_max = (20, 20, 20)

        # Extract mesh using marching cubes for density.
        multi_res_mesh = generate_mesh_with_marching_cubes(
            geometry_callable_field=lambda x: PointcloudFromNeRFExporter._infer_density(
                x, self.pipeline.model
            ).contiguous(),
            resolution=self.resolution,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            isosurface_threshold=0.9,
        )

        vertices = torch.Tensor(multi_res_mesh.vertices.tolist())

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices.double().cpu().numpy())

        if extract_semantic is False:
            return pcd

        semantic_labels = (
            PointcloudFromNeRFExporter._infer_semantic_label(
                positions=vertices, model=self.pipeline.model
            )
            .cpu()
            .numpy()
        )
        semantic_colour = torch.Tensor(
            [self.color_mapping[str(label)] for label in semantic_labels]
        )

        pcd.colors = o3d.utility.Vector3dVector(semantic_colour.double().cpu().numpy())

        return pcd

    def register_point_cloud(
        self,
        point_cloud: o3d.geometry.PointCloud,
        output_dir: Path,
        extract_semantic: bool = True,
    ) -> None:
        """
        Registers `point_cloud` in `output_dir`.

        Set `extract_semantic` to false if the point cloud was generated
        without the labels.
        """

        tpcd = o3d.t.geometry.PointCloud.from_legacy(point_cloud)

        if extract_semantic is True:
            # The legacy PLY writer converts colors to UInt8,
            # let us do the same to save space.
            tpcd.point.colors = (tpcd.point.colors).to(o3d.core.Dtype.UInt8)

        o3d.t.io.write_point_cloud(
            str(output_dir) + "/semantic_sdf_point_cloud.ply", tpcd
        )
