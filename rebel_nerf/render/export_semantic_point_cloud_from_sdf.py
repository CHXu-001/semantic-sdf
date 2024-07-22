from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import open3d as o3d
import torch
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.exporter.marching_cubes import (
    generate_mesh_with_multires_marching_cubes,
)
from nerfstudio.fields.sdf_field import SDFField
from nerfstudio.models.base_surface_model import SurfaceModel
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.io import load_from_json

from rebel_nerf.render.renderer import Renderer


@dataclass
class PointCloudFromSDFExporter(Renderer):
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
        """Creates a renderer loading the model saved at `model_path`.

        `dataset_path` is a useless artifact from loading the pipeline instead of
        the model; it needs to point toward the dataset.
        `eval_num_rays_per_chunk` represents the number of rays to render per forward
        pass and a default value should exist in the loaded config file. Only change
        from `None` if the PC's memory can't handle rendering the default chunck / batch
        value per one forward pass.

        :returns: object of class PointcloudFromSDFExporter
        """
        pipeline, config = Renderer.extract_pipeline(
            model_path=model_path,
            transforms_path=dataset_path,
            eval_num_rays_per_chunk=eval_num_rays_per_chunk,
        )

        # Get class-color mapping for segmentation
        panoptic_classes = load_from_json(dataset_path / "segmentation.json")

        return cls(pipeline, config, panoptic_classes)

    def _infer_semantic_label(
        self, model: SurfaceModel, positions: torch.Tensor
    ) -> torch.Tensor:
        """Infers the labels at `positions` using a `model`"""

        model.field.eval()

        positions = positions.to(torch.device("cuda:0"))

        # Extract geometric features
        hidden_output = model.field.forward_geonetwork(positions)

        sdf, geo_feature = torch.split(
            hidden_output, [1, model.field.config.geo_feat_dim], dim=-1
        )

        sdf = sdf.detach().cpu().numpy()

        # Get the labels
        semantic_input = geo_feature.view(-1, model.field.config.geo_feat_dim)

        semantic = model.field.mlp_semantic(semantic_input)
        semantic = model.field.field_head_semantic(semantic)

        semantic_labels = torch.argmax(
            torch.nn.functional.softmax(semantic, dim=-1), dim=-1
        )

        return semantic_labels

    def generate_point_cloud(self, extract_semantic: bool = True) -> None:
        """
        Generates a semantic point cloud.

        Set `extract_semantic` to false to have a segmentation-free point cloud.
        """
        self.resolution = 2048
        assert (
            self.resolution % 512 == 0
        ), f"""resolution must be divisible by 512, got {self.resolution}.
        This is important because the algorithm uses a multi-resolution approach
        to evaluate the SDF where the minimum resolution is 512."""

        self.bounding_box_min = (-5, -5, -5)
        self.bounding_box_max = (20, 20, 20)

        # Extract mesh using marching cubes for sdf at a multi-scale resolution.
        multi_res_mesh = generate_mesh_with_multires_marching_cubes(
            geometry_callable_field=lambda x: cast(SDFField, self.pipeline.model.field)
            .forward_geonetwork(x)[:, 0]
            .contiguous(),
            resolution=self.resolution,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            isosurface_threshold=0,
            coarse_mask=None,
        )

        vertices = torch.Tensor(multi_res_mesh.vertices.tolist())

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices.double().cpu().numpy())

        if extract_semantic is False:
            return pcd

        semantic_labels = (
            self._infer_semantic_label(model=self.pipeline.model, positions=vertices)
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
