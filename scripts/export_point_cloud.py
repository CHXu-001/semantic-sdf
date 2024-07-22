"""
Export a point cloud from a NeRF model
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append(".")
sys.path.append("./nerfstudio")

from rebel_nerf.render.export_semantic_point_cloud_from_density import (  # noqa: E402
    PointcloudFromNeRFExporter,
)
from rebel_nerf.render.export_semantic_point_cloud_from_sdf import (  # noqa: E402
    PointCloudFromSDFExporter,
)


@dataclass
class Parameters:
    dataset: Path
    model_uri: Path
    model_type: str = "semantic-nerf"
    output_dir: Path = Path("./output")

    def __post_init__(self) -> None:
        mapping_name_to_exporter = {
            "nerfacto": PointcloudFromNeRFExporter,
            "neus": PointCloudFromSDFExporter,
            "neus-facto": PointCloudFromSDFExporter,
            "semantic-nerf": PointcloudFromNeRFExporter,
            "semantic-sdf": PointCloudFromSDFExporter,
            "mono-sdf": PointCloudFromSDFExporter,
        }
        self.exporter = mapping_name_to_exporter[self.model_type]

        self.extract_semantic = True
        if self.model_type == "mono-sdf":
            self.extract_semantic = False


if __name__ == "__main__":
    parameters = tyro.cli(Parameters)

    point_cloud_exporter = parameters.exporter.from_pipeline_path(
        model_path=Path(parameters.model_uri),
        dataset_path=parameters.dataset,
    )

    if not parameters.output_dir.exists():
        parameters.output_dir.mkdir(parents=True)

    pcd = point_cloud_exporter.generate_point_cloud(parameters.extract_semantic)
    torch.cuda.empty_cache()

    point_cloud_exporter.register_point_cloud(
        pcd, parameters.output_dir, parameters.extract_semantic
    )
