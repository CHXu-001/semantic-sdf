"""
Semantic SDF dataset.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import cv2
from nerfstudio.utils.rich_utils import CONSOLE
import numpy as np

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path


class MonoSDFDataset(InputDataset):
    """Dataset that returns images, semantics, depth and normals if available.

    It takes as intput the `dataparser_outputs` that contains a description of where
    and how to read input images (segmentation, depth and normals), and the
    `scale_factor` by which the poses where scaled down.
    """

    def __init__(
        self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0
    ):
        super().__init__(dataparser_outputs, scale_factor)

        self.include_mono_prior = self.metadata["include_mono_prior"]
        if self.include_mono_prior:
            CONSOLE.print(self.metadata["depth_filenames"])
            CONSOLE.print(self.metadata["normal_filenames"])
            self.depth_filenames = self.metadata["depth_filenames"]
            self.normal_filenames = self.metadata["normal_filenames"]

    def get_metadata(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Gets metadata such as segmentation, depth or normal that goes with a view.

        `data` encapsulates information about one view such as the image index.
        """
        metadata = {}
        # handle mono prior
        if self.include_mono_prior:
            depth_filepath = self.depth_filenames[data["image_idx"]]
            normal_filepath = self.normal_filenames[data["image_idx"]]

            # Scale depth images to meter units and also by scaling applied to cameras
            # depth_image, normal_image = self.get_depths_and_normals(
            #     depth_filepath=depth_filepath,
            #     normal_filepath=normal_filepath,
            # )

            metadata["depth"] = depth_filepath
            metadata["normal"] = normal_filepath

        return metadata

    def get_depths_and_normals(
        self, depth_filepath: Path, normal_filepath: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes additional depths and normal information.

        Provide the path to depth and normal files in `depth_filepath`
        and `normal_filepath`.
        """

        # load mono depth in meters
        depth = cv2.imread(str(depth_filepath), flags=cv2.IMREAD_ANYDEPTH)
        depth = torch.from_numpy(depth.astype(np.int32)).float()
        depth = depth / 1000  # Consversion to meters

        # load mono normals : they are scaled to RGB so we convert it back to normals
        normal = cv2.imread(str(normal_filepath)) / 255
        normal = normal * 2.0 - 1.0
        normal = torch.from_numpy(normal).float()

        return depth, normal
