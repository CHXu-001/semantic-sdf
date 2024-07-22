"""
This file implements a class that integrates some tools to build a
nerfStudio-compatible dataset from any dataset.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement


class DatasetConverter(ABC):
    def __init__(
        self,
        path_to_room: Path = Path("./"),
        output_path: Path = Path("./../datasets/"),
    ) -> None:
        self.path_to_room = path_to_room
        self.output_path = output_path
        self.cameras_to_world: list[np.ndarray] = []

    def get_semantic_point_cloud(
        self,
        point_cloud_rgb_and_index: PlyData,
        id_to_label: list[int],
        label_to_colour: np.ndarray,
    ) -> tuple[PlyElement, list[tuple[int, int, int]]]:
        """
        From a point cloud `point_cloud_rgb_and_index` that associates vertices with a
        RGB color and faces (defined by 4 vertices) with an index (intance index), we
        want a point cloud that associates each vertex to a color. The algorithm is
        inspired from the Delaunay triangulation. `id_to_label` maps this instance
        index to a semantic labels and `label_to_colour` maps this label to a semantic
        RGB colour.

        :returns: vertices associated to a semantic RGB colour and the list of used
        semantic RGB colors
        """
        vertices = point_cloud_rgb_and_index.elements[0]
        faces = point_cloud_rgb_and_index.elements[1]

        vertices_output = []
        semantic_colour_list = []
        for face in faces:
            vertex_positions_tuple = vertices[face["vertex_indices"]][["x", "y", "z"]]
            vertex_positions_array = np.array(
                [list(position) for position in vertex_positions_tuple]
            )

            middle_position = np.mean(vertex_positions_array, axis=0).tolist()
            semantic_colour = label_to_colour[id_to_label[face["object_id"]]].tolist()
            middle_position.extend(semantic_colour)
            vertices_output.append(tuple(middle_position))

            if tuple(semantic_colour) not in semantic_colour_list:
                semantic_colour_list.append(tuple(semantic_colour))

        vertices_output = np.array(
            vertices_output,
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )

        return PlyElement.describe(vertices_output, "vertex"), semantic_colour_list

    @staticmethod
    def opencv_to_opengl_camera(transform: np.ndarray) -> np.ndarray:
        return transform @ trimesh.transformations.rotation_matrix(
            np.deg2rad(180), [1, 0, 0]
        )

    @abstractmethod
    def get_cameras_transform(self) -> list[np.ndarray]:
        """
        Reads the file that contains the transform matrixes

        :returns: List of cameras transform
        """
        return

    def get_meta_data(self, width: int, height: int, fov: int = 90) -> dict[str, Any]:
        """
        Creates a dict filled with informations needed by the semantic-SDF dataparser.

        Provide camera informations: `width`, `height`, Field of View (`fov`) in pixels
        """

        meta_data = {
            "camera_model": "OPEN_CV",
            "width": width,
            "height": height,
            "has_mono_prior": True,
            "has_foreground_mask": False,
            "has_sparse_sfm_points": False,
            "scene_box": {"aabb": [[-1, -1, -1], [1, 1, 1]]},
            "frames": [],
        }
        for image_number, cam_to_world in enumerate(self.cameras_to_world):
            cam_to_world = DatasetConverter.opencv_to_opengl_camera(cam_to_world)

            fx = 0.5 * width * np.tan((fov * np.pi / 180) / 2.0)
            fy = 0.5 * height * np.tan((fov * np.pi / 180) / 2.0)
            intrinsic = [
                [fx, 0, width / 2, 0],
                [0, fy, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]

            new_view = {
                "rgb_path": "rgb_{}.png".format(image_number),
                "segmentation_path": "segmentation_{}.png".format(image_number),
                "depth_path": "depth_{}.png".format(image_number),
                "normals_path": "normals_{}.png".format(image_number),
                "camtoworld": cam_to_world.tolist(),
                "intrinsics": intrinsic,
            }
            meta_data["frames"].append(new_view)

        return meta_data

    @staticmethod
    def get_normal_map(depth: np.ndarray) -> np.ndarray:
        """
        Estimate normals from a depth map. The code inspired from stack overflow issue
        called "Surface normal calculation from depth map in python"
        (https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python).
        A point cloud is estimated from the depth image, from which we compute
        the normals.
        """

        height, width = depth.shape
        fx = 0.5 * width * np.tan((90 * np.pi / 180) / 2.0)
        K = [
            [fx, 0, width / 2],
            [0, fx, height / 2],
            [0, 0, 1],
        ]

        def normalization(data):
            """Normalize normals"""
            mo_chang = np.sqrt(
                np.multiply(data[:, :, 0], data[:, :, 0])
                + np.multiply(data[:, :, 1], data[:, :, 1])
                + np.multiply(data[:, :, 2], data[:, :, 2])
            )
            mo_chang = np.dstack((mo_chang, mo_chang, mo_chang))
            return data / mo_chang

        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        x = x.reshape([-1])
        y = y.reshape([-1])
        xyz = np.vstack((x, y, np.ones_like(x)))
        points_in_3d_grid = np.dot(np.linalg.inv(K), xyz * depth.reshape([-1]))
        points_in_3d_grid_world = points_in_3d_grid.reshape((3, height, width))
        f = (
            points_in_3d_grid_world[:, 1 : height - 1, 2:width]
            - points_in_3d_grid_world[:, 1 : height - 1, 1 : width - 1]
        )
        t = (
            points_in_3d_grid_world[:, 2:height, 1 : width - 1]
            - points_in_3d_grid_world[:, 1 : height - 1, 1 : width - 1]
        )
        normal_map = np.cross(f, t, axisa=0, axisb=0)
        normal_map = normalization(normal_map)

        # Replication padding to keep initial size
        normals = np.zeros((height, width, 3))
        normals[1:-1, 1:-1] = normal_map
        normals[1:-1, 0] = normal_map[:, 0]
        normals[1:-1:, -1] = normal_map[:, -1]
        normals[0, :] = normals[1, :]
        normals[-1, :] = normals[-2, :]
        return normals

    @abstractmethod
    def convert(self) -> None:
        raise NotImplementedError("not implemented")
