"""
This scripts creates a nerfStudio-compatible dataset from a room/office of the Replica
dataset. Note that the files in the replica room/office come from two origins. The two
sequences were generated by the authors of semantic-NeRF using habitat-sim. The other
files were downloaded using Replica github.

Dropbox with semantic-NeRF generated sequences:
https://www.dropbox.com/sh/9yu1elddll00sdl/AABWLTQhDhTQ7vCS8PZmrSmJa/Replica_Dataset?dl=0&subfolder_nav_tracking=1

Replica github to download the rest of the files:
https://github.com/facebookresearch/Replica-Dataset
"""

import dataclasses
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import tyro
from imgviz import label_colormap
from open3d import io, visualization
from PIL import Image
from plyfile import PlyData

from rebel_nerf.semantic_sdf.scripts.converter_to_nerf_dataset import DatasetConverter


@dataclasses.dataclass
class PathToDataset:
    """Stores input arguments used for creating the nerf dataset."""

    path_to_room: Path = Path("./")
    """Path to the replica datastet (one room/office)"""
    output_path: Path = Path("./../datasets/")
    """Path to the folder where to store the NeRF dataset"""


class ReplicaConverter(DatasetConverter):
    def __init__(self, paths: PathToDataset) -> None:
        super().__init__(path_to_room=paths.path_to_room, output_path=paths.output_path)
        self._get_path_to_replica_room()

        self.cameras_to_world = self.get_cameras_transform()

    def _get_path_to_replica_room(self):
        dataset_name = self.path_to_room.name + "_for_NeRF"
        self.output_path = self.output_path / dataset_name
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.path_to_room = self.path_to_room / "Sequence_1"
        if not self.path_to_room.is_dir():
            raise FileExistsError("Replica dataset doesn't contain a camera sequence.")

    def get_cameras_transform(self) -> list[np.ndarray]:
        """Reads the file that contains the transform matrixes"""
        cameras_to_world = []
        with open(self.path_to_room / "traj_w_c.txt", "r") as f:
            line = f.readline().split()
            while len(line) != 0:
                cam_to_world = np.zeros((4, 4))
                for i, number in enumerate(line):
                    cam_to_world[i // 4][i % 4] = float(number)
                cameras_to_world.append(cam_to_world)
                line = f.readline().split()

        return cameras_to_world

    def convert(self):
        # Get RGB images
        got_image_size = False
        for source_rgb in (self.path_to_room / "rgb").iterdir():
            if not got_image_size:
                got_image_size = True
                image = np.asarray(Image.open(source_rgb))
                height, width, _ = image.shape

            shutil.copyfile(source_rgb, self.output_path / source_rgb.name)

        # Get segmented images
        for source_semantic in (self.path_to_room / "semantic_class").iterdir():
            if source_semantic.name[0:3] == "vis":
                new_name = "segmentation" + source_semantic.name[13:]
                shutil.copyfile(source_semantic, self.output_path / new_name)

        # Get depth images and normals
        for source_depth in (self.path_to_room / "depth").iterdir():
            # Images are not loaded in order, so find the image number
            image_number = int(re.findall(r"\d+", source_depth.name)[0])

            # Load depth and compute normals from it
            depth = cv2.imread(str(source_depth), flags=cv2.IMREAD_ANYDEPTH)
            normals = DatasetConverter.get_normal_map(depth)

            # Normals are not coherent across views: they are predicted picture by
            # picture. So project normals in the world space using the camera
            # transform matrix
            cam_to_world = self.cameras_to_world[image_number]
            normals = normals.reshape((height * width, 3))
            normals = cam_to_world[:3, :3] @ normals.T
            normals = (normals.T).reshape((height, width, 3))
            normals = -normals

            # Scale normals so that they can be rendered as RGB
            normals = np.uint8((normals + 1) / 2 * 255)[..., ::-1]

            # Save depth and normals
            shutil.copyfile(source_depth, self.output_path / source_depth.name)
            cv2.imwrite(
                str(self.output_path / "normals_{}.png".format(image_number)),
                normals,
            )

        # Get meta_data.json with camera parameters and transform matrixes
        meta_data = self.get_meta_data(width, height)

        with open(self.output_path / "meta_data.json", "w") as outfile:
            json.dump(meta_data, outfile, indent=4)

        # Get semantic point cloud ground truth
        with open(
            self.path_to_room / "../habitat/info_semantic.json", "r"
        ) as semantic_info:
            info = json.load(semantic_info)
            id_to_label = np.array(info["id_to_label"])
            id_to_label[id_to_label <= 0] = 0

        point_cloud_rgb_and_index = PlyData.read(
            self.path_to_room / "../habitat/mesh_semantic.ply"
        )

        vertices, semantic_colour_list = self.get_semantic_point_cloud(
            point_cloud_rgb_and_index, id_to_label, label_colormap()
        )

        # Save semantic point cloud
        PlyData([vertices]).write(self.output_path / "semantic_mesh.ply")

        # Get segmentation metadata
        dict_label_colour_map = {
            str(i): list(color) for i, color in enumerate(semantic_colour_list)
        }
        with open(self.output_path / "segmentation.json", "w") as outfile:
            json.dump(dict_label_colour_map, outfile, indent=4)

        # Visualize semantic point cloud for verification
        cloud = io.read_point_cloud(str(self.output_path / "semantic_mesh.ply"))
        visualization.draw_geometries([cloud])  # type: ignore


def main() -> None:
    replica_converter = ReplicaConverter(tyro.cli(PathToDataset))

    replica_converter.convert()


if __name__ == "__main__":
    main()
