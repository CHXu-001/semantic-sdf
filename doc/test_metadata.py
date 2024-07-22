import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.camera as pc
import trimesh
from pytransform3d.transform_manager import TransformManager


def load_from_json(filename: pathlib.Path):
    """Load a dictionary from a JSON filename."""
    with open(str(filename), encoding="UTF-8") as file:
        return json.load(file)


def main() -> None:
    full_path = "./../datasets/56a0ec536c_for_NeRF/meta_data.json"
    # full_path = "./../datasets/room_1_for_NeRF/meta_data.json"
    # full_path = "./../datasets/mini-synthetic-building2/meta_data.json"
    number_cameras_to_show = 8
    x_y_lim = 4
    z_lim = 2
    # full_path = "./../datasets/mini-synthetic-building2/meta_data.json"
    # number_cameras_to_show = 200
    # x_y_lim = 20
    # z_lim = 20

    meta = load_from_json(full_path)
    poses = []

    frames = meta["frames"]
    for i, frame in enumerate(frames):
        if i < number_cameras_to_show and i % 2 == 0:
            camToWorld = np.array(frame["camtoworld"])
            print(camToWorld)
            camToWorld[1, :] = -camToWorld[1, :]
            camToWorld[2, :] = -camToWorld[2, :]
            camToWorld = camToWorld @ trimesh.transformations.rotation_matrix(
                np.deg2rad(180), [1, 0, 0]
            )
            print("rot :")
            print(
                trimesh.transformations.rotation_matrix(
                    np.deg2rad(90), [0, 0, 1], [0, 0, 0]
                )
            )
            camToWorld = np.matmul(
                trimesh.transformations.rotation_matrix(
                    np.deg2rad(90), [0, 0, 1], [0, 0, 0]
                ),
                camToWorld,
            )
            print(camToWorld)
            poses.append(
                camToWorld
                # @ np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            )
        else:
            if i > number_cameras_to_show:
                break

    # Setup the transform manager
    tm = TransformManager(strict_check=False)
    for i, pose in enumerate(poses):
        tm.add_transform(str(i), "poses", pose)

    # default parameters of a camera in Blender
    sensor_size = np.array([meta["width"], meta["height"]])
    intrinsic_matrix = np.array(
        [[320, 0, sensor_size[0] / 2.0], [0, 240, sensor_size[1] / 2.0], [0, 0, 1]]
    )
    virtual_image_distance = 0.1

    # Plot the transform's axis and the camera
    plt.figure(figsize=(30, 15))
    ax = tm.plot_frames_in("poses", s=0.1)
    for i, pose in enumerate(poses):
        pc.plot_camera(
            ax,
            cam2world=pose,
            M=intrinsic_matrix,
            sensor_size=sensor_size,
            virtual_image_distance=virtual_image_distance,
            strict_check=False,
        )
    ax.set_xlim((0, x_y_lim))
    ax.set_ylim((0, x_y_lim))
    ax.set_zlim((-z_lim, z_lim))
    plt.show()


if __name__ == "__main__":
    main()
