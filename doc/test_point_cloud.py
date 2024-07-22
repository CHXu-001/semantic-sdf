import pathlib

import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
from matplotlib import pyplot as plt

# cloud = o3d.io.read_point_cloud(
#     "/home/antoine-laborde/Documents/Scannet_pp_dataset/data/56a0ec536c/scans/ground_truth_point_cloud.ply"
# )
# cloud = o3d.io.read_point_cloud(
#     "/home/antoine-laborde/Documents/semantic_sdf_point_cloud_3d.ply"
# )
cloud = o3d.io.read_point_cloud(
    "./../datasets/office_1_for_NeRF_mini/semantic_sdf_point_cloud.ply"
)

# print("Number of colors in post-pc :")
# colors = np.array(cloud.colors)
# colors_dict = {}
# for color in colors:
#     color = tuple(np.round(color, 3))
#     if color in colors_dict.keys():
#         colors_dict[color] += 1
#     else:
#         colors_dict[color] = 1

# print(len(list(colors_dict.keys())))
# cloud = o3d.io.read_point_cloud("./../semantic_nerf_point_cloud.ply")

cloud_ref = o3d.io.read_point_cloud(
    "./../datasets/56a0ec536c_for_NeRF_v2/semantic_mesh.ply"
)
# cloud_ref = o3d.io.read_point_cloud(
#     "./../datasets/office_0_for_NeRF_mini/semantic_mesh.ply"
# )
oriented_bounded_box = cloud_ref.get_minimal_oriented_bounding_box()
oriented_bounded_box.color = (1, 0, 0)

# obb_extent = oriented_bounded_box.extent
# oriented_bounded_box.extent = obb_extent * 1.05

obb_center = oriented_bounded_box.center
oriented_bounded_box.center = obb_center + [0, 0.5, -0.5]

cloud_ref = cloud_ref.crop(oriented_bounded_box)

print(pcu.chamfer_distance(np.asarray(cloud.points), np.asarray(cloud_ref.points)))

print(np.asarray(cloud.points).shape)

# cloud_down_2 = cloud.voxel_down_sample(voxel_size=0.03)
# print(np.asarray(cloud_down_2.points).shape)

# cloud = cloud_down_2

gt_points = np.asarray(cloud.points)
print("x : ", np.min(gt_points[:, 0]), np.max(gt_points[:, 0]))
print("y : ", np.min(gt_points[:, 1]), np.max(gt_points[:, 1]))
print("z : ", np.min(gt_points[:, 2]), np.max(gt_points[:, 2]))

# new_points = np.asarray(cloud.points) * (1 / 0.19553596)
# cloud.points = o3d.cpu.pybind.utility.Vector3dVector(new_points)

# o3d.visualization.draw_geometries([cloud])  # type: ignore
bb_size = 1 / 0.19553596 * 3
points = [
    [0, 0, 0],
    [bb_size, 0, 0],
    [0, bb_size, 0],
    [0, 0, bb_size / 2],
]
lines = [
    [0, 1],
    [0, 2],
    [0, 3],
]
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)

# vis = o3d.visualization.Visualizer()
# ctr = vis.get_view_control()
# parameters = o3d.io.read_pinhole_camera_parameters(
#     "ScreenCamera_2024-01-29-15-56-33.json"
# )

# ctr.convert_from_pinhole_camera_parameters(parameters)
o3d.visualization.draw_geometries([line_set, cloud])

# bb_size = 1 / 0.19553596
# points = [
#     [0, 0, 0],
#     [bb_size, 0, 0],
#     [0, bb_size, 0],
#     [bb_size, bb_size, 0],
#     [0, 0, bb_size],
#     [bb_size, 0, bb_size],
#     [0, bb_size, bb_size],
#     [bb_size, bb_size, bb_size],
# ]
# lines = [
#     [0, 1],
#     [0, 2],
#     [1, 3],
#     [2, 3],
#     [4, 5],
#     [4, 6],
#     [5, 7],
#     [6, 7],
#     [0, 4],
#     [1, 5],
#     [2, 6],
#     [3, 7],
# ]
# colors = [[1, 0, 0] for i in range(len(lines))]
# line_set = o3d.geometry.LineSet(
#     points=o3d.utility.Vector3dVector(points),
#     lines=o3d.utility.Vector2iVector(lines),
# )
# line_set.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([line_set, cloud])

##################################################################################################

# from plyfile import *
# import numpy as np

# path_in = "./../datasets/room_0_for_NeRF/semantic_mesh.ply"
# path_in = "./../semantic_nerf_point_cloud.ply"

# print("Reading input...")
# file_in = PlyData.read(path_in)
# vertices_in = file_in.elements[0]
# print(vertices_in.properties)
# print(vertices_in.count)
# print(vertices_in[0])
# faces_in = file_in.elements[1]
# print(faces_in.properties)
# print(faces_in.count)
# print(faces_in[0:5])

# path_out = path_in + "_0.ply"
# PlyData([vertices_in]).write(path_out)

# vertex_list = np.array([])
# for f in faces_in:
#     vertex_indexes = f[0]
#     for vertex in vertex_indexes:
#         if vertex not in vertex_list:
#             vertex_list = np.append(vertex_list, vertex)
# print("number tot indexes: ", len(list(np.unique(vertex_list))))

# print("Filtering data...")
# objects = {}
# for f in faces_in:
#     object_id = f[1]
#     if not object_id in objects:
#         objects[object_id] = []
#     objects[object_id].append((f[0],))

# print("Writing data...")
# for object_id, faces in objects.items():
#     path_out = path_in + f"_{object_id}.ply"
#     faces_out = PlyElement.describe(
#         np.array(faces, dtype=[("vertex_indices", "O")]), "face"
#     )
#     PlyData([vertices_in, faces_out]).write(path_out)

##################################################################################################

# import trimesh

# path = "./../sdf_marching_cubes_mesh.ply"

# mesh = trimesh.load(path)

# print(mesh)
# vertices = mesh.vertices.tolist()
# print(type())

# print(pathlib.Path("./../datasets/room_0_for_NeRF/semantic_mesh.ply").is_file())

# cloud = io.read_point_cloud("./../sdf_marching_cubes_mesh.ply")
# cloud = o3d.io.read_point_cloud("./../datasets/room_1_for_NeRF/semantic_mesh.ply")
# cloud = io.read_point_cloud("./../datasets/room_0_for_NeRF/semantic_mesh.ply")

gt_points = np.asarray(cloud_ref.points)
print("gt x : ", np.min(gt_points[:, 0]), np.max(gt_points[:, 0]))
print("gt y : ", np.min(gt_points[:, 1]), np.max(gt_points[:, 1]))
print("gt z : ", np.min(gt_points[:, 2]), np.max(gt_points[:, 2]))

# print(np.asarray(cloud.points))
# new_points = np.asarray(cloud.points) * (1 / 0.19553596)
# cloud.points = o3d.cpu.pybind.utility.Vector3dVector(new_points)
# print(np.asarray(cloud.points))

# o3d.visualization.draw_geometries([cloud])  # type: ignore

bb_size = 1 / 0.19553596 * 3
points = [
    [0, 0, 0],
    [bb_size, 0, 0],
    [0, bb_size, 0],
    [0, 0, bb_size / 2],
]
lines = [
    [0, 1],
    [0, 2],
    [0, 3],
]
# bb_size = 1 / 0.19553596
# points = [
#     [0, 0, 0],
#     [bb_size, 0, 0],
#     [0, bb_size, 0],
#     [bb_size, bb_size, 0],
#     [0, 0, bb_size],
#     [bb_size, 0, bb_size],
#     [0, bb_size, bb_size],
#     [bb_size, bb_size, bb_size],
# ]
# lines = [
#     [0, 1],
#     [0, 2],
#     [1, 3],
#     [2, 3],
#     [4, 5],
#     [4, 6],
#     [5, 7],
#     [6, 7],
#     [0, 4],
#     [1, 5],
#     [2, 6],
#     [3, 7],
# ]
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([line_set, cloud_ref])
