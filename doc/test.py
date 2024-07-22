import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import tyro
from imgviz import label_colormap
from matplotlib import pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.colors import ListedColormap
from nerfstudio.utils.io import load_from_json
from torch import Tensor, nn

print(label_colormap() / 255)

# class A:
#     def __init__(self) -> None:
#         for layer in range(3):
#             lin = nn.Linear(5, 5)
#             lin = nn.utils.weight_norm(lin)
#             setattr(self, "glin" + str(layer), lin)

#     def freeze(self):
#         for layer in range(3):
#             for param in getattr(self, "glin" + str(layer)).parameters():
#                 param.requires_grad = False

#     def is_freeze(self):
#         for layer in range(3):
#             for param in getattr(self, "glin" + str(layer)).parameters():
#                 print(param.requires_grad)

#     def get_weights(self):
#         for layer in range(3):
#             x = getattr(self, "glin" + str(layer))
#             print(x.weight)


# network = A()
# network.freeze()
# network.is_freeze()
# network.get_weights()


# A = [1, 1, 0, 1, 0, 1, 0, 0.001, 0.01]

# weights_at_zero_idx = np.where(
#     np.equal(
#         np.isclose(
#             A,
#             0,
#             atol=0.005,
#         ),
#         False,
#     )
# )[0:-1]
# print(weights_at_zero_idx)

# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# normals_PIL = Image.open(
#     "./../datasets/56a0ec536c_mini_for_NeRF/segmentation_000000.png"
# )
# fig, ax = plt.subplots()
# im = ax.imshow(normals_PIL)
# plt.show()


# A = np.array([[0.1, 0, 0.1], [0, 0.1234, 0.2], [0.2, 0.1, 0], [0, 0, 0]])
# print(np.shape(A))
# B = np.array([[60.1, 60.1, 60.1], [60, 70, 780], [1.1, 10.1, 10.1], [30, 20, 10]])
# print(np.shape(B))
# A = np.round(A, 3)
# mask = A != (0, 0.123, 0.2)
# mask = np.sum(mask, axis=1).astype(bool)
# print(mask)

# masked_A = A[mask, :]
# masked_B = B[mask, :]

# print(masked_A.shape)
# print(masked_B.shape)
# print(masked_A)
# print(masked_B)

# A = torch.Tensor([1, 2, 0, 1, 3.1, 2.1])
# B = torch.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
# A = ~A.bool()
# print(B.shape)
# print(A.shape)
# B[A, :] = torch.Tensor([0, 0, 0])
# print(B)

#######################################################################################

# def get_normal_map(depth: np.ndarray) -> np.ndarray:
#     """
#     Estimate normals from a depth map. The code inspired from stack overflow issue
#     called "Surface normal calculation from depth map in python"
#     (https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python).
#     A point cloud is estimated from the depth image, from which we compute
#     the normals.
#     """

#     height, width = depth.shape
#     fx = 0.5 * width * np.tan((90 * np.pi / 180) / 2.0)
#     K = [
#         [fx, 0, width / 2],
#         [0, fx, height / 2],
#         [0, 0, 1],
#     ]

#     def normalization(data):
#         """Normalize normals"""
#         mo_chang = np.sqrt(
#             np.multiply(data[:, :, 0], data[:, :, 0])
#             + np.multiply(data[:, :, 1], data[:, :, 1])
#             + np.multiply(data[:, :, 2], data[:, :, 2])
#         )
#         mo_chang = np.dstack((mo_chang, mo_chang, mo_chang))
#         return data / mo_chang

#     x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
#     x = x.reshape([-1])
#     y = y.reshape([-1])
#     xyz = np.vstack((x, y, np.ones_like(x)))
#     points_in_3d_grid = np.dot(np.linalg.inv(K), xyz * depth.reshape([-1]))
#     points_in_3d_grid_world = points_in_3d_grid.reshape((3, height, width))
#     f = (
#         points_in_3d_grid_world[:, 2 : height - 2, 3 : width - 1]
#         - points_in_3d_grid_world[:, 2 : height - 2, 1 : width - 3]
#     )
#     print(f.shape)
#     t = (
#         points_in_3d_grid_world[:, 3 : height - 1, 2 : width - 2]
#         - points_in_3d_grid_world[:, 1 : height - 3, 2 : width - 2]
#     )
#     print(t.shape)
#     # f = (
#     #     points_in_3d_grid_world[:, 3 : height - 3, 4 : width - 2]
#     #     - points_in_3d_grid_world[:, 3 : height - 3, 1 : width - 5]
#     # )
#     # print(f.shape)
#     # t = (
#     #     points_in_3d_grid_world[:, 4 : height - 2, 3 : width - 3]
#     #     - points_in_3d_grid_world[:, 1 : height - 5, 3 : width - 3]
#     # )
#     # print(t.shape)
#     # f = (
#     #     points_in_3d_grid_world[:, 1 : height - 1, 2:width]
#     #     - points_in_3d_grid_world[:, 1 : height - 1, 1 : width - 1]
#     # )
#     # print(f.shape)
#     # t = (
#     #     points_in_3d_grid_world[:, 2:height, 1 : width - 1]
#     #     - points_in_3d_grid_world[:, 1 : height - 1, 1 : width - 1]
#     # )
#     # print(t.shape)
#     normal_map = np.cross(f, t, axisa=0, axisb=0)
#     normal_map = normalization(normal_map)

#     # Replication padding to keep initial size
#     # normals = np.zeros((height, width, 3))
#     # normals[2:-2, 2:-2] = normal_map
#     # normals[2:-2, 0] = normal_map[:, 0]
#     # normals[2:-2, 1] = normal_map[:, 0]
#     # normals[2:-2:, -1] = normal_map[:, -1]
#     # normals[2:-2:, -2] = normal_map[:, -1]
#     # normals[0, :] = normals[2, :]
#     # normals[1, :] = normals[2, :]
#     # normals[-1, :] = normals[-3, :]
#     # normals[-2, :] = normals[-3, :]

#     # normals = np.zeros((height, width, 3))
#     # normals[3:-3, 3:-3] = normal_map
#     # for i in range(3):
#     #     normals[2:-2, i] = normal_map[:, 0]
#     # for i in range(1, 4):
#     #     normals[2:-2:, -i] = normal_map[:, -1]
#     # for i in range(3):
#     #     normals[i, :] = normals[3, :]
#     # for i in range(1, 4):
#     #     normals[-i, :] = normals[-4, :]
#     normals = normal_map
#     return normals


# # normals_PIL = Image.open("./../datasets/room_0_for_NeRF_mini/normals_0.png")
# normals_PIL = Image.open("./../datasets/56a0ec536c_mini_for_NeRF/normals_002000.jpg")
# depth = cv2.imread(
#     "./../datasets/56a0ec536c_for_NeRF/depth_000000.png", flags=cv2.IMREAD_ANYDEPTH
# )
# normals_PIL = np.array(normals_PIL) / 255
# # print(normals[0:5, 0, :])
# print(normals_PIL[0:5, 0, :])


# class Formatter(object):
#     def __init__(self, im):
#         self.im = im

#     def __call__(self, x, y):
#         # z = self.im.get_array()[int(y), int(x)]
#         # return "x={}, y={}, z={}".format(x, y, z)
#         print(self.im)
#         RGB = self.im[int(y), int(x), :]
#         normal = RGB * 2 - 1
#         return "x={}, y={}, RGB={}".format(int(y), int(x), normal)


# fig, ax = plt.subplots()
# im = ax.imshow(depth, interpolation="none")

# # normals_PIL = get_normal_map(depth)
# fig, ax = plt.subplots()
# im = ax.imshow(normals_PIL, interpolation="none")
# ax.format_coord = Formatter(normals_PIL)
# plt.show()

#######################################################################3


# def load_from_json(filename: Path):
#     """Load a dictionary from a JSON filename.

#     Args:
#         filename: The filename to load from.
#     """
#     assert filename.suffix == ".json"
#     with open(filename, encoding="UTF-8") as file:
#         return json.load(file)


# panoptic_classes = load_from_json(
#     Path("./../datasets/room_0_for_NeRF/") / "segmentation.json"
# )
# print(panoptic_classes)
# classes = list(panoptic_classes.keys())
# colors = torch.tensor(list(panoptic_classes.values()), dtype=torch.float32) / 255.0
# list_index = [int(key) for key in panoptic_classes.keys()]
# number_semantic_class = int(max(list_index)) + 1
# print("number class : ", number_semantic_class)


# from imgviz import label_colormap

# label_colour_map = label_colormap()
# print(label_colour_map[76, :])


# loss = torch.nn.CrossEntropyLoss(reduction="mean")
# labels_at_zero_weight = torch.Tensor(
#     [
#         [0.0, 0.0, 1, 0.0],
#         [0.0, 0.0, 1, 0.0],
#         [-0.1, -0.1, np.nan, -0.1],
#         [0.0, 0.0, 1, 0.0],
#         [0.0, 1.0, 0, 0.0],
#     ]
# )
# A = loss(labels_at_zero_weight, torch.Tensor([1, 1, 1, 1, 1]).long())
# print(A)
# if not torch.isnan(A):
#     print("b")


# def set_colorbar(colorbar: Colorbar):
#     colorbar.ax.get_yaxis().set_ticks([])
#     for j, lab in enumerate(["Wall", "Background", "Roof", "Window"]):
#         colorbar.ax.text(
#             1.3,
#             (2 * j + 1) / 2.70,
#             lab,
#             ha="center",
#             va="center",
#             rotation=270,
#         )


# def plot_colormap(metric, x_mesh, y_mesh, nbr_points, cMap):
#     fig, ax = plt.subplots()
#     metric = metric.reshape((nbr_points, nbr_points))
#     plot_colormap = ax.pcolormesh(
#         x_mesh,
#         y_mesh,
#         metric.cpu(),
#         cmap=cMap,
#     )
#     return fig, ax, plot_colormap


# colors = ["indigo", "darkviolet", "mediumvioletred", "pink"]
# cMap = ListedColormap(colors)

# nbr_points = 7
# x = np.linspace(-0.2, 0.3, nbr_points)
# y = np.linspace(-0.2, 0.3, nbr_points)

# x_mesh, y_mesh = np.meshgrid(x, y)
# x_list = x_mesh.reshape((nbr_points**2, 1))
# y_list = y_mesh.reshape((nbr_points**2, 1))

# positions = np.concatenate([x_list, y_list], axis=1)

# Z = torch.Tensor(
#     [
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 1, 2, 1, 1, 0],
#         [0, 1, 1, 1, 1, 1, 0],
#         [0, 1, 1, 0, 1, 1, 0],
#         [0, 1, 1, 2, 1, 1, 0],
#         [0, 1, 1, 2, 2, 1, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#     ]
# )


# sdf = Z.numpy().reshape((nbr_points**2, 1)) / 100
# print(sdf)
# print(np.equal(np.isclose(sdf, 0.01, atol=0.001), False))
# zero_points = positions[
#     np.where(np.equal(np.isclose(sdf, 0.01, atol=0.001), False))[0], :
# ]
# print(zero_points)

# print(sdf)
# sdf[np.where(np.equal(np.isclose(sdf, 0.01, atol=0.001), False))[0]] = 0
# print(sdf)

# fig, ax, plot_segmentation = plot_colormap(Z, x_mesh, y_mesh, nbr_points, cMap)
# plot_segmentation.set_clim(0, 3)
# colorbar_segmentation = fig.colorbar(plot_segmentation)
# ax.scatter(zero_points[:, 0], zero_points[:, 1], c="red", s=0.5)
# set_colorbar(colorbar_segmentation)
# plt.show()

# Z = Z.reshape((7, 7)).numpy()
# nbr_points = 7
# building_labels = {1: 0, 2: 0}
# for side in range(4):
#     print("New side")
#     for i in range(nbr_points):
#         line_labels = {1: 0, 2: 0}
#         for j in range(nbr_points):
#             if side == 0:
#                 label = Z[i, j]
#             elif side == 1:
#                 label = Z[j, i]
#             elif side == 2:
#                 label = Z[i, nbr_points - j - 1]
#             elif side == 3:
#                 label = Z[nbr_points - j - 1, i]

#             if label == 0:
#                 continue
#             line_labels[label] += 1
#             if sum(line_labels.values()) == 3:
#                 building_labels[max(line_labels, key=line_labels.get)] += 1
#                 print(i, j, line_labels, max(line_labels, key=line_labels.get))
#                 break

# print(building_labels)
# print("WWR = ", building_labels[2] / building_labels[1])
