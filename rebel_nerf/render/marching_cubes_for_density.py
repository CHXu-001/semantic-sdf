"""
This module implements the Marching Cubes algorithm for extracting
isosurfaces from a density field
"""

from typing import Callable

import numpy as np
import torch
import trimesh
from nerfstudio.exporter.marching_cubes import evaluate_sdf
from skimage import measure


@torch.no_grad()
def generate_mesh_with_marching_cubes(
    geometry_callable_field: Callable,
    resolution: int = 512,
    bounding_box_min: tuple[float, float, float] = (-1.0, -1.0, -1.0),
    bounding_box_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
    isosurface_threshold: float = 0.0,
) -> trimesh.Trimesh:
    """
    Computes the isosurface of a density field defined by the callable `denisty` in a
    given bounding box (`bounding_box_min` and `bounding_box_max`) with a specified
    `resolution`. The density is sampled at a set of points within a regular grid, and
    the marching cubes algorithm is used to generate a mesh that approximates the
    isosurface at a specified isovalue `isosurface_threshold`.

    :returns: A mesh with vertices, faces and normals
    """
    # Check if resolution is divisible by 512
    assert (
        resolution % 512 == 0
    ), f"""resolution must be divisible by 512, got {resolution}.
       This is important because the algorithm uses a multi-resolution approach
       to evaluate the density where the mimimum resolution is 512."""

    # Initialize variables
    crop_n = 512
    N = resolution // crop_n
    grid_min = bounding_box_min
    grid_max = bounding_box_max
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    # Initialize meshes list
    meshes = []

    # Iterate over the grid
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # Calculate grid cell boundaries
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                # Create point grid
                x = np.linspace(x_min, x_max, crop_n)
                y = np.linspace(y_min, y_max, crop_n)
                z = np.linspace(z_min, z_max, crop_n)
                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(
                    np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float
                ).cuda()

                # Function to evaluate the density for a batch of points. Note that the
                # function evaluate_sdf from nerfstudio only evaluates a callable on
                # points and thus can be used with a density field
                def evaluate(points: torch.Tensor) -> torch.Tensor:
                    return evaluate_sdf(geometry_callable_field, points)

                # Construct point pyramids
                points = points.reshape(crop_n, crop_n, crop_n, 3).permute(3, 0, 1, 2)
                points = points.reshape(3, -1).permute(1, 0).contiguous()

                # Evaluate density
                points_density = evaluate(points)

                z = points_density.detach().cpu().numpy()

                # Skip if no surface found
                if np.min(z) > isosurface_threshold or np.max(z) < isosurface_threshold:
                    continue

                z = z.astype(np.float32)
                verts, faces, normals, _ = measure.marching_cubes(  # type: ignore
                    volume=z.reshape(crop_n, crop_n, crop_n),
                    level=isosurface_threshold,
                    spacing=(
                        (x_max - x_min) / (crop_n - 1),
                        (y_max - y_min) / (crop_n - 1),
                        (z_max - z_min) / (crop_n - 1),
                    ),
                )
                verts = verts + np.array([x_min, y_min, z_min])

                meshcrop = trimesh.Trimesh(verts, faces, normals)
                meshes.append(meshcrop)

    combined_mesh: trimesh.Trimesh = trimesh.util.concatenate(meshes)  # type: ignore
    return combined_mesh
