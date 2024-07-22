# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Dict, Optional, Type, Union

import torch
from jaxtyping import Float
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.sdf_field import SDFField, SDFFieldConfig
from torch import Tensor, nn


class LaplaceDensity(nn.Module):
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter(
            "beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False)
        )
        self.register_parameter(
            "beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True)
        )

    def forward(self, sdf: Tensor, beta: Union[Tensor, None] = None) -> Tensor:
        """convert sdf value to density value with beta, if beta is missing,
        then use learnable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


@dataclass
class VolSDFFieldConfig(SDFFieldConfig):
    """Vol-SDF Field Config"""

    _target: Type = field(default_factory=lambda: VolSDFField)


class VolSDFField(SDFField):
    """
    A field that learns a Signed Distance Functions (SDF), an RGB color and
    a density using the Laplace density module
    """

    config: VolSDFFieldConfig

    def __init__(
        self,
        config: VolSDFFieldConfig,
        aabb: Float[Tensor, "2 3"],  # noqa: F722
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        """
        The field is built on top of the SDF field. It needs a `config` file, the scene
        size as an axis-aligned bounding box in `aabb` and the number of images used for
        embedding appearance in `num_images`. Set `use_average_appearance_embedding` if
        necessary and specify the `spatial_distortion` if there is some.
        """
        super().__init__(
            config,
            aabb,
            num_images,
            use_average_appearance_embedding,
            spatial_distortion,
        )

        # laplace function for transform sdf to density from VolSDF
        self.laplace_density = LaplaceDensity(init_val=self.config.beta_init)

    def get_outputs(
        self,
        ray_samples: RaySamples,
        density_embedding: Optional[Tensor] = None,
        return_alphas: bool = False,
    ) -> Dict[FieldHeadNames, Tensor]:
        """
        Compute output of the field using the `ray_samples` as input.
        `density_embedding` is a useless artifact from base class. Use
        `return_aphas` if needed
        """
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        outputs = {}

        camera_indices = ray_samples.camera_indices.squeeze()

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        inputs.requires_grad_(True)
        with torch.enable_grad():
            hidden_output = self.forward_geonetwork(inputs)
            sdf, geo_feature = torch.split(
                hidden_output, [1, self.config.geo_feat_dim], dim=-1
            )
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=inputs,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        rgb = self.get_colors(
            inputs, directions_flat, gradients, geo_feature, camera_indices
        )

        density = self.laplace_density(sdf)

        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        density = density.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = torch.nn.functional.normalize(gradients, p=2, dim=-1)

        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.DENSITY: density,
                FieldHeadNames.NORMALS: normals,
                FieldHeadNames.GRADIENT: gradients,
            }
        )

        if return_alphas:
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        return outputs
