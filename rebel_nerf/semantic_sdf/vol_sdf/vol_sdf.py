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

"""
Implementation of VolSDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Type

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import monosdf_normal_loss
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig

from rebel_nerf.semantic_sdf.utils.error_bounded_sampler import ErrorBoundedSampler
from rebel_nerf.semantic_sdf.utils.ray_samples_utils import (
    get_weights_and_transmittance,
)


@dataclass
class VolSDFModelConfig(SurfaceModelConfig):
    """VolSDF Model Config"""

    _target: Type = field(default_factory=lambda: VolSDFModel)
    num_samples: int = 64
    """Number of samples after error bounded sampling"""
    num_samples_eval: int = 128
    """Number of samples per iteration used in error bounded sampling"""
    num_samples_extra: int = 32
    """Number of uniformly sampled points for training"""
    mono_normal_loss_mult: float = 0.1
    """Monocular normal consistency loss multiplier."""
    mono_depth_loss_mult: float = 0.05
    """Monocular depth consistency loss multiplier."""


class VolSDFModel(SurfaceModel):
    """VolSDF model

    Provide a VolSDF configuration to instantiate model.
    """

    config: VolSDFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.sampler = ErrorBoundedSampler(
            num_samples=self.config.num_samples,
            num_samples_eval=self.config.num_samples_eval,
            num_samples_extra=self.config.num_samples_extra,
        )

    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        ray_samples, eik_points = self.sampler(
            ray_bundle, density_fn=self.field.laplace_density, sdf_fn=self.field.get_sdf
        )
        field_outputs = self.field(ray_samples)
        weights, transmittance = get_weights_and_transmittance(
            ray_samples, field_outputs[FieldHeadNames.DENSITY]
        )
        bg_transmittance = transmittance[:, -1, :]

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "eik_points": eik_points,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
        }
        return samples_and_field_outputs

    def get_loss_dict(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        metrics_dict: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Compute the loss dictionary from the `outputs` of the model, the `batch`
        that contains the ground truth data and the `metrics_dict`."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # # monocular depth loss
        # depth_gt = batch["depth"].to(self.device)[..., None]
        # depth_pred = outputs["depth"]

        # mask = ~depth_gt.bool()
        # depth_pred[mask] = 0
        # loss_dict["depth_loss"] = (
        #     self.depth_loss(
        #         depth_pred,
        #         depth_gt,
        #     )
        #     * self.config.mono_depth_loss_mult
        # )

        # # monocular normal loss
        # normal_gt = batch["normal"].to(self.device)
        # normal_pred = outputs["normal"]
        # normal_pred[mask.squeeze()] = torch.Tensor([0, 0, 0]).to(torch.device("cuda:0"))
        # loss_dict["normal_loss"] = (
        #     monosdf_normal_loss(normal_pred, normal_gt)
        #     * self.config.mono_normal_loss_mult
        # )

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            # training statics
            metrics_dict["beta"] = self.field.laplace_density.get_beta().item()
            metrics_dict["alpha"] = 1.0 / self.field.laplace_density.get_beta().item()

        return metrics_dict
