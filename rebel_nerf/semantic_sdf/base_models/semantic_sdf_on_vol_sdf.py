"""
Implementation of Semantic-SDF. This model is built on top of NeusFacto and adds
a 3D semantic segmentation model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Type, cast

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.dataparsers.base_dataparser import Semantics
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import monosdf_normal_loss
from nerfstudio.model_components.renderers import SemanticRenderer
from nerfstudio.utils.rich_utils import CONSOLE

from rebel_nerf.semantic_sdf.vol_sdf.vol_sdf import VolSDFModel, VolSDFModelConfig


@dataclass
class SemanticSDFModelConfig(VolSDFModelConfig):
    """Semantic-SDF Model Config"""

    _target: Type = field(default_factory=lambda: SemanticSDFModel)
    semantic_loss_weight: float = 1.0
    """Factor that multiplies the semantic loss"""
    semantic_3D_loss_weight: float = 0.01
    """Factor that multiplies the semantic 3D loss"""


class SemanticSDFModel(VolSDFModel):
    """SemanticSDFModel extends NeuSFactoModel to add semantic segmentation in 3D."""

    config: SemanticSDFModelConfig

    def __init__(
        self, config: SemanticSDFModelConfig, metadata: Dict, **kwargs
    ) -> None:
        """
        To setup the model, provide a model `config` and the `metadata` from the
        outputs of the dataparser.
        """
        super().__init__(config=config, **kwargs)

        assert "semantics" in metadata.keys() and isinstance(
            metadata["semantics"], Semantics
        )
        self.colormap = metadata["semantics"].colors.clone().detach().to(self.device)

        self.color_mapping = {
            tuple(np.round(np.array(color), 3)): index
            for index, color in enumerate(metadata["semantics"].colors.tolist())
        }
        self.step = 0

    def populate_modules(self) -> None:
        """Instantiate modules and fields, including proposal networks."""
        super().populate_modules()

        # Fields
        self.field = self.config.sdf_field.setup(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            use_average_appearance_embedding=(
                self.config.use_average_appearance_embedding
            ),
            spatial_distortion=self.scene_contraction,
        )

        self.renderer_semantics = SemanticRenderer()
        self.semantic_loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Pass the `ray_bundle` through the model's field and renderer to get
        the model's output."""
        outputs = super().get_outputs(ray_bundle)

        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        field_outputs: Dict[FieldHeadNames, torch.Tensor] = cast(
            Dict[FieldHeadNames, torch.Tensor],
            samples_and_field_outputs["field_outputs"],
        )

        outputs["3D_semantics"] = field_outputs[FieldHeadNames.SEMANTICS]
        outputs["semantics"] = self.renderer_semantics(
            field_outputs[FieldHeadNames.SEMANTICS], weights=outputs["weights"]
        )

        # semantics colormaps
        semantic_labels = torch.argmax(
            torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1
        )
        outputs["semantics_colormap"] = self.colormap.to(self.device)[semantic_labels]

        return outputs

    def get_loss_dict(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        metrics_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute the loss dictionary from the `outputs` of the model, the `batch`
        that contains the ground truth data and the `metrics_dict`."""
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        # Map each color to its class
        GT_segmentation = []
        for color in list(batch["semantics"][..., 0][:, 0:3]):
            try:
                GT_segmentation.append(
                    self.color_mapping[tuple(np.round(np.array(color) / 255, 3))]
                )
            except KeyError:
                GT_segmentation.append(self.color_mapping[(0, 0, 0)])
                CONSOLE.print(
                    "Error in segmentation."
                    + "Wrong color automatically assigned to background"
                )
        GT_segmentation = torch.Tensor(GT_segmentation)

        # Semantic loss
        loss_dict["semantics_loss"] = (
            self.semantic_loss(
                outputs["semantics"], GT_segmentation.long().to(self.device)
            )
            * self.config.semantic_loss_weight
        )

        # Semantic 3d loss
        weights_at_zero_idx = np.where(
            np.equal(
                np.isclose(
                    outputs["field_outputs"][FieldHeadNames.SDF].detach().cpu(),
                    0,
                    atol=0.005,
                ),
                False,
            )
        )[0:-1]
        labels_at_zero_weight = outputs["3D_semantics"][weights_at_zero_idx]

        GT_background = torch.Tensor(
            [
                self.color_mapping[(0, 0, 0)]
                for _ in range(labels_at_zero_weight.shape[0])
            ]
        )
        loss_dict["semantics_3D_loss"] = (
            self.semantic_loss(
                labels_at_zero_weight, GT_background.long().to(self.device)
            )
            * self.config.semantic_3D_loss_weight
        )

        # monocular depth loss
        if "depth" in list(batch.keys()):
            depth_gt = batch["depth"].to(self.device)[..., None]
            depth_pred = outputs["depth"]

            mask = ~depth_gt.bool()
            depth_pred[mask] = 0
            loss_dict["depth_loss"] = (
                self.depth_loss(
                    depth_pred,
                    depth_gt,
                )
                * self.config.mono_depth_loss_mult
            )
        # monocular normal loss
        if "normal" in list(batch.keys()):
            normal_gt = batch["normal"].to(self.device)
            normal_pred = outputs["normal"]
            normal_pred[mask.squeeze()] = torch.Tensor([0, 0, 0]).to(
                torch.device("cuda:0")
            )
            loss_dict["normal_loss"] = (
                monosdf_normal_loss(normal_pred, normal_gt)
                * self.config.mono_normal_loss_mult
            )

        # self.step += 1
        # if self.step >= 35000:
        #     # Semantic 3d loss
        #     weights_at_zero_idx = np.where(
        #         np.equal(
        #             np.isclose(
        #                 outputs["field_outputs"][FieldHeadNames.SDF].detach().cpu(),
        #                 0,
        #                 atol=0.005,
        #             ),
        #             False,
        #         )
        #     )[0:-1]
        #     labels_at_zero_weight = outputs["3D_semantics"][weights_at_zero_idx]

        #     GT_background = torch.Tensor(
        #         [
        #             self.color_mapping[(0, 0, 0)]
        #             for _ in range(labels_at_zero_weight.shape[0])
        #         ]
        #     )
        #     loss_dict["semantics_3D_loss"] = (
        #         self.semantic_loss(
        #             labels_at_zero_weight, GT_background.long().to(self.device)
        #         )
        #         * self.config.semantic_3D_loss_weight
        #     )
        # CONSOLE.print("Step:", self.step)
        # CONSOLE.print("Semantic_VOLSDF: Losses:")
        # for loss in loss_dict.keys():
        #     CONSOLE.print(loss, ":", loss_dict[loss])
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, Any], batch: Dict[str, Any]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Compute image metrics and images from the `outputs` of the model and
        the `batch` which contains input and ground truth data."""
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        # semantics
        semantic_labels = torch.argmax(
            torch.nn.functional.softmax(outputs["semantics"], dim=-1), dim=-1
        )
        images_dict["semantics_colormap"] = self.colormap.to(self.device)[
            semantic_labels
        ]

        return metrics_dict, images_dict
