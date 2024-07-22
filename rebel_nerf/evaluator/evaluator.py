import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from PIL import Image

from rebel_nerf.render.renderer import RenderedImageModality


class Evaluator:
    """
    Evaluates a model by computing metrics on the eval data extracted from the model"""

    def __init__(
        self,
        pipeline: Pipeline,
        config: TrainerConfig,
        job_param_identifier: Optional[str] = None,
    ) -> None:
        """
        Initializes the parameters which are `output_file` to save the metrics, the
        'job_param_identifier' is an optional parameter to identify the job parameters.
        It is saved with metrics to identify job parameters in the metrics json.
        """
        self._pipeline = pipeline
        self._pipeline.datamanager.setup_eval()
        self.identifier = job_param_identifier
        self._evaluation_images: dict[RenderedImageModality, list[np.ndarray]] = {}
        self._metrics = self._compute_metrics()

        self._benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "job_param_identifier": self.identifier,
            "results": self._metrics,
        }

    def _compute_metrics(
        self,
        modalities_to_save: list[RenderedImageModality] = [RenderedImageModality.rgb],
    ) -> dict[str, float]:
        """
        Computes metrics on eval data extracted from 'self._pipeline'

        :returns: dictionary of metrics
        """
        metrics_dict_list = []
        datamanager = self._pipeline.datamanager

        if datamanager.fixed_indices_eval_dataloader is None:
            raise RuntimeError(
                "Cannot evaluate without a fixed indices eval dataloader"
            )
        for modality in modalities_to_save:
            self._evaluation_images[modality] = []

        for (
            camera_ray_bundle,
            batch,
        ) in datamanager.fixed_indices_eval_dataloader:
            outputs = self._pipeline.model.get_outputs_for_camera_ray_bundle(
                camera_ray_bundle
            )
            (
                metrics_dict,
                images_dict,
            ) = self._pipeline.model.get_image_metrics_and_images(outputs, batch)

            # Save imgs
            images_dict["rgb"] = images_dict.pop("img")
            for modality in modalities_to_save:
                self._evaluation_images[modality].append(
                    (images_dict[modality.value] * 255).byte().cpu().numpy()
                )

            metrics_dict_list.append(metrics_dict)

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            key_std, key_mean = torch.std_mean(
                torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
            )
            metrics_dict[f"{key}_mean"] = float(key_mean)
            metrics_dict[f"{key}_std"] = float(key_std)
            metrics_dict[key] = [
                metrics_dict[key] for metrics_dict in metrics_dict_list
            ]

        return metrics_dict

    def save_images(
        self, modalities: list[RenderedImageModality], output_path: Path
    ) -> None:
        """
        Saves evaluation images to `output_path`.
        """
        for modality in modalities:
            for idx, image in enumerate(self._evaluation_images[modality]):
                Image.fromarray(image).save(
                    output_path / f"{modality.value}_{idx:05d}.jpg"
                )

    def save_metrics(self, output_folder: Path) -> None:
        """
        Saves the metrics in the `output_folder`
        """
        output_file = Path(output_folder, "metrics.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(self._benchmark_info, indent=2), "utf8")

        if self.identifier is None:
            return

        psnr_folder_path = Path(output_folder, "psnr", self.identifier + ".dat")
        psnr_folder_path.mkdir(parents=True, exist_ok=True)
        psnr_folder_path.write_text(json.dumps(self._metrics["psnr"], indent=2), "utf8")

        ssim_folder_path = Path(output_folder, "ssim", self.identifier + ".dat")
        ssim_folder_path.mkdir(parents=True, exist_ok=True)
        ssim_folder_path.write_text(json.dumps(self._metrics["ssim"], indent=2), "utf8")

        lpips_folder_path = Path(output_folder, "lpips", self.identifier + ".dat")
        lpips_folder_path.mkdir(parents=True, exist_ok=True)
        lpips_folder_path.write_text(
            json.dumps(self._metrics["lpips"], indent=2), "utf8"
        )
