from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig
from rebel_nerf.semantic_sdf.vol_sdf.vol_sdf import VolSDFModelConfig
from rebel_nerf.semantic_sdf.vol_sdf.vol_sdf_field import VolSDFFieldConfig

from rebel_nerf.semantic_sdf.base_models.pipeline_tracking import (
    VanillaPipelineTrackingConfig,
)
from rebel_nerf.semantic_sdf.base_models.semantic_sdf_dataparser import (
    SemanticSDFDataParserConfig,
)
from rebel_nerf.semantic_sdf.utils.schedulers import ExponentialSchedulerConfig


VolSDFTrackConfig = TrainerConfig(
    method_name="vol-sdf",
    steps_per_eval_image=5000,
    steps_per_eval_batch=5000,
    steps_per_save=2000,
    steps_per_eval_all_images=1000000,
    max_num_iterations=200000,
    mixed_precision=False,
    pipeline=VanillaPipelineTrackingConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=SemanticSDFDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
            ),
        ),
        model=VolSDFModelConfig(
            eval_num_rays_per_chunk=1024,
            sdf_field=VolSDFFieldConfig(bias=1.5),
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialSchedulerConfig(decay_rate=0.1, max_steps=300000),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": ExponentialSchedulerConfig(decay_rate=0.1, max_steps=300000),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)
