from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.models.neus import NeuSModelConfig

from rebel_nerf.semantic_sdf.base_models.pipeline_tracking import (
    VanillaPipelineTrackingConfig,
)
from rebel_nerf.semantic_sdf.base_models.semantic_sdf_dataparser import (
    SemanticSDFDataParserConfig,
)
from rebel_nerf.semantic_sdf.base_models.semantic_sdf_dataset import SemanticSDFDataset

NeuSTrackConfig = TrainerConfig(
    method_name="neus",
    steps_per_eval_image=500,
    steps_per_eval_batch=5000,
    steps_per_save=20000,
    steps_per_eval_all_images=1000000,
    max_num_iterations=100000,
    mixed_precision=False,
    pipeline=VanillaPipelineTrackingConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[SemanticSDFDataset],
            dataparser=SemanticSDFDataParserConfig(),
            train_num_rays_per_batch=1024,
            eval_num_rays_per_batch=1024,
            camera_optimizer=CameraOptimizerConfig(
                mode="off",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
            ),
        ),
        model=NeuSModelConfig(
            sdf_field=SDFFieldConfig(
                bias=1.5,
                beta_init=0.8,
                inside_outside=True,
            ),
            background_model="none",
            eval_num_rays_per_chunk=1024,
        ),
    ),
    optimizers={
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000
            ),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=5000, learning_rate_alpha=0.05, max_steps=300000
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)
