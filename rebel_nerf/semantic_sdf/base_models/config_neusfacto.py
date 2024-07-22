from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    CosineDecaySchedulerConfig,
    MultiStepSchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.fields.sdf_field import SDFFieldConfig
from nerfstudio.models.neus_facto import NeuSFactoModelConfig

from rebel_nerf.semantic_sdf.base_models.pipeline_tracking import (
    VanillaPipelineTrackingConfig,
)
from rebel_nerf.semantic_sdf.base_models.semantic_sdf_dataparser import (
    SemanticSDFDataParserConfig,
)
from rebel_nerf.semantic_sdf.base_models.semantic_sdf_dataset import SemanticSDFDataset

NeuSFactoTrackConfig = TrainerConfig(
    method_name="neus-facto",
    steps_per_eval_image=5000,
    steps_per_eval_batch=5000,
    steps_per_save=2000,
    steps_per_eval_all_images=1000000,
    max_num_iterations=20001,
    mixed_precision=False,
    pipeline=VanillaPipelineTrackingConfig(
        datamanager=VanillaDataManagerConfig(
            _target=VanillaDataManager[SemanticSDFDataset],
            dataparser=SemanticSDFDataParserConfig(),
            train_num_rays_per_batch=2048,
            eval_num_rays_per_batch=2048,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
            ),
        ),
        model=NeuSFactoModelConfig(
            # proposal network allows for significantly smaller sdf/color network
            sdf_field=SDFFieldConfig(
                use_grid_feature=True,
                num_layers=2,
                num_layers_color=2,
                hidden_dim=256,
                inside_outside=False,
                bias=0.5,
                beta_init=0.8,
                use_appearance_embedding=False,
            ),
            background_model="none",
            eval_num_rays_per_chunk=2048,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": MultiStepSchedulerConfig(
                max_steps=20001, milestones=(10000, 1500, 18000)
            ),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001
            ),
        },
        "field_background": {
            "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-15),
            "scheduler": CosineDecaySchedulerConfig(
                warm_up_end=500, learning_rate_alpha=0.05, max_steps=20001
            ),
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
)
