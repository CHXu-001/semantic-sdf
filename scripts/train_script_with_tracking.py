"""
Train a radiance field with nerfstudio.
"""
from dataclasses import dataclass
from pathlib import Path
import sys
import tyro
from typing import Optional


# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append(".")
sys.path.append("./nerfstudio")

from rebel_nerf.semantic_sdf.base_models.config_nerfacto import (  # noqa: E402
    NeRFactoTrackConfig,
)
from rebel_nerf.semantic_sdf.base_models.config_neusfacto import (  # noqa: E402
    NeuSFactoTrackConfig,
)
from rebel_nerf.semantic_sdf.base_models.config_semantic_nerf import (  # noqa: E402
    SemanticNeRFTrackConfig,
)
from rebel_nerf.semantic_sdf.base_models.config_semantic_sdf import (  # noqa: E402
    SemanticSDFTrackConfig,
)
from rebel_nerf.semantic_sdf.base_models.config_neus import (  # noqa: E402
    NeuSTrackConfig,
)
from rebel_nerf.semantic_sdf.vol_sdf.config_vol_sdf import (  # noqa: E402
    VolSDFTrackConfig,
)
from rebel_nerf.semantic_sdf.base_models.config_semantic_sdf_on_vol_sdf import (  # noqa: E402, E501
    SemanticSDFonVolSDFTrackConfig,
)
from rebel_nerf.semantic_sdf.mono_sdf.config_mono_sdf import (  # noqa: E402
    MonoSDFTrackConfig,
)
from rebel_nerf.semantic_sdf.base_models.config_semantic_sdf_on_mono_sdf import (  # noqa: E402, E501
    SemanticSDFonMonoSDFTrackConfig,
)

from nerfstudio.scripts.train import main  # noqa: E402


@dataclass
class TrainingParameters:
    model_type: str = "nerfacto"
    """What NeRF model to train. Defaults to Nerfacto"""
    experiment_name: str = "nerfacto training"
    """Name of the model to train"""
    output_dir: Path = "./outputs"
    """Where to save the model and outputs"""
    max_num_iterations: int = 50000
    data: Path = "./inputs"
    load_config: Optional[Path] = None
    """Input data in azure format"""

    def __post_init__(self) -> None:
        mapping_name_to_config = {
            "nerfacto": NeRFactoTrackConfig,
            "neus-facto": NeuSFactoTrackConfig,
            "semantic-nerf": SemanticNeRFTrackConfig,
            "semantic-sdf": SemanticSDFTrackConfig,
            "neus": NeuSTrackConfig,
            "vol-sdf": VolSDFTrackConfig,
            "semantic-sdf-on-vol-sdf": SemanticSDFonVolSDFTrackConfig,
            "mono-sdf": MonoSDFTrackConfig,
            "semantic-sdf-on-mono-sdf": SemanticSDFonMonoSDFTrackConfig,
        }
        self.model = mapping_name_to_config[self.model_type]


if __name__ == "__main__":
    parameters = tyro.cli(TrainingParameters)

    parameters.model.experiment_name = parameters.experiment_name
    parameters.model.output_dir = parameters.output_dir
    parameters.model.max_num_iterations = parameters.max_num_iterations
    parameters.model.data = parameters.data
    parameters.model.viewer.quit_on_train_completion = True
    parameters.model.load_config = parameters.load_config

    main(parameters.model)
