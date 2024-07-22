"""
Train a radiance field with nerfstudio.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro

# This is a hack so that the script work with azure sdkv2.
# The root folder is _not_ added to the python path as it was in sdkv1
# and thus, local imports of rebel_nerf are not found.
# Remove when https://github.com/Azure/azure-sdk-for-python/issues/29724 is fixed
sys.path.append(".")
sys.path.append("./nerfstudio")

from nerfstudio.scripts.exporter import ExportTSDFMesh  # noqa: E402

from rebel_nerf.semantic_sdf.base_models.config_nerfacto import (  # noqa: E402
    NeRFactoTrackConfig,
)
from rebel_nerf.semantic_sdf.base_models.config_neus import (  # noqa: E402
    NeuSTrackConfig,
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
from rebel_nerf.semantic_sdf.base_models.config_semantic_sdf_on_vol_sdf import (  # noqa: E402, E501
    SemanticSDFonVolSDFTrackConfig,
)
from rebel_nerf.semantic_sdf.vol_sdf.config_vol_sdf import (  # noqa: E402
    VolSDFTrackConfig,
)
@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""

if __name__ == "__main__":
    parameters = tyro.cli(Exporter)
    ExportTSDFMesh(parameters.load_config, parameters.output_dir).main()