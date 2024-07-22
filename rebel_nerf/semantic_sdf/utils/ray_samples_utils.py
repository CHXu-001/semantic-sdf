from typing import Tuple
from nerfstudio.cameras.rays import RaySamples
import torch

from torch import Tensor


def get_weights_and_transmittance(
    ray_samples: RaySamples, densities: Tensor
) -> Tuple[Tensor, Tensor]:
    """Compute weights and transmittance based on predicted `densities`
    for each ray sample in `ray_samples`.

    :returns: weights and transmittance for each sample
    """

    delta_density = ray_samples.deltas * densities
    alphas = 1 - torch.exp(-delta_density)

    transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
    transmittance = torch.cat(
        [
            torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device),
            transmittance,
        ],
        dim=-2,
    )
    transmittance = torch.exp(-transmittance)  # [..., "num_samples"]

    weights = alphas * transmittance  # [..., "num_samples"]

    return weights, transmittance
