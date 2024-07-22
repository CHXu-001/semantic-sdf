import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from rebel_nerf.thermal_eval.thermal_evaluator import ThermalEvaluator


@dataclass
class Parameters:
    masks_paths: Path
    """Path to the directory containing masks of tracked pixels"""
    input_directory_base: Path
    """Path to the directory containing baseline rendered images"""
    input_directory_ours: Path
    """Path to the directory containing our rendered images"""
    path_to_json_file: Path
    """Path to the json file containing temperature bounds"""
    results_path: Path
    """Path to the directory where plots will be saved"""
    mask_available: bool
    """Boolean to indicate if the input in the 'pixel_dir' contains
    masks of tracked pixels. Assign True if the input contains masks
    and False if it contains tensors."""


def main():
    parameters = tyro.cli(Parameters)

    with open(parameters.path_to_json_file, "r") as f:
        json_file = json.load(f)
        absolute_max_temperature = json_file["absolute_max_temperature"]
        absolute_min_temperature = json_file["absolute_min_temperature"]

    evaluator = ThermalEvaluator(
        results_path=parameters.results_path,
        pixel_dir=parameters.masks_paths,
        input_directory_base=parameters.input_directory_base,
        input_directory_ours=parameters.input_directory_ours,
        absolute_max_temperature=absolute_max_temperature,
        absolute_min_temperature=absolute_min_temperature,
        mask_available=parameters.mask_available,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
