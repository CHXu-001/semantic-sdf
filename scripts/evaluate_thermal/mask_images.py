from dataclasses import dataclass
from pathlib import Path

import tyro

from rebel_nerf.thermal_eval.drawer import Drawer


@dataclass
class Parameters:
    folder_path: Path
    """Path to the directory containing images to mask"""
    output_folder: Path
    """Path to the directory where masks will be saved"""
    start_image: Path = None
    """Path to the image to start from if you don't want to start from the beginning"""


def main():
    parameters = tyro.cli(Parameters)

    drawer = Drawer(
        parameters.folder_path,
        parameters.output_folder,
        parameters.start_image,
    )
    drawer.mask_images()


if __name__ == "__main__":
    main()
