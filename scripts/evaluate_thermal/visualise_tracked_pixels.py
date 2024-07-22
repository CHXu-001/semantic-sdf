from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro


@dataclass
class Parameters:
    tensor_path: Path
    """Path to the .pt tensor containing pixel information"""
    input_directory: Path
    """Path to the directory containing unmarked images"""
    output_directory: Path
    """Path to the directory where marked images will be saved"""


def mark_pixels(image_path: Path, pixel: torch.Tensor) -> np.ndarray:
    """
    Mark a `pixel` on an image extracted from
    `image_path` with a purple circle.

    :returns: The image with the marked pixel
    """
    image = cv2.imread(str(image_path))

    cv2.circle(
        image, (int(pixel[0]), int(pixel[1])), 3, (255, 0, 255), -1
    )  # purple colour

    return image


def main():
    parameters = tyro.cli(Parameters)

    tensor = torch.load(parameters.tensor_path, map_location=torch.device("cpu"))
    tensor = tensor.squeeze()

    for i, (pixel_coords, image_filename) in enumerate(
        zip(tensor, parameters.input_directory.iterdir())
    ):
        output_path = Path(parameters.output_directory, f"marked_image_{i}.png")

        marked_img = mark_pixels(image_filename, pixel_coords)

        cv2.imwrite(str(output_path), marked_img)


if __name__ == "__main__":
    main()
