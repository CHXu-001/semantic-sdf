import cv2
import numpy as np
from PIL import Image, ImageOps


class PseudoTeX:
    """
    Class for creating pseudo texture image from thermal, texture and semantic images
    """

    def __init__(
        self,
        thermal_img: Image.Image,
        semantic_img: np.ndarray,
        width: int = 800,
        height: int = 600,
        alpha: float = 0.5,
    ) -> None:
        """
        Initialize the class with `thermal_img` and `semantic images`
        and creates texture image from the `thermal_img`.
        Set the `width`, `height` and `alpha` values for the output.
        """
        self._semantic_image = semantic_img

        self._thermal_image = thermal_img.convert("L")

        self._texture_image = ImageOps.equalize(self._thermal_image)
        self._texture_image = cv2.cvtColor(
            np.array(self._texture_image), cv2.COLOR_RGB2BGR
        )

        self._thermal_image = cv2.cvtColor(np.array(thermal_img), cv2.COLOR_RGB2BGR)

        self.width = width
        self.height = height
        self.alpha = alpha

        self.pseudo_tex = self._pseudo_tex()

    def _pseudo_tex(
        self,
    ) -> np.ndarray:
        """
        Creates pseudo texture images from thermal images, and overlays them
        with semantic images.
        The output image has 'width' and 'height' dimensions as well as 'alpha'
        transparency.

        :return: pseudo_tex texture image
        """

        self._semantic_image = cv2.resize(
            self._semantic_image, (self.width, self.height)
        )
        self._thermal_image = cv2.resize(self._thermal_image, (self.width, self.height))
        self._texture_image = cv2.resize(self._texture_image, (self.width, self.height))

        pseudo_tex = self._semantic_image.copy()

        cv2.addWeighted(
            self._thermal_image, self.alpha, pseudo_tex, 1 - self.alpha, 0, pseudo_tex
        )
        cv2.addWeighted(
            self._texture_image, self.alpha, pseudo_tex, 1 - self.alpha, 0, pseudo_tex
        )

        return pseudo_tex
