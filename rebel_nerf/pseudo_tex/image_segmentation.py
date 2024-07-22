from typing import Optional, Union

import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import ColorMode, Visualizer

from Mask2Former.mask2former import add_maskformer2_config


class ImageSegmentation(Visualizer):
    """
    A class that performs image segmentation using Mask2Former.
    It outputs coco-format semantic segmentation without the text labels.
    """

    COCO_METADATA = MetadataCatalog.get("coco_2017_val_panoptic")
    _OFF_WHITE = (1.0, 1.0, 240.0 / 255)
    _BLACK = (0, 0, 0)
    _RED = (1.0, 0, 0)

    def __init__(self, rgb_image) -> None:
        super().__init__(
            rgb_image[:, :, ::-1],
            self.COCO_METADATA,
            scale=1.2,
            instance_mode=ColorMode.IMAGE_BW,
        )

        self.rgb_image = rgb_image
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(
            "Mask2Former/configs/coco/panoptic-segmentation/swin/"
            "maskformer2_swin_large_IN21k_384_bs16_100ep.yaml"
        )

        cfg.MODEL.WEIGHTS = (
            "https://dl.fbaipublicfiles.com/"
            "maskformer/mask2former/coco/panoptic/"
            "maskformer2_swin_large_IN21k_384_bs16_100ep"
            "/model_final_f07440.pkl"
        )
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True

        self._predictor = DefaultPredictor(cfg)

    def get_segmentation(self) -> np.ndarray:
        """
        Sementically segment rgb_image in a one-shot manner.

        :return: semantic segmentation of the input image
        """

        outputs = self._predictor(self.rgb_image)

        semantic_image = self.draw_sem_seg(
            outputs["sem_seg"].argmax(0).to("cpu")
        ).get_image()

        return semantic_image

    def draw_sem_seg(
        self,
        sem_seg: Union[torch.Tensor, np.ndarray],
        area_threshold: Optional[int] = None,
        alpha: float = 0.8,
    ) -> object:
        """
        Draw semantic segmentation predictions of `sem_seg` given
        `alpha` value that controls transparency of the segmentation.

        If given the `area_threshold`, the segments with less than
        `area_threshold` are not drawn.

        :returns: output image object with visualizations.
        """
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        for label in filter(
            lambda label_: label_ < len(self.metadata.stuff_classes), labels
        ):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (sem_seg == label).astype(np.uint8)

            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=self._OFF_WHITE,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output
