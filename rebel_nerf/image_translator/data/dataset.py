import os
from pathlib import Path
from typing import Callable

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class TransDataset(Dataset):
    def __init__(
        self,
        input_size: int,
        scale_size: int,
        rgb_data_dir: str,
        thermal_data_dir: str,
        transform: Callable[[torch.Tensor], torch.Tensor],
        prior_transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
            [transforms.ToTensor()]
        ),
    ) -> None:
        super().__init__()
        self._input_size = input_size
        self._scale_size = scale_size
        self.transform = transform
        self.prior_transform = prior_transform
        self.rgb_data_dir = rgb_data_dir
        self.thermal_data_dir = thermal_data_dir

        self.rgb_imgs = self._get_images(self.rgb_data_dir)
        self.thermal_imgs = self._get_images(self.thermal_data_dir)
        # paired images have the same image name.
        self.rgb_imgs.sort()
        self.thermal_imgs.sort()

        if len(self.rgb_imgs) != len(self.thermal_imgs):
            raise ValueError("The number of RGB and thermal images is not equal.")

    def __len__(self) -> int:
        return len(self.rgb_imgs)

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        rgb_img_path = self.rgb_imgs[index]
        thermal_img_path = self.thermal_imgs[index]

        rgb_img_pil = Image.open(rgb_img_path).convert("RGB")
        # stop PIL from autorotating the loaded image
        # https://github.com/python-pillow/Pillow/issues/4703
        rgb_img_pil = ImageOps.exif_transpose(rgb_img_pil)
        thermal_img_pil = Image.open(thermal_img_path).convert("RGB")
        thermal_img_pil = ImageOps.exif_transpose(thermal_img_pil)

        wr, hr = rgb_img_pil.size
        wt, ht = thermal_img_pil.size
        if (wr != wt) or (hr != ht):
            raise ValueError(
                f"The size of RGB image ({rgb_img_path}, wr={wr}, hr={hr}) and"
                "thermal image ({thermal_img_path}, wt={wt}, ht={ht}) is not matched."
            )
        if wr > self._input_size:
            rgb_img_pil = rgb_img_pil.resize(
                (self._scale_size, self._scale_size), Image.BICUBIC
            )
            thermal_img_pil = thermal_img_pil.resize(
                (self._scale_size, self._scale_size), Image.BICUBIC
            )

        # PIL.Image -> torch.Tensor
        rgb_img_tensor = self.prior_transform(rgb_img_pil)
        thermal_img_tensor = self.prior_transform(thermal_img_pil)

        if self.transform:
            concat_img = torch.cat((rgb_img_tensor, thermal_img_tensor), dim=0)
            output_image = self.transform(concat_img)
            rgb_img_tensor = output_image[0:3]
            thermal_img_tensor = output_image[3:6]

        return {"RGB": rgb_img_tensor, "thermal": thermal_img_tensor}

    def _get_images(
        self, img_dir: str, extensions: list = [".jpg", ".png", ".jpeg", ".JPG"]
    ) -> list[str]:
        images = []
        img_dir = os.path.expanduser(img_dir)
        image_names = [d for d in os.listdir(img_dir)]
        for image_name in image_names:
            if Path(image_name).suffix in extensions:
                file = os.path.join(img_dir, image_name)
                images.append(file)
        return images
