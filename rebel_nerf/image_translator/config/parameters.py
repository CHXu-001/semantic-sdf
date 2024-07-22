from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranslatorParameters:
    data_root: str
    """Path to the data folder"""
    output_dir: str
    """Model will be saved at this path"""
    input_size = 224
    """Size of the image as the input to the translator model"""
    scale_size = 256
    """Being rescaled to `scale_size` before implementing other transforms"""
    manual_seed = True
    """If True, the seed will be manually set but still be random"""
    batch_size = 32
    """Size of the batch size for training"""
    num_workers = 0
    """Number of subprocesses to use for data loading"""
    eval_batch_size = 16
    """Size of the batch size for validation"""
    niter_total = 300
    lr_policy = "lambda"
    niter_decay_start = 30
    """The first epoch that decay starts"""
    niter_decay = 270
    """The number of epochs running weight decay in total"""
    lr = 2e-4
    """Learning rate"""
    content_layers = ["l0", "l1", "l2", "l3", "l4"]
    alpha_content = 1
    arch = "resnet18"
    pretrained = "imagenet"
    # Useless parameters
    save_image_path = "./checkpoints/translator/outputs"
    """Save the montage image in the process of testing"""
    save_best = True
    """If True, the model will de saved every time it performs better than a previous
    training at validation time"""
    checkpoint_dir = "checkpoints"
    """Path to save the checkpoints in `output_dir`"""
    test_ckpt_path = "translator/translator_best.pth"
    """Path to load the test model in test mode"""

    @property
    def _rgb_raw_data_dir(self) -> Path:
        return Path(self.data_root, "rgb_raw_data")

    @property
    def _thermal_raw_data_dir(self) -> Path:
        return Path(self.data_root, "thermal_raw_data")

    @property
    def rgb_train_data_dir(self) -> Path:
        return Path(self._rgb_raw_data_dir, "train")

    @property
    def thermal_train_data_dir(self) -> Path:
        return Path(self._thermal_raw_data_dir, "train")

    @property
    def rgb_val_data_dir(self) -> Path:
        return Path(self._rgb_raw_data_dir, "val")

    @property
    def thermal_val_data_dir(self) -> Path:
        return Path(self._thermal_raw_data_dir, "val")

    @property
    def rgb_test_data_dir(self) -> Path:
        return Path(self._rgb_raw_data_dir, "test")

    @property
    def thermal_test_data_dir(self) -> Path:
        return Path(self._thermal_raw_data_dir, "test")

    def as_dict(self):
        """Returns both CLI and default values."""
        arguments = {}
        for k, _ in self.__class__.__dict__.items():
            if not k.startswith("__"):
                arguments[k] = getattr(self, k)
        return arguments
