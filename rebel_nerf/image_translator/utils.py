import cv2
import torch


class AverageMeter:
    """Computes and stores the average and current value"""

    val: float
    avg: float
    sum: float
    count: int

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 3)


def psnr(output: torch.Tensor, target: torch.Tensor) -> tuple[list[float], float]:
    psnr_values = []
    batch_size = output.size(0)
    output_array = output.cpu().numpy()
    target_array = target.cpu().numpy()

    for i in range(batch_size):
        psnr_values.append(cv2.PSNR(output_array[i], target_array[i]))

    return psnr_values, sum(psnr_values) / batch_size
