import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as tvF
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from rebel_nerf.image_translator.config.parameters import TranslatorParameters
from rebel_nerf.image_translator.model import networks
from rebel_nerf.image_translator.model.content_model import SemanticContentModel
from rebel_nerf.image_translator.utils import AverageMeter, psnr


class Translator(nn.Module):
    """
    Translator aims to translate images from RGB domain to thermal domain.
    Adapted from https://github.com/ownstyledu/Translate-to-Recognize-Networks
    """

    def __init__(
        self,
        cfg: TranslatorParameters,
        client: MlflowClient,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # networks
        net = networks.TranslatorUpsampleResidual()
        self._net = nn.DataParallel(net).to(self._device)

        self._content_model = SemanticContentModel(self._cfg.content_layers).to(
            self._device
        )
        # fix the model as we only use this pretrained model to compute loss
        self._content_model.requires_grad_(False)
        self._content_model.eval()

        # objective function and optimizer
        self._criterion_content = torch.nn.L1Loss()
        self._optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self._cfg.lr, betas=(0.5, 0.999)
        )
        self._schedulers = self._get_scheduler(
            self._optimizer,
            decay_start=self._cfg.niter_decay_start,
            decay_epochs=self._cfg.niter_decay,
        )

        # logger
        self._metrics_dict = {
            "train_loss": [],
            "val_loss": [],
            "val_psnr": [],
            "lr": [],
        }
        self._client = client

        self._loss_meters: defaultdict[str, AverageMeter] = defaultdict()
        log_keys = ["train_loss", "val_loss", "val_psnr"]
        for item in log_keys:
            self._loss_meters[item] = AverageMeter()

        self.save_dir = Path(self._cfg.output_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_signature = None

    def _get_input(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_modal = data["RGB"]
        target_modal = data["thermal"]
        return source_modal.to(self._device), target_modal.to(self._device)

    def train_parameters(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> None:
        best_psnr = 0

        # This check is needed to make sure that source_modal, target_modal, and
        # generation are not unbounded during image write
        if len(train_loader) == 0:
            raise RuntimeError("No data is found in train_loader")
        for epoch in range(1, self._cfg.niter_total + 1):
            self._net.train()

            for key in self._loss_meters:
                self._loss_meters[key].reset()

            for _, data in enumerate(train_loader):
                source_modal, target_modal = self._get_input(data)
                gen = self._forward(source_modal)

                loss = self._call_loss(gen, target_modal)
                self._loss_meters["train_loss"].update(
                    loss.item(), source_modal.size(0)  # type: ignore
                )

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            # logging intermediate metrics
            self._metrics_dict["lr"].append(self._optimizer.param_groups[0]["lr"])
            self._metrics_dict["train_loss"].append(self._loss_meters["train_loss"].avg)

            # Validation after each epoch of training
            mean_psnr, val_semantic_loss = self._evaluate(val_loader)
            self._metrics_dict["val_psnr"].append(mean_psnr)
            self._metrics_dict["val_loss"].append(val_semantic_loss)

            if self._cfg.save_best and epoch >= self._cfg.niter_total - 10:
                # save model
                is_best = mean_psnr > best_psnr
                best_psnr = max(mean_psnr, best_psnr)

                if is_best:
                    model_filename = "translator_best.pth"
                    self.save_checkpoint(epoch, model_filename)

            self._schedulers.step(epoch)

        # log metrics of interest on the azure ml
        current_run = mlflow.active_run()
        self._client.log_batch(
            current_run.info.run_id,  # type: ignore
            metrics=[
                Metric(
                    key="lr",
                    value=val,
                    timestamp=int(time.time() * self._cfg.niter_total),
                    step=0,
                )
                for val in self._metrics_dict["lr"]
            ],
        )

        self._client.log_batch(
            current_run.info.run_id,  # type: ignore
            metrics=[
                Metric(
                    key="mean validation PSNR",
                    value=val,
                    timestamp=int(time.time() * self._cfg.niter_total),
                    step=0,
                )
                for val in self._metrics_dict["val_psnr"]
            ],
        )

        self._client.log_batch(
            current_run.info.run_id,  # type: ignore
            metrics=[
                Metric(
                    key="mean training loss",
                    value=val,
                    timestamp=int(time.time() * self._cfg.niter_total),
                    step=0,
                )
                for val in self._metrics_dict["train_loss"]
            ],
        )

        self._client.log_batch(
            current_run.info.run_id,  # type: ignore
            metrics=[
                Metric(
                    key="mean validation loss",
                    value=val,
                    timestamp=int(time.time() * self._cfg.niter_total),
                    step=0,
                )
                for val in self._metrics_dict["val_loss"]
            ],
        )

    def _forward(self, source_modal: torch.Tensor) -> torch.Tensor:
        return self._net(input=source_modal)

    def _call_loss(
        self,
        generation: torch.Tensor,
        target_modal: torch.Tensor,
    ) -> torch.Tensor:
        loss_total = torch.zeros(1)
        loss_total = loss_total.to(self._device)

        # compute content semantics loss
        source_features = self._content_model((generation + 1) / 2)
        target_features = self._content_model((target_modal + 1) / 2)
        len_layers = len(self._cfg.content_layers)
        loss_fns = [self._criterion_content] * len_layers
        alpha = [1] * len_layers

        layer_wise_losses = [
            alpha[i] * loss_fns[i](source_feature, target_features[i])
            for i, source_feature in enumerate(source_features)
        ] * self._cfg.alpha_content

        content_loss = sum(layer_wise_losses)
        loss_total += content_loss

        # compute mse loss
        mse_loss = nn.MSELoss()(generation, target_modal)
        loss_total += mse_loss

        # total loss
        return loss_total

    def save_checkpoint(self, epoch: int, filename: str) -> None:
        net_state_dict = self._net.state_dict()
        save_state_dict = {}
        for k, v in net_state_dict.items():
            if "content_model" in k:
                continue
            save_state_dict[k] = v

        state = {
            "epoch": epoch,
            "state_dict": save_state_dict,
            "optimizer": self._optimizer.state_dict(),
        }
        filepath = Path(self.save_dir, filename)
        torch.save(state, filepath)

    def load_checkpoint(self, checkpoint: dict) -> None:
        self._net.load_state_dict(checkpoint, strict=True)

    def _get_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        decay_start: int,
        decay_epochs: int,
    ) -> lr_scheduler.LambdaLR:
        def lambda_rule(epoch):
            assert decay_epochs is not None
            lr_l = 1 - max(0, epoch - decay_start - 1) / float(decay_epochs)
            return lr_l

        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    def _evaluate(self, val_loader: DataLoader) -> tuple[float, float]:
        self._loss_meters["val_psnr"].reset()

        self._net.eval()
        with torch.no_grad():
            for _, data in enumerate(val_loader):
                source_modal, target_modal = self._get_input(data)
                gen = self._forward(source_modal)

                val_loss = self._call_loss(gen, target_modal)
                self._loss_meters["val_loss"].update(
                    val_loss.item(), source_modal.size(0)
                )

                _, mean_psnr = psnr(gen, target_modal)
                self._loss_meters["val_psnr"].update(mean_psnr, source_modal.size(0))

        return (
            self._loss_meters["val_psnr"].avg,
            self._loss_meters["val_loss"].avg,
        )

    def test(self, test_loader: DataLoader) -> None:
        test_psnr = self._predict_images(data_loader=test_loader)
        mlflow.log_metric("mean test PSNR", test_psnr)

    def log_images(self, data_loader: DataLoader) -> None:
        self._predict_images(data_loader=data_loader, log_canvas_on_azure=True)

    def _create_image(
        self,
        source_t: torch.Tensor,
        output_t: torch.Tensor,
        target_t: torch.Tensor,
        psnr_values: list[float],
        img_idx: list[int],
        log_canvas_on_azure: bool = False,
    ) -> None:
        """Create montage images for easy comparison."""
        num_img = source_t.size(0)

        for i in range(num_img):
            # bring tensors to cpu
            source = (
                torchvision.utils.make_grid(
                    source_t[i].clone().cpu().data, normalize=True
                ),
            )
            output = (
                torchvision.utils.make_grid(
                    output_t[i].clone().cpu().data, normalize=True
                ),
            )
            target = (
                torchvision.utils.make_grid(
                    target_t[i].clone().cpu().data, normalize=True
                ),
            )

            source = tvF.to_pil_image(source[0])
            output = tvF.to_pil_image(output[0])
            target = tvF.to_pil_image(target[0])

            titles = [
                "Source modality",
                "Target modality (translated): {:.2f} dB".format(psnr_values[i]),
                "Target modality (Ground truth)",
            ]

            if log_canvas_on_azure:
                # create canvas
                fig, ax = plt.subplots(1, 3, figsize=(9, 3))
                zipped = zip(titles, [source, output, target])
                for j, (title, img) in enumerate(zipped):
                    ax[j].imshow(img)
                    ax[j].set_title(title, fontsize=9)
                    ax[j].axis("off")

                mlflow.log_figure(fig, f"montage-{img_idx[i]+1}.png")
                plt.close()
                continue

            output.save(
                Path(self.save_dir, "frame_" + str(img_idx[i] + 1).zfill(5) + ".jpeg")
            )

    def _predict_images(
        self, data_loader: DataLoader, log_canvas_on_azure: bool = False
    ) -> float:
        loss_meters = defaultdict()
        loss_meters["psnr"] = AverageMeter()

        self._net.eval()
        batch_size = data_loader.batch_size
        if batch_size is None:
            raise RuntimeError("batch_size of the given dataloader is missing")
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                source_modal, target_modal = self._get_input(data)
                gen = self._forward(source_modal)

                psnr_values, mean_psnr = psnr(gen, target_modal)
                img_idx = list(range(i * batch_size, (i + 1) * batch_size))
                self._create_image(
                    source_modal,
                    gen,
                    target_modal,
                    psnr_values,
                    img_idx,
                    log_canvas_on_azure,
                )
                loss_meters["psnr"].update(mean_psnr, source_modal.size(0))
        return loss_meters["psnr"].avg
