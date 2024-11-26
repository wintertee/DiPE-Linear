import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LongTermForecasting(L.LightningModule):

    def __init__(self,
                 model: nn.Module,
                 log_forecast: bool = False,
                 profile=False):
        super().__init__()
        self.model = model
        self.example_input_array = self.model.example_input_array
        self.log_forecast = log_forecast
        self.profile = profile
        self.profile_time = []

        self.save_hyperparameters()

        self.metrics_fn = {
            'mse': F.mse_loss,
            'mae': F.l1_loss,
        }

    def forward(self, x) -> Tensor:
        return self.model(x)

    def shared_step(self, x, y):
        y_hat = self.forward(x)

        with torch.no_grad():
            metrics = {
                metric_name: metric_fn(y_hat, y)
                for metric_name, metric_fn in self.metrics_fn.items()
            }

        return metrics, y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        metrics, y_hat = self.shared_step(x, y)
        loss = self.model.loss(y, y_hat)

        self.log_dict(
            {
                f'train_{metric_name}': metric_value
                for metric_name, metric_value in metrics.items()
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log_add_forecast('train', batch_idx, batch[0], y_hat, batch[1])

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        metrics, y_hat = self.shared_step(x, y)
        loss = self.model.loss(y, y_hat)
        self.log_dict(
            {
                f'val_{metric_name}': metric_value
                for metric_name, metric_value in metrics.items()
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log_add_forecast('val', batch_idx, batch[0], y_hat, batch[1])

    def test_step(self, batch, batch_idx):
        x, y = batch

        if self.profile:
            torch.cuda.empty_cache()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            metrics, _ = self.shared_step(x, y)
            end_event.record()
            torch.cuda.synchronize()
            inference_time = start_event.elapsed_time(end_event)
            self.profile_time.append(inference_time)
            print(f"Inference time: {np.sum(self.profile_time)} ms")
        else:
            metrics, _ = self.shared_step(x, y)
        self.log_dict(
            {
                f'test_{metric_name}': metric_value
                for metric_name, metric_value in metrics.items()
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def log_add_forecast(self, name, batch_idx, x, y_hat, y):

        if (batch_idx % self.trainer.log_every_n_steps == 0 and
                self.log_forecast and
                isinstance(self.logger.experiment,
                           torch.utils.tensorboard.writer.SummaryWriter)):

            tensorboard = self.logger.experiment

            x_np = x[0].detach().cpu().numpy()
            y_np = y[0].detach().cpu().numpy()
            y_hat_np = y_hat[0].detach().cpu().numpy()

            num_plots = min(10, x.shape[2])
            rows = (num_plots + 2) // 3
            cols = 3

            fig, axs = plt.subplots(rows, cols, figsize=(12, 2 * rows))

            if rows == 1:
                axs = np.expand_dims(axs, axis=0)

            for i in range(num_plots):
                row = i // cols
                col = i % cols
                ax = axs[row, col]

                ax.plot(range(self.model.input_len),
                        x_np[:, i],
                        label='Input Sequence',
                        color='black')
                ax.plot(range(self.model.input_len,
                              self.model.input_len + self.model.output_len),
                        y_np[:, i],
                        color='green')
                ax.plot(range(self.model.input_len,
                              self.model.input_len + self.model.output_len),
                        y_hat_np[:, i],
                        label='Prediction',
                        color='blue',
                        linestyle='dashed',
                        alpha=0.5)

                # ax.set_title(f'Forecast for Feature {i}')
                # ax.set_xlabel('Time Steps')
                # ax.set_ylabel('Value')

            for j in range(num_plots, rows * cols):
                fig.delaxes(axs[j // cols, j % cols])

            plt.subplots_adjust(left=0.05,
                                right=0.95,
                                top=0.95,
                                bottom=0.05,
                                wspace=0.3,
                                hspace=0.3)

            tensorboard.add_figure(name, fig, self.global_step)
