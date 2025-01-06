"""LightningModule for DP."""

from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops.layers.torch import Rearrange
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from matplotlib import pyplot as plt
from torch import Tensor, nn
from tqdm import tqdm
from wandb import Image

from dp.core import DiffusionPolicy
from dp.ddpm import NoiseSchedulerConfig
from dp.unet import ConditionalUnet1d, ConditionalUnet1dConfig


class LitDP(LightningModule):
    """LightningModule wrapper for DiffusionPolicy."""

    def __init__(
        self,
        states_seq_len: int,
        action_generate_len: int,
        action_seq_len: int,
        unet_config: ConditionalUnet1dConfig,
        noise_scheduler_config: NoiseSchedulerConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.states_seq_len = states_seq_len
        self.action_generate_len = action_generate_len
        self.action_seq_len = action_seq_len
        self.unet_config = unet_config
        self.noise_scheduler_config = noise_scheduler_config
        unet = ConditionalUnet1d(unet_config)
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config.num_train_timesteps,
            beta_start=noise_scheduler_config.beta_start,
            beta_end=noise_scheduler_config.beta_end,
            beta_schedule=noise_scheduler_config.beta_schedule,
            clip_sample=noise_scheduler_config.clip_sample,
        )
        state_to_condition = nn.Sequential(
            Rearrange("b s c h w -> b (s c h w)"),
            nn.LazyLinear(unet_config.condition_dim),
        )
        self.model = DiffusionPolicy(
            states_seq_len=states_seq_len,
            action_generate_len=action_generate_len,
            action_seq_len=action_seq_len,
            num_train_timesteps=noise_scheduler_config.num_train_timesteps,
            num_inference_steps=noise_scheduler_config.num_inference_steps,
            action_dim=unet_config.action_dim,
            state_to_condition=state_to_condition,
            add_condition_to_action=unet,
            noise_scheduler=noise_scheduler,
        )

    def training_step(self, batch: tuple[Tensor, Tensor], *_args: int) -> Tensor:
        """Run training step."""
        observations, actions = batch
        observations = observations[:, : self.states_seq_len]
        actions = actions[:, : self.action_generate_len]
        loss = self.model.forward(actions=actions, states=observations)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], *_args: int) -> Tensor:
        """Run validation step."""
        observations, actions = batch
        observations = observations[:, : self.states_seq_len]
        actions = actions[:, : self.action_generate_len]
        loss = self.model.forward(actions=actions, states=observations)
        self.log("val_loss", loss)
        return loss


class LogDPPrediction(Callback):
    """
    Callback to log DP predictions.

    Parameters
    ----------
    every_n_epochs : int
        The interval to log the predictions.
    num_samples : int
        The number of samples to log.
    """

    def __init__(self, every_n_epochs: int, num_samples: int) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.num_samples = num_samples

    def _get_test_batch(self, trainer: Trainer, pl_module: LightningModule) -> tuple[Tensor, ...]:
        dataloader = trainer.datamodule.test_dataloader()  # type: ignore[attr-defined]
        return tuple(b[: self.num_samples].to(pl_module.device).to(pl_module.dtype) for b in next(iter(dataloader)))

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """
        Log predictions to WandbLogger.

        Parameters
        ----------
        trainer : Trainer
            Trainer which has WandbLogger and test_dataloader().
        pl_module : LightningModule
            LightningModule(DP).

        Raises
        ------
        TypeError
            If trainer.logger is not WandbLogger.
        """
        if not isinstance(logger := trainer.logger, WandbLogger):
            msg = "LogWorldModelGenerations requires WandbLogger."
            raise TypeError(msg)
        if trainer.current_epoch % self.every_n_epochs != 0 or trainer.current_epoch <= 1:
            return
        if not isinstance(pl_module, LitDP):
            msg = f"LogDPPrediction requires LitDP, got {type(pl_module)}."
            raise TypeError(msg)

        observations, actions = self._get_test_batch(trainer, pl_module=pl_module)
        chunked_observations: tuple[Tensor, ...] = observations.split(pl_module.action_seq_len, dim=1)
        prediction_list = [actions[:, : pl_module.states_seq_len]]
        for observation_seq in tqdm(chunked_observations):
            prediction = pl_module.model.generate_actions(
                states=observation_seq[:, : pl_module.states_seq_len],
            )
            prediction_list.append(prediction)
        predictions = torch.cat(prediction_list, dim=1)

        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "predictions.png"
            plot_action_prediction(target=actions, prediction=predictions, save_path=file_path)
            logger.experiment.log({"predictions": Image(str(file_path))})


def plot_action_prediction(target: Tensor, prediction: Tensor, save_path: Path) -> None:
    """
    Visualize 3D action sequence and save it as a PNG file.

    Parameters
    ----------
    target : Float[Tensor, "B T1 D"]
        Target action sequence.
    prediction : Float[Tensor, "B T2 D"]
        Predicted action sequence.
    title : str
        Title of the figure.
    """
    target = target.detach().cpu()
    prediction = prediction.detach().cpu()
    batch_size, _, dim = target.shape
    fig, ax = plt.subplots(nrows=dim, ncols=batch_size, figsize=(batch_size * 4, dim * 2))
    for b, d in product(range(batch_size), range(dim)):
        ax[d][b].plot(target[b, :, d], label="target")
        ax[d][b].plot(prediction[b, :, d], label="prediction")
        if b == 0 and d == 0:
            ax[d][b].legend()
        ax[d][b].set_title(f"Batch #{b}" if d == 0 else None)
        ax[d][b].set_ylabel(f"Dim #{d}" if b == 0 else None)
    fig.savefig(save_path)
    plt.close(fig)
    plt.clf()
