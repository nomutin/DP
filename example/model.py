"""LightningModule for DP."""

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops.layers.torch import Rearrange
from lightning import LightningModule
from torch import Tensor, nn
from torchgeometry.contrib import SpatialSoftArgmax2d

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
            Rearrange("b s c h w -> (b s) c h w"),
            SpatialSoftArgmax2d(),
            Rearrange("(b s) d xy -> b (s d xy)", s=states_seq_len),
            nn.Linear(states_seq_len * 6, unet_config.condition_dim),
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
