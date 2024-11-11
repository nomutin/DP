# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab, and The HuggingFace Inc. team. All rights reserved.
"""DDPM Scheduler."""

from dataclasses import dataclass


@dataclass
class NoiseSchedulerConfig:
    """
    Configuration class for NoiseScheduler.

    Attributes
    ----------
    num_train_timesteps : int
        Number of diffusion steps for the forward diffusion schedule.
    num_inference_steps : int | None
        Number of reverse diffusion steps to use at inference time (steps are evenly spaced).
        If not provided, this defaults to be the same as `num_train_timesteps`.
    beta_schedule : str
        Name of the diffusion beta schedule as per DDPMScheduler from Hugging Face diffusers.
    beta_start : float
        Beta value for the first forward-diffusion step.
    beta_end : float
        Beta value for the last forward-diffusion step.
    clip_sample : bool
        Whether to clip the sample to [-`clip_sample_range`, +`clip_sample_range`] for each
        denoising step at inference time. WARNING: you will need to make sure your action-space is
        normalized to fit within this range.
    clip_sample_range : float
        The magnitude of the clipping range as described above.
    """

    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    clip_sample: bool = True
    clip_sample_range: float = 1.0
