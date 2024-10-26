# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab, and The HuggingFace Inc. team. All rights reserved.
"""
Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion".

Modified from https://github.com/huggingface/lerobot/blob/main/lerobot/common/policies/diffusion/modeling_diffusion.py
"""

from collections import deque

import torch
import torch.nn.functional as tf
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import IntTensor, Tensor, nn

__all__ = ["DiffusionPolicy"]


class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion".

    References
    ----------
    [1] https://arxiv.org/abs/2303.04137
    [2] https://github.com/real-stanford/diffusion_policy

    Parameters
    ----------
    states_seq_len : int
        Number of environment steps worth of observations to pass to the policy
        (takes the current step and additional steps going back).
    action_dim: int
        The dimension of the action space.
    action_seq_len: int
        The number of action steps to run in the environment for one invocation of the policy.
        kept for execution, starting from the current step.
    action_generate_len: int
        The number of action steps to generate in the environment for one invocation of the policy.
        The diffusion model generates this steps worth of actions.
    num_train_timesteps: int
        Number of diffusion steps for the forward diffusion schedule.
    num_inference_steps: int
        Number of reverse diffusion steps to use at inference time (steps are evenly spaced).
    state_to_condition: nn.Module
        The module that converts the environment state into a condition tensor.
        [B, *] -> [B, condition_dim]
    noise_scheduler: DDPMScheduler
        The noise scheduler for the diffusion model.
    unet: nn.Module
        Network for conditioning.

    (legend: o = state_seq_len, h = action_generate_len, a = action_seq_len)
    ----------------------------------------------------------------------------------------------
    |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
    |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
    |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
    |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
    ----------------------------------------------------------------------------------------------
    `action_seq_len <= action_generate_len - state_seq_len + 1`.
    """

    def __init__(
        self,
        *,
        states_seq_len: int,
        action_dim: int,
        action_seq_len: int,
        action_generate_len: int,
        num_train_timesteps: int,
        num_inference_steps: int,
        state_to_condition: nn.Module,
        noise_scheduler: DDPMScheduler,
        unet: nn.Module,
    ) -> None:
        super().__init__()
        self.states_seq_len = states_seq_len
        self.action_dim = action_dim
        self.action_seq_len = action_seq_len
        self.action_generate_len = action_generate_len

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.state_to_condition = state_to_condition
        self.unet = unet
        self.noise_scheduler = noise_scheduler

        self.action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)
        self.state_queue: deque[Tensor] = deque(maxlen=self.config.n_obs_steps)

    def reset(self) -> None:
        """Clear state and action queues."""
        self.action_queue = deque(maxlen=self.action_seq_len)
        self.state_queue = deque(maxlen=self.states_seq_len)

    @torch.no_grad()
    def select_action(self, states: Tensor) -> Tensor:
        """
        Select a single action given environment observations.

        Parameters
        ----------
        states: Tensor
            The states tensor. Shape: [B, states_seq_len, state_dim]

        Returns
        -------
        Tensor
            The predicted action sequence. Shape: [B, action_seq_len, action_dim]
        """
        # Note: It's important that this happens after stacking the images into a single key.
        self.state_queue = populate_queues(self.state_queue, states)
        if len(self.action_queue) == 0:
            actions = self.generate_actions(states=states)
            self.action_queue.extend(actions.transpose(0, 1))
        return self.action_queue.popleft()

    def conditional_sample(self, batch_size: int, condition: Tensor) -> Tensor:
        """
        Sample from the diffusion model.

        Parameters
        ----------
        batch_size: int
            The batch size.
        condition: Tensor
            The condition tensor. Shape: [B, condition_dim]

        Returns
        -------
        Tensor
            The sampled action sequence. Shape: [B, action_generate_len, action_dim]

        Raises
        ------
        TypeError
            If the output of `self.noise_scheduler.step` is not a `torch.Tensor`.
        """
        # Sample prior.
        sample = torch.randn((batch_size, self.action_generate_len, self.action_dim))
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            timestep = torch.full(sample.shape[:1], t.item(), dtype=torch.long, device=sample.device)
            model_output = self.unet.forward(sample, timestep=timestep, condition=condition)
            # Compute previous image: x_t -> x_t-1
            step_output = self.noise_scheduler.step(model_output, timestep=int(t.item()), sample=sample)
            if isinstance(step_output, tuple):
                msg = f"Expected `step_output` to be a `torch.Tensor`, got {type(step_output)}."
                raise TypeError(msg)
            sample = step_output.prev_sample

        return sample

    def generate_actions(self, states: Tensor) -> Tensor:
        """
        Generate actions given environment states.

        Parameters
        ----------
        states: Tensor
            The states tensor. Shape: [B, states_seq_len, state_dim]

        Returns
        -------
        Tensor
            The predicted action sequence. Shape: [B, action_seq_len, action_dim]

        Raises
        ------
        ValueError
            If the shapes of `states` do not match the expected shapes.
        """
        batch_size, states_seq_len = states.shape[:2]

        # Input validation
        if states_seq_len != self.states_seq_len:
            msg = f"Expected `states` to have shape [B, {self.states_seq_len}, ...], got {states.shape}."
            raise ValueError(msg)

        # Encode image features and concatenate them all together along with the state vector.
        condition = self.state_to_condition(states)

        # run sampling
        actions = self.conditional_sample(batch_size, condition=condition)

        # Extract `action_seq_len` steps worth of actions (from the current observation).
        start = states_seq_len - 1
        end = start + self.action_seq_len
        return actions[:, start:end]

    def forward(self, states: Tensor, actions: Tensor) -> dict[str, Tensor]:
        """
        Compute the loss for the given batch of observations and actions.

        Parameters
        ----------
        states: Tensor
            The states tensor. Shape: [B, states_seq_len, state_dim]
        actions: Tensor
            The actions tensor. Shape: [B, action_generate_len, action_dim]

        Returns
        -------
        dict[str, Tensor]
            The loss dictionary.

        Raises
        ------
        ValueError
            If the shapes of `states` and `actions` do not match the expected shapes.
        """
        # Input validation
        if states.shape[1] != self.states_seq_len or actions.shape[1] != self.action_generate_len:
            msg = f"Expected `states` to have shape [B, {self.states_seq_len}, ...] and `actions` to have "
            msg += f"shape [B, {self.action_generate_len}, ...], got {states.shape} and {actions.shape}."
            raise ValueError(msg)

        # Encode image features and concatenate them all together along with the state vector.
        condition = self.state_to_condition(states)

        # Forward diffusion.
        # Sample noise to add to the trajectory.
        eps = torch.randn(actions.shape, device=actions.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = IntTensor(
            torch.randint(low=0, high=self.num_train_timesteps, size=(actions.shape[0],), device=actions.device).long(),
        )
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(actions, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet.forward(noisy_trajectory, timestep=timesteps, condition=condition)

        # Compute the loss.
        loss = tf.mse_loss(pred, eps, reduction="none").mean()
        return {"loss": loss}


def populate_queues(queue: deque[Tensor], tensor: Tensor) -> deque[Tensor]:
    """
    Populate a queue with a tensor.

    Parameters
    ----------
    queue: deque[Tensor]
        The queue to populate.
    tensor: Tensor
        The tensor to add to the queue.

    Returns
    -------
    deque[Tensor]
        The populated queue.
    """
    if len(queue) != queue.maxlen:
        # initialize by copying the first observation several times until the queue is full
        while len(queue) != queue.maxlen:
            queue.append(tensor)
    else:
        # add latest observation to the queue
        queue.append(tensor)
    return queue
