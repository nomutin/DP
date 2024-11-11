# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab, and The HuggingFace Inc. team. All rights reserved.
"""
Networks for the diffusion policy.

Modified from https://github.com/huggingface/lerobot/blob/main/lerobot/common/policies/diffusion/modeling_diffusion.py
"""

import math
from dataclasses import dataclass
from itertools import pairwise

import torch
from einops import rearrange
from torch import Tensor, nn

__all__ = ["ConditionalUnet1d", "ConditionalUnet1dConfig"]


class SinusoidalPosEmb(nn.Module):
    """
    1D sinusoidal positional embeddings as in Attention is All You Need.

    Parameters
    ----------
    dim : int
        Dimension of the embedding.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(10, 16)
    >>> emb = SinusoidalPosEmb(100)
    >>> emb(x).shape
    torch.Size([10, 16, 100])
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, T).

        Returns
        -------
        Tensor
            Output tensor of shape (B, T, dim).
        """
        device = x.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Sequential):
    """
    Conv1d --> GroupNorm --> Mish.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Kernel size.
    n_groups : int
        Number of groups to separate the channels into.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(8, 5, 16)
    >>> block = Conv1dBlock(5, 32, kernel_size=3)
    >>> block(x).shape
    torch.Size([8, 32, 16])
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )


class ConditionalResidualBlock1d(nn.Module):
    """
    ResNet style 1D convolutional block with FiLM modulation for conditioning.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    condition_dim : int
        Dimension of the conditioning.
    kernel_size : int
        Kernel size for `Conv1dBlock`.
    n_groups : int
        Number of groups for `Conv1dBlock`.

    Examples
    --------
    >>> import torch
    >>> x = torch.randn(8, 5, 16)
    >>> condition = torch.randn(8, 50)
    >>> block = ConditionalResidualBlock1d(
    ...     in_channels=5,
    ...     out_channels=32,
    ...     condition_dim=50,
    ... )
    >>> block(x=x, condition=condition).shape
    torch.Size([8, 32, 16])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(condition_dim, out_channels * 2))
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (B, in_channels, T).
        condition : Tensor
            Condition tensor of shape (B, cond_dim).

        Returns
        -------
        Tensor
            Output tensor of shape (B, out_channels, T).
        """
        out = self.conv1(x)
        cond_embed = self.cond_encoder(condition).unsqueeze(-1)
        scale = cond_embed[:, : self.out_channels]
        bias = cond_embed[:, self.out_channels :]
        out = scale * out + bias
        out = self.conv2(out)
        return out + self.residual_conv(x)  # type: ignore[no-any-return]


@dataclass
class ConditionalUnet1dConfig:
    """
    Configuration class for ConditionalUnet1d.

    Attributes
    ----------
    diffusion_step_embed_dim : int
        The Unet is conditioned on the diffusion timestep via a small non-linear
        network. This is the output dimension of that network, i.e., the embedding dimension.
    condition_dim : int
        The dimension of the conditioning.
    action_dim : int
        The dimension of the output.
    kernel_size : int
        The convolutional kernel size of the diffusion modeling Unet.
    n_groups : int
        Number of groups used in the group norm of the Unet's convolutional blocks.
    down_dims : tuple[int, ...]
        Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet.
        You may provide a variable number of dimensions, therefore also controlling the degree of downsampling.
        Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet.
        The action length should be an integer multiple of the downsampling factor
        (which is determined by `len(down_dims)`)
    """

    diffusion_step_embed_dim: int
    condition_dim: int
    action_dim: int
    kernel_size: int
    n_groups: int
    down_dims: tuple[int, ...]


class ConditionalUnet1d(nn.Module):
    """
    A 1D convolutional UNet with FiLM modulation for conditioning.

    Parameters
    ----------
    config : ConditionalUnet1dConfig
        The configuration for the UNet.

    Examples
    --------
    >>> import torch
    >>> x, timestep, condition = torch.randn(8, 16, 10), torch.randint(0, 10, (8,)), torch.randn(8, 50)
    >>> config = ConditionalUnet1dConfig(
    ...    diffusion_step_embed_dim=128,
    ...    condition_dim=50,
    ...    action_dim=10,
    ...    kernel_size=3,
    ...    n_groups=8,
    ...    down_dims=(32, 64, 128),
    ... )
    >>> unet = ConditionalUnet1d(config)
    >>> unet(x=x, timestep=timestep, condition=condition).shape
    torch.Size([8, 16, 10])
    """

    def __init__(self, config: ConditionalUnet1dConfig) -> None:
        super().__init__()
        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # In channels / out channels for each downsampling block in the Unet's encoder.
        # For the decoder, we just reverse these.
        in_out = [(config.action_dim, config.down_dims[0]), *list(pairwise(config.down_dims))]

        # Unet encoder.
        common_res_block_kwargs = {
            "condition_dim": config.diffusion_step_embed_dim + config.condition_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
        }
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        ConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ],
                ),
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs),
                ConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs),
            ],
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList()
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        ConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        ConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ],
                ),
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_dim, 1),
        )

    def forward(self, x: Tensor, timestep: Tensor, condition: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor. Shape: [B, T, input_dim]
        timestep : Tensor
            Current -1 timestep. Shape: [B,]
        condition : Tensor
            Condition tensor. Shape: [B, condition_dim]

        Returns
        -------
        Tensor
            Diffution model prediciton. Shape: [B, T, input_dim]

        """
        # For 1D convolutions we'll need feature dimension first.
        x = rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        global_feature = torch.cat([timesteps_embed, condition], dim=-1)

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return rearrange(x, "b d t -> b t d")
