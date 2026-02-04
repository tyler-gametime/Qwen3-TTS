# Copyright 2026 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration for Multi-Token Prediction (MTP) module."""

from dataclasses import dataclass


@dataclass
class MTPConfig:
    """Configuration for the MTP (Multi-Token Prediction) module.

    This module accelerates codec prediction by predicting multiple codebook
    tokens in parallel instead of sequentially. Based on research from
    "Accelerating Codec-based Speech Synthesis with MTP and Speculative Decoding"
    (arxiv:2410.13839).

    Args:
        num_mtp_heads: Number of codebook positions to predict in parallel.
            Default is 4 (conservative) based on research showing quality
            degrades beyond 4 heads without speculative decoding verification.
        hidden_size: Hidden dimension of the MTP trunk. Should match the
            talker model's hidden size for efficient projection.
        trunk_hidden_size: Hidden dimension within the MTP trunk layers.
            Defaults to hidden_size if not specified.
        num_trunk_layers: Number of transformer layers in the shared trunk.
            More layers = better quality but slower inference.
        num_attention_heads: Number of attention heads in trunk layers.
        num_key_value_heads: Number of key-value heads for GQA. If None,
            defaults to num_attention_heads (standard MHA).
        intermediate_size: Size of the MLP intermediate layer.
        vocab_size: Vocabulary size for each codebook (typically 2048).
        head_dim: Dimension of each attention head.
        hidden_act: Activation function for MLP layers.
        rms_norm_eps: Epsilon for RMS normalization.
        use_codebook_conditioning: Whether to add learnable codebook position
            embeddings to help heads specialize.
        codebook_embed_dim: Dimension of codebook position embeddings.
        dropout: Dropout probability during training.
        loss_weight_decay: Exponential decay factor for per-codebook loss
            weighting. Loss for codebook k is weighted by loss_weight_decay^k.
            Set to 1.0 for uniform weighting.
        warmup_steps: Number of training steps for MTP loss warmup curriculum.
            MTP loss weight α increases from 0 to 1 over these steps.
    """

    num_mtp_heads: int = 4
    hidden_size: int = 1024
    trunk_hidden_size: int | None = None
    num_trunk_layers: int = 2
    num_attention_heads: int = 8
    num_key_value_heads: int | None = None
    intermediate_size: int = 2048
    vocab_size: int = 2048
    head_dim: int = 128
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    use_codebook_conditioning: bool = True
    codebook_embed_dim: int = 64
    dropout: float = 0.0
    loss_weight_decay: float = 0.8
    warmup_steps: int = 1000

    def __post_init__(self):
        if self.trunk_hidden_size is None:
            self.trunk_hidden_size = self.hidden_size
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @classmethod
    def from_talker_config(cls, talker_config, num_mtp_heads: int = 4, **kwargs):
        """Create MTPConfig from a Qwen3TTSTalkerConfig.

        This helper extracts relevant dimensions from the talker config
        to ensure compatibility.

        Args:
            talker_config: The Qwen3TTSTalkerConfig instance.
            num_mtp_heads: Number of parallel prediction heads.
            **kwargs: Override any other MTPConfig parameters.

        Returns:
            MTPConfig instance compatible with the talker.
        """
        return cls(
            num_mtp_heads=num_mtp_heads,
            hidden_size=talker_config.hidden_size,
            vocab_size=getattr(
                talker_config.code_predictor_config, "vocab_size", 2048
            ),
            hidden_act=talker_config.hidden_act,
            rms_norm_eps=talker_config.rms_norm_eps,
            **kwargs,
        )
