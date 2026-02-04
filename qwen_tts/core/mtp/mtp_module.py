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
"""Multi-Token Prediction (MTP) module for parallel codebook prediction.

This module implements a parallel prediction strategy for codec-based TTS,
predicting multiple codebook tokens simultaneously instead of sequentially.
Based on research from arxiv:2410.13839.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput

from .mtp_config import MTPConfig


@dataclass
class MTPOutput(ModelOutput):
    """Output from the MTP module.

    Attributes:
        logits: Predicted logits for each MTP head, shape (batch, num_heads, vocab_size).
        loss: Optional loss if labels were provided.
        hidden_states: Optional hidden states from the trunk.
    """

    logits: torch.FloatTensor = None
    loss: torch.FloatTensor | None = None
    hidden_states: torch.FloatTensor | None = None


class MTPRMSNorm(nn.Module):
    """RMS Normalization for MTP module."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MTPMLP(nn.Module):
    """Gated MLP for MTP trunk layers."""

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.hidden_size = config.trunk_hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MTPAttention(nn.Module):
    """Self-attention layer for MTP trunk.

    Uses standard multi-head attention (or GQA if num_key_value_heads differs).
    No positional embeddings since we're operating on a single frame context.
    """

    def __init__(self, config: MTPConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.trunk_hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = config.dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = MTPRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MTPRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Expand KV for GQA
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class MTPTrunkLayer(nn.Module):
    """Single transformer layer for the MTP trunk."""

    def __init__(self, config: MTPConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MTPAttention(config, layer_idx)
        self.mlp = MTPMLP(config)
        self.input_layernorm = MTPRMSNorm(config.trunk_hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MTPRMSNorm(config.trunk_hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MTPModule(nn.Module):
    """Multi-Token Prediction module for parallel codebook prediction.

    Architecture:
    1. Input projection: Takes hidden state from talker + cb0 embedding
    2. Codebook conditioning: Adds learnable embeddings for target positions
    3. Shared trunk: Small transformer (2-3 layers) processes combined input
    4. Parallel heads: Separate MLPs predict each codebook token

    The module predicts codebooks 1 to num_mtp_heads in parallel, while
    remaining codebooks (num_mtp_heads+1 to 31) fall back to AR generation.

    Example usage:
        >>> config = MTPConfig(num_mtp_heads=4, hidden_size=1024)
        >>> mtp = MTPModule(config)
        >>> # hidden_state: (batch, hidden_size) from talker
        >>> # cb0_embed: (batch, hidden_size) embedding of first codebook token
        >>> output = mtp(hidden_state, cb0_embed)
        >>> # output.logits: (batch, 4, vocab_size)
    """

    def __init__(self, config: MTPConfig):
        super().__init__()
        self.config = config

        # Input projection if hidden_size != trunk_hidden_size
        if config.hidden_size != config.trunk_hidden_size:
            self.input_proj = nn.Linear(config.hidden_size, config.trunk_hidden_size, bias=True)
        else:
            self.input_proj = nn.Identity()

        # Codebook position embeddings for conditioning
        if config.use_codebook_conditioning:
            self.codebook_embeddings = nn.Embedding(config.num_mtp_heads, config.codebook_embed_dim)
            self.codebook_proj = nn.Linear(config.codebook_embed_dim, config.trunk_hidden_size, bias=False)
        else:
            self.codebook_embeddings = None
            self.codebook_proj = None

        # Shared transformer trunk
        self.trunk_layers = nn.ModuleList(
            [MTPTrunkLayer(config, layer_idx) for layer_idx in range(config.num_trunk_layers)]
        )
        self.trunk_norm = MTPRMSNorm(config.trunk_hidden_size, eps=config.rms_norm_eps)

        # Parallel prediction heads - one per codebook position
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.trunk_hidden_size, config.intermediate_size, bias=False),
                    nn.SiLU(),
                    nn.Linear(config.intermediate_size, config.vocab_size, bias=False),
                )
                for _ in range(config.num_mtp_heads)
            ]
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, MTPRMSNorm):
            module.weight.data.fill_(1.0)

    def forward(
        self,
        talker_hidden: torch.Tensor,
        cb0_embed: torch.Tensor,
        labels: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> MTPOutput:
        """Forward pass for parallel codebook prediction.

        Args:
            talker_hidden: Hidden state from the talker model, shape (batch, hidden_size).
            cb0_embed: Embedding of the first codebook token (cb0), shape (batch, hidden_size).
            labels: Optional target codebook tokens for computing loss,
                shape (batch, num_mtp_heads) with values in [0, vocab_size).
            output_hidden_states: Whether to return trunk hidden states.

        Returns:
            MTPOutput with logits for each head and optional loss.
        """
        batch_size = talker_hidden.shape[0]
        device = talker_hidden.device

        # Combine talker hidden state and cb0 embedding
        # Shape: (batch, 2, hidden_size)
        combined = torch.stack([talker_hidden, cb0_embed], dim=1)
        hidden_states = self.input_proj(combined)

        # Add codebook position conditioning if enabled
        # We expand to (batch, 2 + num_heads, trunk_hidden_size) for processing
        if self.codebook_embeddings is not None:
            # Create position indices for each head: [0, 1, 2, ..., num_mtp_heads-1]
            positions = torch.arange(self.config.num_mtp_heads, device=device)
            cb_embeds = self.codebook_proj(self.codebook_embeddings(positions))  # (num_heads, trunk_hidden_size)
            cb_embeds = cb_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_heads, trunk_hidden_size)

            # Append codebook queries to the sequence
            hidden_states = torch.cat([hidden_states, cb_embeds], dim=1)  # (batch, 2+num_heads, trunk_hidden_size)

        # Process through trunk layers
        for layer in self.trunk_layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.trunk_norm(hidden_states)

        # Extract hidden states for prediction heads
        # If using codebook conditioning, take positions 2:2+num_heads
        # Otherwise, use the last position and broadcast
        if self.codebook_embeddings is not None:
            head_hidden = hidden_states[:, 2 : 2 + self.config.num_mtp_heads, :]  # (batch, num_heads, trunk_hidden)
        else:
            # Use the combined representation for all heads
            head_hidden = hidden_states[:, -1:, :].expand(-1, self.config.num_mtp_heads, -1)

        # Apply parallel prediction heads
        logits_list = []
        for i, head in enumerate(self.heads):
            head_logits = head(head_hidden[:, i, :])  # (batch, vocab_size)
            logits_list.append(head_logits)

        logits = torch.stack(logits_list, dim=1)  # (batch, num_heads, vocab_size)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)

        return MTPOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states if output_hidden_states else None,
        )

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss across all heads.

        Uses exponential decay weighting: loss_k is weighted by decay^k.

        Args:
            logits: Predicted logits, shape (batch, num_heads, vocab_size).
            labels: Target tokens, shape (batch, num_heads).

        Returns:
            Weighted average loss.
        """
        batch_size, num_heads, vocab_size = logits.shape
        decay = self.config.loss_weight_decay

        total_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        total_weight = 0.0

        for k in range(num_heads):
            weight = decay**k
            head_logits = logits[:, k, :]  # (batch, vocab_size)
            head_labels = labels[:, k]  # (batch,)

            head_loss = F.cross_entropy(head_logits, head_labels, reduction="mean")
            total_loss = total_loss + weight * head_loss
            total_weight += weight

        return total_loss / total_weight

    def generate(
        self,
        talker_hidden: torch.Tensor,
        cb0_embed: torch.Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Generate codebook tokens for positions 1 to num_mtp_heads.

        Args:
            talker_hidden: Hidden state from talker, shape (batch, hidden_size).
            cb0_embed: Embedding of cb0 token, shape (batch, hidden_size).
            temperature: Sampling temperature.
            top_k: Top-k filtering value.
            top_p: Nucleus sampling probability.
            do_sample: Whether to sample or take argmax.

        Returns:
            Generated tokens, shape (batch, num_mtp_heads).
        """
        output = self.forward(talker_hidden, cb0_embed)
        logits = output.logits  # (batch, num_heads, vocab_size)

        if not do_sample:
            return logits.argmax(dim=-1)

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        batch_size, num_heads, vocab_size = probs.shape
        probs_flat = probs.view(batch_size * num_heads, vocab_size)
        tokens_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        tokens = tokens_flat.view(batch_size, num_heads)

        return tokens

    @classmethod
    def from_talker_config(cls, talker_config, num_mtp_heads: int = 4, **kwargs):
        """Create MTPModule from a Qwen3TTSTalkerConfig.

        Args:
            talker_config: The Qwen3TTSTalkerConfig instance.
            num_mtp_heads: Number of parallel prediction heads.
            **kwargs: Override any other MTPConfig parameters.

        Returns:
            Initialized MTPModule instance.
        """
        config = MTPConfig.from_talker_config(talker_config, num_mtp_heads=num_mtp_heads, **kwargs)
        return cls(config)
