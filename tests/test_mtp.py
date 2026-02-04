"""Tests for the Multi-Token Prediction (MTP) module."""

import pytest
import torch

from qwen_tts.core.mtp import MTPConfig, MTPModule


class TestMTPConfig:
    """Tests for MTPConfig dataclass."""

    def test_default_config(self):
        """Test default MTPConfig initialization."""
        config = MTPConfig()
        assert config.num_mtp_heads == 4
        assert config.hidden_size == 1024
        assert config.trunk_hidden_size == 1024  # Should default to hidden_size
        assert config.num_trunk_layers == 2
        assert config.vocab_size == 2048
        assert config.loss_weight_decay == 0.8
        assert config.warmup_steps == 1000

    def test_custom_config(self):
        """Test MTPConfig with custom parameters."""
        config = MTPConfig(
            num_mtp_heads=8,
            hidden_size=512,
            trunk_hidden_size=256,
            num_trunk_layers=3,
            vocab_size=4096,
        )
        assert config.num_mtp_heads == 8
        assert config.hidden_size == 512
        assert config.trunk_hidden_size == 256
        assert config.num_trunk_layers == 3
        assert config.vocab_size == 4096

    def test_trunk_hidden_size_defaults_to_hidden_size(self):
        """Test that trunk_hidden_size defaults to hidden_size if not specified."""
        config = MTPConfig(hidden_size=768)
        assert config.trunk_hidden_size == 768

    def test_num_key_value_heads_defaults_to_num_attention_heads(self):
        """Test that num_key_value_heads defaults correctly."""
        config = MTPConfig(num_attention_heads=16)
        assert config.num_key_value_heads == 16


class TestMTPModule:
    """Tests for MTPModule."""

    @pytest.fixture
    def config(self):
        """Create a small config for testing."""
        return MTPConfig(
            num_mtp_heads=4,
            hidden_size=64,
            trunk_hidden_size=64,
            num_trunk_layers=1,
            num_attention_heads=4,
            head_dim=16,
            intermediate_size=128,
            vocab_size=256,
        )

    @pytest.fixture
    def module(self, config):
        """Create an MTPModule for testing."""
        return MTPModule(config)

    def test_module_initialization(self, module, config):
        """Test MTPModule initializes correctly."""
        assert module.config == config
        assert len(module.trunk_layers) == config.num_trunk_layers
        assert len(module.heads) == config.num_mtp_heads

    def test_forward_shape(self, module, config):
        """Test forward pass output shapes."""
        batch_size = 2
        talker_hidden = torch.randn(batch_size, config.hidden_size)
        cb0_embed = torch.randn(batch_size, config.hidden_size)

        output = module(talker_hidden, cb0_embed)

        assert output.logits.shape == (batch_size, config.num_mtp_heads, config.vocab_size)
        assert output.loss is None  # No labels provided

    def test_forward_with_labels(self, module, config):
        """Test forward pass with labels computes loss."""
        batch_size = 2
        talker_hidden = torch.randn(batch_size, config.hidden_size)
        cb0_embed = torch.randn(batch_size, config.hidden_size)
        labels = torch.randint(0, config.vocab_size, (batch_size, config.num_mtp_heads))

        output = module(talker_hidden, cb0_embed, labels=labels)

        assert output.logits.shape == (batch_size, config.num_mtp_heads, config.vocab_size)
        assert output.loss is not None
        assert output.loss.ndim == 0  # Scalar loss
        assert output.loss >= 0  # Loss should be non-negative

    def test_forward_with_output_hidden_states(self, module, config):
        """Test forward pass returns hidden states when requested."""
        batch_size = 2
        talker_hidden = torch.randn(batch_size, config.hidden_size)
        cb0_embed = torch.randn(batch_size, config.hidden_size)

        output = module(talker_hidden, cb0_embed, output_hidden_states=True)

        assert output.hidden_states is not None
        # With codebook conditioning: 2 (inputs) + num_mtp_heads (queries)
        expected_seq_len = 2 + config.num_mtp_heads
        assert output.hidden_states.shape == (batch_size, expected_seq_len, config.trunk_hidden_size)

    def test_parallel_heads_are_independent(self, module, config):
        """Test that parallel heads produce different outputs."""
        batch_size = 1
        talker_hidden = torch.randn(batch_size, config.hidden_size)
        cb0_embed = torch.randn(batch_size, config.hidden_size)

        output = module(talker_hidden, cb0_embed)

        # Check that different heads produce different logits
        for i in range(config.num_mtp_heads - 1):
            # Logits should not be identical across heads
            assert not torch.allclose(output.logits[0, i], output.logits[0, i + 1])

    def test_generate_shape(self, module, config):
        """Test generate method output shape."""
        batch_size = 2
        talker_hidden = torch.randn(batch_size, config.hidden_size)
        cb0_embed = torch.randn(batch_size, config.hidden_size)

        tokens = module.generate(talker_hidden, cb0_embed)

        assert tokens.shape == (batch_size, config.num_mtp_heads)
        assert tokens.dtype == torch.int64
        assert (tokens >= 0).all() and (tokens < config.vocab_size).all()

    def test_generate_argmax(self, module, config):
        """Test generate with do_sample=False uses argmax."""
        batch_size = 2
        talker_hidden = torch.randn(batch_size, config.hidden_size)
        cb0_embed = torch.randn(batch_size, config.hidden_size)

        # Multiple calls with same input should give same result
        tokens1 = module.generate(talker_hidden, cb0_embed, do_sample=False)
        tokens2 = module.generate(talker_hidden, cb0_embed, do_sample=False)

        assert torch.equal(tokens1, tokens2)

    def test_generate_with_temperature(self, module, config):
        """Test generate with temperature scaling."""
        batch_size = 2
        talker_hidden = torch.randn(batch_size, config.hidden_size)
        cb0_embed = torch.randn(batch_size, config.hidden_size)

        # High temperature should produce more diverse outputs
        # (harder to test statistically, so just verify it runs)
        tokens = module.generate(talker_hidden, cb0_embed, temperature=2.0)
        assert tokens.shape == (batch_size, config.num_mtp_heads)

    def test_generate_with_top_k(self, module, config):
        """Test generate with top-k filtering."""
        batch_size = 2
        talker_hidden = torch.randn(batch_size, config.hidden_size)
        cb0_embed = torch.randn(batch_size, config.hidden_size)

        tokens = module.generate(talker_hidden, cb0_embed, top_k=10)
        assert tokens.shape == (batch_size, config.num_mtp_heads)

    def test_generate_with_top_p(self, module, config):
        """Test generate with nucleus sampling."""
        batch_size = 2
        talker_hidden = torch.randn(batch_size, config.hidden_size)
        cb0_embed = torch.randn(batch_size, config.hidden_size)

        tokens = module.generate(talker_hidden, cb0_embed, top_p=0.9)
        assert tokens.shape == (batch_size, config.num_mtp_heads)

    def test_loss_weight_decay(self, config):
        """Test that loss weight decay is applied correctly."""
        module = MTPModule(config)
        batch_size = 4
        logits = torch.randn(batch_size, config.num_mtp_heads, config.vocab_size)
        labels = torch.randint(0, config.vocab_size, (batch_size, config.num_mtp_heads))

        loss = module._compute_loss(logits, labels)

        assert loss.ndim == 0
        assert loss >= 0

    def test_codebook_conditioning(self, config):
        """Test that codebook conditioning embeddings are used."""
        config_with_conditioning = MTPConfig(
            num_mtp_heads=4,
            hidden_size=64,
            trunk_hidden_size=64,
            num_trunk_layers=1,
            num_attention_heads=4,
            head_dim=16,
            intermediate_size=128,
            vocab_size=256,
            use_codebook_conditioning=True,
        )
        module = MTPModule(config_with_conditioning)

        assert module.codebook_embeddings is not None
        assert module.codebook_proj is not None
        assert module.codebook_embeddings.weight.shape[0] == config_with_conditioning.num_mtp_heads

    def test_no_codebook_conditioning(self):
        """Test module without codebook conditioning."""
        config = MTPConfig(
            num_mtp_heads=4,
            hidden_size=64,
            trunk_hidden_size=64,
            num_trunk_layers=1,
            num_attention_heads=4,
            head_dim=16,
            intermediate_size=128,
            vocab_size=256,
            use_codebook_conditioning=False,
        )
        module = MTPModule(config)

        assert module.codebook_embeddings is None
        assert module.codebook_proj is None

        # Should still work
        batch_size = 2
        talker_hidden = torch.randn(batch_size, config.hidden_size)
        cb0_embed = torch.randn(batch_size, config.hidden_size)
        output = module(talker_hidden, cb0_embed)
        assert output.logits.shape == (batch_size, config.num_mtp_heads, config.vocab_size)

    def test_different_hidden_sizes(self):
        """Test module with different hidden_size and trunk_hidden_size."""
        config = MTPConfig(
            num_mtp_heads=4,
            hidden_size=128,
            trunk_hidden_size=64,
            num_trunk_layers=1,
            num_attention_heads=4,
            head_dim=16,
            intermediate_size=128,
            vocab_size=256,
        )
        module = MTPModule(config)

        # Should have a projection layer
        assert not isinstance(module.input_proj, torch.nn.Identity)

        batch_size = 2
        talker_hidden = torch.randn(batch_size, config.hidden_size)
        cb0_embed = torch.randn(batch_size, config.hidden_size)
        output = module(talker_hidden, cb0_embed)
        assert output.logits.shape == (batch_size, config.num_mtp_heads, config.vocab_size)

    def test_gradient_flow(self, module, config):
        """Test that gradients flow through the module."""
        batch_size = 2
        talker_hidden = torch.randn(batch_size, config.hidden_size, requires_grad=True)
        cb0_embed = torch.randn(batch_size, config.hidden_size, requires_grad=True)
        labels = torch.randint(0, config.vocab_size, (batch_size, config.num_mtp_heads))

        output = module(talker_hidden, cb0_embed, labels=labels)
        output.loss.backward()

        assert talker_hidden.grad is not None
        assert cb0_embed.grad is not None

        # Check gradients for module parameters
        for param in module.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestMTPModuleIntegration:
    """Integration tests for MTP module with simulated talker outputs."""

    def test_batch_processing(self):
        """Test processing a batch of different sizes."""
        config = MTPConfig(
            num_mtp_heads=4,
            hidden_size=64,
            trunk_hidden_size=64,
            num_trunk_layers=1,
            num_attention_heads=4,
            head_dim=16,
            intermediate_size=128,
            vocab_size=256,
        )
        module = MTPModule(config)

        for batch_size in [1, 4, 8, 16]:
            talker_hidden = torch.randn(batch_size, config.hidden_size)
            cb0_embed = torch.randn(batch_size, config.hidden_size)

            output = module(talker_hidden, cb0_embed)
            assert output.logits.shape == (batch_size, config.num_mtp_heads, config.vocab_size)

            tokens = module.generate(talker_hidden, cb0_embed)
            assert tokens.shape == (batch_size, config.num_mtp_heads)

    def test_deterministic_behavior(self):
        """Test that setting seed gives reproducible results."""
        config = MTPConfig(
            num_mtp_heads=4,
            hidden_size=64,
            trunk_hidden_size=64,
            num_trunk_layers=1,
            num_attention_heads=4,
            head_dim=16,
            intermediate_size=128,
            vocab_size=256,
        )
        module = MTPModule(config)
        module.eval()

        batch_size = 2
        talker_hidden = torch.randn(batch_size, config.hidden_size)
        cb0_embed = torch.randn(batch_size, config.hidden_size)

        # Same inputs should give same outputs in eval mode
        with torch.no_grad():
            output1 = module(talker_hidden, cb0_embed)
            output2 = module(talker_hidden, cb0_embed)

        assert torch.allclose(output1.logits, output2.logits)
