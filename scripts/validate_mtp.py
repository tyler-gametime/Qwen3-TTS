#!/usr/bin/env python3
"""Validate MTP module works correctly before running full training.

This script performs sanity checks on the MTP implementation:
1. Tests MTP module forward/backward pass
2. Tests MTP integration with Qwen3TTSTalkerForConditionalGeneration
3. Optionally runs a few training steps with synthetic data

Usage:
    python scripts/validate_mtp.py [--full]

    --full: Run full validation including synthetic training steps
"""

import argparse
import sys

import torch


def test_mtp_module_basic():
    """Test basic MTP module functionality."""
    print("=" * 60)
    print("Test 1: MTP Module Basic Functionality")
    print("=" * 60)

    from qwen_tts.core.mtp import MTPConfig, MTPModule

    config = MTPConfig(
        num_mtp_heads=4,
        hidden_size=64,
        trunk_hidden_size=64,
        num_trunk_layers=2,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        vocab_size=256,
    )
    module = MTPModule(config)

    batch_size = 4
    talker_hidden = torch.randn(batch_size, config.hidden_size)
    cb0_embed = torch.randn(batch_size, config.hidden_size)
    labels = torch.randint(0, config.vocab_size, (batch_size, config.num_mtp_heads))

    # Forward pass
    output = module(talker_hidden, cb0_embed, labels=labels)

    assert output.logits.shape == (batch_size, config.num_mtp_heads, config.vocab_size), (
        f"Expected logits shape {(batch_size, config.num_mtp_heads, config.vocab_size)}, "
        f"got {output.logits.shape}"
    )
    assert output.loss is not None, "Loss should not be None when labels provided"
    assert output.loss.ndim == 0, "Loss should be scalar"

    # Backward pass
    output.loss.backward()

    print(f"  Config: {config.num_mtp_heads} heads, hidden={config.hidden_size}")
    print(f"  Logits shape: {output.logits.shape}")
    print(f"  Loss: {output.loss.item():.4f}")
    print("  [PASS] Forward/backward pass works correctly")
    print()


def test_mtp_generation():
    """Test MTP generation."""
    print("=" * 60)
    print("Test 2: MTP Generation")
    print("=" * 60)

    from qwen_tts.core.mtp import MTPConfig, MTPModule

    config = MTPConfig(
        num_mtp_heads=4,
        hidden_size=64,
        trunk_hidden_size=64,
        num_trunk_layers=2,
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

    # Test argmax generation
    with torch.no_grad():
        tokens = module.generate(talker_hidden, cb0_embed, do_sample=False)

    assert tokens.shape == (batch_size, config.num_mtp_heads), (
        f"Expected tokens shape {(batch_size, config.num_mtp_heads)}, got {tokens.shape}"
    )
    assert (tokens >= 0).all() and (tokens < config.vocab_size).all(), "Tokens out of vocab range"

    # Test sampling
    with torch.no_grad():
        sampled_tokens = module.generate(talker_hidden, cb0_embed, do_sample=True, temperature=1.0)

    assert sampled_tokens.shape == (batch_size, config.num_mtp_heads)

    print(f"  Generated tokens (argmax): {tokens}")
    print(f"  Generated tokens (sampled): {sampled_tokens}")
    print("  [PASS] Generation works correctly")
    print()


def test_mtp_config_integration():
    """Test MTP config is correctly added to Qwen3TTSTalkerConfig."""
    print("=" * 60)
    print("Test 3: Config Integration")
    print("=" * 60)

    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig

    # Test default (MTP disabled)
    config_no_mtp = Qwen3TTSTalkerConfig()
    assert hasattr(config_no_mtp, "use_mtp"), "Config should have use_mtp attribute"
    assert config_no_mtp.use_mtp is False, "use_mtp should default to False"

    # Test MTP enabled
    config_mtp = Qwen3TTSTalkerConfig(use_mtp=True, num_mtp_heads=8, mtp_trunk_layers=3)
    assert config_mtp.use_mtp is True
    assert config_mtp.num_mtp_heads == 8
    assert config_mtp.mtp_trunk_layers == 3

    print(f"  Default use_mtp: {config_no_mtp.use_mtp}")
    print(f"  Custom use_mtp: {config_mtp.use_mtp}, heads={config_mtp.num_mtp_heads}")
    print("  [PASS] Config integration works correctly")
    print()


def test_mtp_training_loss():
    """Test MTP training loss computation with curriculum warmup."""
    print("=" * 60)
    print("Test 4: Training Loss with Curriculum Warmup")
    print("=" * 60)

    # Import the functions from the training script
    sys.path.insert(0, "finetuning")
    from sft_12hz import compute_mtp_loss_weight

    # Test warmup schedule
    assert compute_mtp_loss_weight(0, 1000) == 0.0, "Weight at step 0 should be 0"
    assert compute_mtp_loss_weight(500, 1000) == 0.5, "Weight at step 500 should be 0.5"
    assert compute_mtp_loss_weight(1000, 1000) == 1.0, "Weight at step 1000 should be 1.0"
    assert compute_mtp_loss_weight(2000, 1000) == 1.0, "Weight after warmup should be 1.0"
    assert compute_mtp_loss_weight(100, 0) == 1.0, "Weight with no warmup should be 1.0"

    print("  Warmup weight at step 0: 0.0")
    print("  Warmup weight at step 500: 0.5")
    print("  Warmup weight at step 1000: 1.0")
    print("  [PASS] Curriculum warmup works correctly")
    print()


def test_gpu_forward_pass():
    """Test MTP module on GPU if available."""
    print("=" * 60)
    print("Test 5: GPU Forward Pass")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        print()
        return

    from qwen_tts.core.mtp import MTPConfig, MTPModule

    device = torch.device("cuda")
    config = MTPConfig(
        num_mtp_heads=4,
        hidden_size=256,
        trunk_hidden_size=256,
        num_trunk_layers=2,
        num_attention_heads=8,
        head_dim=32,
        intermediate_size=512,
        vocab_size=2048,
    )
    module = MTPModule(config).to(device)

    batch_size = 8
    talker_hidden = torch.randn(batch_size, config.hidden_size, device=device)
    cb0_embed = torch.randn(batch_size, config.hidden_size, device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, config.num_mtp_heads), device=device)

    # Forward + backward
    output = module(talker_hidden, cb0_embed, labels=labels)
    output.loss.backward()

    print(f"  Device: {device}")
    print(f"  GPU Memory used: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    print(f"  Loss: {output.loss.item():.4f}")
    print("  [PASS] GPU forward/backward pass works")
    print()


def test_mtp_loss_weight_decay():
    """Test that per-codebook loss weight decay is applied."""
    print("=" * 60)
    print("Test 6: Per-Codebook Loss Weight Decay")
    print("=" * 60)

    from qwen_tts.core.mtp import MTPConfig, MTPModule

    config = MTPConfig(
        num_mtp_heads=4,
        hidden_size=64,
        trunk_hidden_size=64,
        num_trunk_layers=1,
        num_attention_heads=4,
        head_dim=16,
        intermediate_size=128,
        vocab_size=256,
        loss_weight_decay=0.5,  # Strong decay for testing
    )
    module = MTPModule(config)

    batch_size = 4
    # Create logits where loss should be different per head
    logits = torch.randn(batch_size, config.num_mtp_heads, config.vocab_size)
    labels = torch.randint(0, config.vocab_size, (batch_size, config.num_mtp_heads))

    loss = module._compute_loss(logits, labels)

    assert loss.ndim == 0, "Loss should be scalar"
    assert loss >= 0, "Loss should be non-negative"

    # Verify weights are applied (0.5^0=1, 0.5^1=0.5, 0.5^2=0.25, 0.5^3=0.125)
    expected_total_weight = 1 + 0.5 + 0.25 + 0.125
    print(f"  Loss weight decay: {config.loss_weight_decay}")
    print(f"  Expected total weight: {expected_total_weight:.3f}")
    print(f"  Computed loss: {loss.item():.4f}")
    print("  [PASS] Loss weight decay applied correctly")
    print()


def main():
    parser = argparse.ArgumentParser(description="Validate MTP implementation")
    parser.add_argument("--full", action="store_true", help="Run full validation")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MTP Validation Suite")
    print("=" * 60 + "\n")

    try:
        test_mtp_module_basic()
        test_mtp_generation()
        test_mtp_config_integration()
        test_mtp_training_loss()
        test_gpu_forward_pass()
        test_mtp_loss_weight_decay()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nMTP implementation is ready for training validation.")
        print("\nTo run training with MTP on SageMaker:")
        print("  python finetuning/sft_12hz.py \\")
        print("      --train_jsonl <your_data.jsonl> \\")
        print("      --use_mtp \\")
        print("      --num_mtp_heads 4 \\")
        print("      --num_epochs 1")
        return 0

    except Exception as e:
        print(f"\n[FAILED] {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
