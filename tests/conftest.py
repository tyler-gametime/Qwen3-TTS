"""Pytest configuration and fixtures for Qwen3-TTS tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Return the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype():
    """Return the default dtype for testing."""
    return torch.float32


@pytest.fixture
def sample_audio():
    """Generate a sample audio waveform for testing."""
    # 1 second of audio at 24kHz
    sample_rate = 24000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Simple sine wave at 440Hz
    audio = 0.5 * torch.sin(2 * torch.pi * 440 * t)
    return audio.numpy(), sample_rate


@pytest.fixture
def sample_text():
    """Return sample text for TTS testing."""
    return "Hello, this is a test of the Qwen3 TTS system."
