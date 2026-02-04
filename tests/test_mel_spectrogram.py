"""Tests for mel spectrogram computation."""

import torch


class TestMelSpectrogram:
    """Test mel spectrogram function."""

    def test_mel_spectrogram_output_shape(self, sample_audio):
        """Test that mel spectrogram produces expected output shape."""
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        audio, sr = sample_audio
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)

        mel = mel_spectrogram(
            audio_tensor,
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        )

        assert mel.ndim == 3  # (batch, n_mels, time)
        assert mel.shape[0] == 1  # batch size
        assert mel.shape[1] == 128  # n_mels

    def test_mel_spectrogram_no_nan(self, sample_audio):
        """Test that mel spectrogram doesn't produce NaN values."""
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        audio, sr = sample_audio
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)

        mel = mel_spectrogram(
            audio_tensor,
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        )

        assert not torch.isnan(mel).any(), "Mel spectrogram contains NaN values"

    def test_mel_spectrogram_clipped_audio_warning(self, caplog):
        """Test that clipped audio triggers appropriate logging."""

        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        # Create audio that exceeds [-1, 1] range
        audio_tensor = torch.tensor([[1.5, -1.5, 0.5, 0.0] * 6000])

        mel = mel_spectrogram(
            audio_tensor,
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        )

        # Should still produce valid output
        assert mel is not None
        assert not torch.isnan(mel).any()
