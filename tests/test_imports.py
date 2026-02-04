"""Test that core modules can be imported without errors."""



class TestImports:
    """Test module imports."""

    def test_import_core_models(self):
        """Test that core models can be imported."""
        from qwen_tts.core.models import (
            Qwen3TTSConfig,
            Qwen3TTSForConditionalGeneration,
        )
        assert Qwen3TTSConfig is not None
        assert Qwen3TTSForConditionalGeneration is not None

    def test_import_tokenizer_12hz(self):
        """Test that 12Hz tokenizer can be imported."""
        from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Config,
        )
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Model,
        )
        assert Qwen3TTSTokenizerV2Config is not None
        assert Qwen3TTSTokenizerV2Model is not None

    def test_import_inference(self):
        """Test that inference module can be imported."""
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
        assert Qwen3TTSModel is not None
        assert Qwen3TTSTokenizer is not None

    def test_import_processing(self):
        """Test that processing module can be imported."""
        from qwen_tts.core.models import Qwen3TTSProcessor
        assert Qwen3TTSProcessor is not None


class TestConfigInstantiation:
    """Test that config classes can be instantiated."""

    def test_tts_config_defaults(self):
        """Test Qwen3TTSConfig can be created with defaults."""
        from qwen_tts.core.models import Qwen3TTSConfig
        config = Qwen3TTSConfig()
        assert config is not None

    def test_tokenizer_config_defaults(self):
        """Test tokenizer config can be created with defaults."""
        from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Config,
        )
        config = Qwen3TTSTokenizerV2Config()
        assert config is not None
