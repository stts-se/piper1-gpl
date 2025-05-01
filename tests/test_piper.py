"""Tests for Piper."""

import tempfile
import wave
from pathlib import Path

from piper import PiperVoice
from piper.phonemize_espeak import EspeakPhonemizer

_DIR = Path(__file__).parent
_TESTS_DIR = _DIR
_TEST_VOICE = _TESTS_DIR / "test_voice.onnx"


def test_load_voice() -> None:
    """Test loading a voice that generates silence."""
    voice = PiperVoice.load(_TEST_VOICE)
    assert voice.config.sample_rate == 22050
    assert voice.config.num_symbols == 256
    assert voice.config.num_speakers == 1
    assert voice.config.phoneme_type == "espeak"
    assert voice.config.espeak_voice == "en-us"


def test_phonemize_synthesize() -> None:
    """Test phonemizing and synthesizing."""
    voice = PiperVoice.load(_TEST_VOICE)
    phonemes = voice.phonemize("Test 1. Test 2.")
    assert phonemes == [
        # Test 1.
        ["t", "ˈ", "ɛ", "s", "t", " ", "w", "ˈ", "ʌ", "n", "."],
        # Test 2.
        ["t", "ˈ", "ɛ", "s", "t", " ", "t", "ˈ", "u", "ː", "."],
    ]

    phoneme_ids = [voice.phonemes_to_ids(ps) for ps in phonemes]

    # Test 1.
    assert phoneme_ids[0] == [
        1,  # BOS
        0,
        32,
        0,  # PAD
        120,
        0,
        61,
        0,
        31,
        0,
        32,
        0,
        3,
        0,
        35,
        0,
        120,
        0,
        102,
        0,
        26,
        0,
        10,
        0,
        2,  # EOS
    ]

    # Test 2.
    assert phoneme_ids[1] == [
        1,  # BOS
        0,
        32,
        0,  # PAD
        120,
        0,
        61,
        0,
        31,
        0,
        32,
        0,
        3,
        0,
        32,
        0,
        120,
        0,
        33,
        0,
        122,
        0,
        10,
        0,
        2,  # EOS
    ]

    audio = voice.synthesize_ids_to_raw(phoneme_ids[0])
    assert len(audio) == 22050 * 2  # 1 second of silence (16-bit samples)
    assert not any(audio)


def test_language_switch_flags_removed() -> None:
    """Test that (language) switch (flags) are removed."""
    phonemizer = EspeakPhonemizer()
    phonemes = phonemizer.phonemize("ar", "test")
    assert phonemes == [["t", "ˈ", "ɛ", "s", "t"]]


def test_synthesize() -> None:
    """Test text to audio synthesis with WAV output."""
    voice = PiperVoice.load(_TEST_VOICE)

    with tempfile.NamedTemporaryFile("wb+", suffix=".wav") as temp_wav_file:
        with wave.open(temp_wav_file, "wb") as wav_file:
            voice.synthesize("This is a test.", wav_file)

        temp_wav_file.seek(0)

        # Verify audio
        with wave.open(temp_wav_file, "rb") as wav_file:
            assert wav_file.getframerate() == 22050
            assert wav_file.getsampwidth() == 2
            assert wav_file.getnchannels() == 1

            audio = wav_file.readframes(wav_file.getnframes())
            assert len(audio) == 22050 * 2  # 1 second of silence
            assert not any(audio)


def test_synthesize_stream_raw() -> None:
    """Test text to audio synthesis with raw stream output."""
    voice = PiperVoice.load(_TEST_VOICE)

    audio_stream = voice.synthesize_stream_raw("This is a test. This is another test.")
    audio_iter = iter(audio_stream)

    # This is a test.
    audio = next(audio_iter)
    assert len(audio) == 22050 * 2  # 1 second of silence
    assert not any(audio)

    # This is a another test.
    audio = next(audio_iter)
    assert len(audio) == 22050 * 2  # 1 second of silence
    assert not any(audio)

    # End of stream
    assert next(audio_iter, None) is None


def test_ar_tashkeel() -> None:
    """Test Arabic diacritization."""
    voice = PiperVoice.load(_TEST_VOICE)
    voice.config.espeak_voice = "ar"

    phonemes_with_diacritics = "bismˌi ʔalllˈahi ʔarrrˈaħmanˌi ʔarrrˈaħiːm"
    phonemes_without_diacritics = "bˈismillˌaːh ʔˈarɹaħmˌaːn ʔarrˈaħiːm"

    # Diacritization is enabled by default
    phonemes = voice.phonemize("بسم الله الرحمن الرحيم")
    assert phonemes_with_diacritics == "".join(phonemes[0])

    # Disable diacritization
    voice.use_tashkeel = False
    phonemes = voice.phonemize("بسم الله الرحمن الرحيم")
    assert phonemes_without_diacritics == "".join(phonemes[0])
