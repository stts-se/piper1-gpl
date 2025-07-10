"""Phonemization and synthesis for Piper."""

import json
import logging
import re
import threading
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import onnxruntime

from .config import PhonemeType, PiperConfig, SynthesisConfig
from .phoneme_ids import phonemes_to_ids
from .phonemize_espeak import ESPEAK_DATA_DIR, EspeakPhonemizer
from .tashkeel import TashkeelDiacritizer

_ESPEAK_PHONEMIZER: Optional[EspeakPhonemizer] = None
_ESPEAK_PHONEMIZER_LOCK = threading.Lock()

_DEFAULT_SYNTHESIS_CONFIG = SynthesisConfig()
_MAX_WAV_VALUE = 32767.0
_PHONEME_BLOCK_PATTERN = re.compile(r"(\[\[.*?\]\])")

_LOGGER = logging.getLogger(__name__)


@dataclass
class AudioChunk:
    """Chunk of raw audio."""

    sample_rate: int
    sample_width: int
    sample_channels: int
    audio_float_array: np.ndarray

    _audio_int16_array: Optional[np.ndarray] = None
    _audio_int16_bytes: Optional[bytes] = None

    @property
    def audio_int16_array(self) -> np.ndarray:
        """Get audio as an int16 numpy array."""
        if self._audio_int16_array is None:
            self._audio_int16_array = np.clip(
                self.audio_float_array * _MAX_WAV_VALUE, -_MAX_WAV_VALUE, _MAX_WAV_VALUE
            ).astype(np.int16)

        return self._audio_int16_array

    @property
    def audio_int16_bytes(self) -> bytes:
        """Get audio as 16-bit PCM bytes."""
        return self.audio_int16_array.tobytes()


@dataclass
class PiperVoice:
    """A voice for Piper."""

    session: onnxruntime.InferenceSession
    config: PiperConfig
    espeak_data_dir: Path = ESPEAK_DATA_DIR

    # For Arabic text only
    use_tashkeel: bool = True
    tashkeel_diacritizier: Optional[TashkeelDiacritizer] = None
    taskeen_threshold: Optional[float] = 0.8

    @staticmethod
    def load(
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        use_cuda: bool = False,
        espeak_data_dir: Union[str, Path] = ESPEAK_DATA_DIR,
    ) -> "PiperVoice":
        """Load an ONNX model and config."""
        if config_path is None:
            config_path = f"{model_path}.json"
            _LOGGER.debug("Guessing voice config path: %s", config_path)

        with open(config_path, "r", encoding="utf-8") as config_file:
            config_dict = json.load(config_file)

        providers: list[Union[str, tuple[str, dict[str, Any]]]]
        if use_cuda:
            providers = [
                (
                    "CUDAExecutionProvider",
                    {"cudnn_conv_algo_search": "HEURISTIC"},
                )
            ]
            _LOGGER.debug("Using CUDA")
        else:
            providers = ["CPUExecutionProvider"]

        return PiperVoice(
            config=PiperConfig.from_dict(config_dict),
            session=onnxruntime.InferenceSession(
                str(model_path),
                sess_options=onnxruntime.SessionOptions(),
                providers=providers,
            ),
            espeak_data_dir=Path(espeak_data_dir),
        )

    def phonemize(self, text: str) -> list[list[str]]:
        """Text to phonemes grouped by sentence."""
        global _ESPEAK_PHONEMIZER

        if self.config.phoneme_type == PhonemeType.TEXT:
            # Phonemes = codepoints
            return [list(unicodedata.normalize("NFD", text))]

        if self.config.phoneme_type != PhonemeType.ESPEAK:
            raise ValueError(f"Unexpected phoneme type: {self.config.phoneme_type}")

        phonemes: list[list[str]] = []
        text_parts = _PHONEME_BLOCK_PATTERN.split(text)
        for i, text_part in enumerate(text_parts):
            if text_part.startswith("[["):
                # Phonemes
                if not phonemes:
                    # Start new sentence
                    phonemes.append([])

                if (i > 0) and (text_parts[i - 1].endswith(" ")):
                    phonemes[-1].append(" ")

                phonemes[-1].extend(text_part[2:-2].strip())

                if (i < (len(text_parts)) - 1) and (text_parts[i + 1].startswith(" ")):
                    phonemes[-1].append(" ")

                continue

            # Arabic diacritization
            if (self.config.espeak_voice == "ar") and self.use_tashkeel:
                if self.tashkeel_diacritizier is None:
                    self.tashkeel_diacritizier = TashkeelDiacritizer()

                text_part = self.tashkeel_diacritizier(
                    text_part, taskeen_threshold=self.taskeen_threshold
                )

            with _ESPEAK_PHONEMIZER_LOCK:
                if _ESPEAK_PHONEMIZER is None:
                    _ESPEAK_PHONEMIZER = EspeakPhonemizer(self.espeak_data_dir)

                text_part_phonemes = _ESPEAK_PHONEMIZER.phonemize(
                    self.config.espeak_voice, text_part
                )
                phonemes.extend(text_part_phonemes)

        return phonemes

    def phonemes_to_ids(self, phonemes: list[str]) -> list[int]:
        """Phonemes to ids."""
        return phonemes_to_ids(phonemes, self.config.phoneme_id_map)

    def synthesize(
        self,
        text: str,
        syn_config: Optional[SynthesisConfig] = None,
    ) -> Iterable[AudioChunk]:
        """Synthesize one audio chunk per sentence from from text."""
        if syn_config is None:
            syn_config = _DEFAULT_SYNTHESIS_CONFIG

        sentence_phonemes = self.phonemize(text)

        for phonemes in sentence_phonemes:
            phoneme_ids = self.phonemes_to_ids(phonemes)
            audio = self.phoneme_ids_to_audio(phoneme_ids, syn_config)

            if syn_config.normalize_audio:
                max_val = np.max(np.abs(audio))
                if max_val < 1e-8:
                    # Prevent division by zero
                    audio = np.zeros_like(audio)
                else:
                    audio = audio / max_val

            if syn_config.volume != 1.0:
                audio = audio * syn_config.volume

            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

            yield AudioChunk(
                sample_rate=self.config.sample_rate,
                sample_width=2,
                sample_channels=1,
                audio_float_array=audio,
            )

    def synthesize_wav(
        self,
        text: str,
        wav_file: wave.Wave_write,
        syn_config: Optional[SynthesisConfig] = None,
    ) -> None:
        """Synthesize and write WAV audio from text."""
        first_chunk = True
        for audio_chunk in self.synthesize(text, syn_config=syn_config):
            if first_chunk:
                # Set audio format on first chunk
                wav_file.setframerate(audio_chunk.sample_rate)
                wav_file.setsampwidth(audio_chunk.sample_width)
                wav_file.setnchannels(audio_chunk.sample_channels)
                first_chunk = False

            wav_file.writeframes(audio_chunk.audio_int16_bytes)

    def phoneme_ids_to_audio(
        self, phoneme_ids: list[int], syn_config: Optional[SynthesisConfig] = None
    ) -> np.ndarray:
        """Synthesize raw audio from phoneme ids."""
        if syn_config is None:
            syn_config = _DEFAULT_SYNTHESIS_CONFIG

        speaker_id = syn_config.speaker_id
        length_scale = syn_config.length_scale
        noise_scale = syn_config.noise_scale
        noise_w_scale = syn_config.noise_w_scale

        if length_scale is None:
            length_scale = self.config.length_scale

        if noise_scale is None:
            noise_scale = self.config.noise_scale

        if noise_w_scale is None:
            noise_w_scale = self.config.noise_w_scale

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array(
            [noise_scale, length_scale, noise_w_scale],
            dtype=np.float32,
        )

        args = {
            "input": phoneme_ids_array,
            "input_lengths": phoneme_ids_lengths,
            "scales": scales,
        }

        if self.config.num_speakers <= 1:
            speaker_id = None

        if (self.config.num_speakers > 1) and (speaker_id is None):
            # Default speaker
            speaker_id = 0

        if speaker_id is not None:
            sid = np.array([speaker_id], dtype=np.int64)
            args["sid"] = sid

        # Synthesize through onnx
        audio = self.session.run(
            None,
            args,
        )[0].squeeze()

        return audio
