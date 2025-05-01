"""Phonemization and synthesis for Piper."""

import json
import logging
import threading
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import numpy as np
import onnxruntime

from .config import PhonemeType, PiperConfig
from .phoneme_ids import phonemes_to_ids
from .phonemize_espeak import ESPEAK_DATA_DIR, EspeakPhonemizer
from .tashkeel import TashkeelDiacritizer
from .util import audio_float_to_int16

_ESPEAK_PHONEMIZER: Optional[EspeakPhonemizer] = None
_ESPEAK_PHONEMIZER_LOCK = threading.Lock()

_LOGGER = logging.getLogger(__name__)


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

        # Arabic diacritization
        if (self.config.espeak_voice == "ar") and self.use_tashkeel:
            if self.tashkeel_diacritizier is None:
                self.tashkeel_diacritizier = TashkeelDiacritizer()

            text = self.tashkeel_diacritizier(
                text, taskeen_threshold=self.taskeen_threshold
            )

        if self.config.phoneme_type == PhonemeType.ESPEAK:
            with _ESPEAK_PHONEMIZER_LOCK:
                if _ESPEAK_PHONEMIZER is None:
                    _ESPEAK_PHONEMIZER = EspeakPhonemizer(self.espeak_data_dir)

                return _ESPEAK_PHONEMIZER.phonemize(self.config.espeak_voice, text)

        if self.config.phoneme_type == PhonemeType.TEXT:
            return [list(unicodedata.normalize("NFD", text))]

        raise ValueError(f"Unexpected phoneme type: {self.config.phoneme_type}")

    def phonemes_to_ids(self, phonemes: list[str]) -> list[int]:
        """Phonemes to ids."""
        return phonemes_to_ids(phonemes, self.config.phoneme_id_map)

    def synthesize(
        self,
        text: str,
        wav_file: wave.Wave_write,
        speaker_id: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sentence_silence: float = 0.0,
        set_wav_params: bool = True,
    ):
        """Synthesize WAV audio from text."""
        if set_wav_params:
            wav_file.setframerate(self.config.sample_rate)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setnchannels(1)  # mono

        for audio_bytes in self.synthesize_stream_raw(
            text,
            speaker_id=speaker_id,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w=noise_w,
            sentence_silence=sentence_silence,
        ):
            wav_file.writeframes(audio_bytes)

    def synthesize_stream_raw(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sentence_silence: float = 0.0,
    ) -> Iterable[bytes]:
        """Synthesize raw audio per sentence from text."""
        sentence_phonemes = self.phonemize(text)

        # 16-bit mono
        num_silence_samples = int(sentence_silence * self.config.sample_rate)
        silence_bytes = bytes(num_silence_samples * 2)

        for phonemes in sentence_phonemes:
            phoneme_ids = self.phonemes_to_ids(phonemes)
            yield self.synthesize_ids_to_raw(
                phoneme_ids,
                speaker_id=speaker_id,
                length_scale=length_scale,
                noise_scale=noise_scale,
                noise_w=noise_w,
            ) + silence_bytes

    def synthesize_ids_to_raw(
        self,
        phoneme_ids: list[int],
        speaker_id: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
    ) -> bytes:
        """Synthesize raw audio from phoneme ids."""
        if length_scale is None:
            length_scale = self.config.length_scale

        if noise_scale is None:
            noise_scale = self.config.noise_scale

        if noise_w is None:
            noise_w = self.config.noise_w

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array(
            [noise_scale, length_scale, noise_w],
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

        # Synthesize through Onnx
        audio = self.session.run(
            None,
            args,
        )[
            0
        ].squeeze((0, 1))
        audio = audio_float_to_int16(audio.squeeze())
        return audio.tobytes()
