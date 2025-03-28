"""Phonemization with espeak-ng."""

import unicodedata
from pathlib import Path
from typing import List, Union

from . import espeak_ng
from .const import (
    AUDIO_OUTPUT_SYNCHRONOUS,
    CLAUSE_TYPE_SENTENCE,
    IPA_PHONEMES,
    ClauseTerminator,
    EspeakError,
    espeakCHARS_AUTO,
)


class EspeakPhonemizer:
    """Phonemizer that uses espeak-ng."""

    def __init__(self, espeak_data_dir: Union[str, Path]) -> None:
        """Initialize phonemizer."""
        result = espeak_ng.espeak_Initialize(
            AUDIO_OUTPUT_SYNCHRONOUS,
            0,  # buflength
            str(espeak_data_dir),
            0,  # options
        )
        if result < 0:
            raise EspeakError("Error initializing espeak-ng", result)

    def phonemize(self, voice: str, text: str) -> List[List[str]]:
        """Text to phonemes grouped by sentence."""
        result = espeak_ng.espeak_SetVoiceByName(voice)
        if result < 0:
            raise EspeakError(f"Error setting espeak-ng voice to {voice}", result)

        all_phonemes: List[List[str]] = []
        sentence_phonemes: List[str] = []

        remaining_text = text
        while remaining_text:
            phonemes_str, remaining_text, terminator_code = (
                espeak_ng.py_espeak_TextToPhonemesWithTerminator(
                    remaining_text, espeakCHARS_AUTO, IPA_PHONEMES
                )
            )

            sentence_phonemes.extend(list(unicodedata.normalize("NFD", phonemes_str)))

            terminator = ClauseTerminator.from_espeak(terminator_code)

            # TODO: Use config for punctuation
            if terminator == ClauseTerminator.PERIOD:
                sentence_phonemes.append(".")
            elif terminator == ClauseTerminator.QUESTION:
                sentence_phonemes.append("?")
            elif terminator == ClauseTerminator.EXCLAMATION:
                sentence_phonemes.append("!")
            elif terminator == ClauseTerminator.COMMA:
                sentence_phonemes.append(",")
                sentence_phonemes.append(" ")
            elif terminator == ClauseTerminator.COLON:
                sentence_phonemes.append(":")
                sentence_phonemes.append(" ")
            elif terminator == ClauseTerminator.SEMICOLON:
                sentence_phonemes.append(";")
                sentence_phonemes.append(" ")

            if (terminator_code & CLAUSE_TYPE_SENTENCE) == CLAUSE_TYPE_SENTENCE:
                # End of sentence
                all_phonemes.append(sentence_phonemes)
                sentence_phonemes = []

        if sentence_phonemes:
            all_phonemes.append(sentence_phonemes)

        return all_phonemes
