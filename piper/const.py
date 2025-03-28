"""Constants"""

from enum import Enum, auto
from typing import Optional

PAD = "_"  # padding (0)
BOS = "^"  # beginning of sentence
EOS = "$"  # end of sentence

# espeak-ng
AUDIO_OUTPUT_SYNCHRONOUS = 2
espeakCHARS_AUTO = 0
IPA_PHONEMES = 0x02

CLAUSE_INTONATION_FULL_STOP = 0x00000000
CLAUSE_INTONATION_COMMA = 0x00001000
CLAUSE_INTONATION_QUESTION = 0x00002000
CLAUSE_INTONATION_EXCLAMATION = 0x00003000

CLAUSE_TYPE_CLAUSE = 0x00040000
CLAUSE_TYPE_SENTENCE = 0x00080000

CLAUSE_PERIOD = 40 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_SENTENCE
CLAUSE_COMMA = 20 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE
CLAUSE_QUESTION = 40 | CLAUSE_INTONATION_QUESTION | CLAUSE_TYPE_SENTENCE
CLAUSE_EXCLAMATION = 45 | CLAUSE_INTONATION_EXCLAMATION | CLAUSE_TYPE_SENTENCE
CLAUSE_COLON = 30 | CLAUSE_INTONATION_FULL_STOP | CLAUSE_TYPE_CLAUSE
CLAUSE_SEMICOLON = 30 | CLAUSE_INTONATION_COMMA | CLAUSE_TYPE_CLAUSE


class ClauseTerminator(Enum):
    """Terminator for an espeak-ng clause."""

    PERIOD = auto()
    COMMA = auto()
    QUESTION = auto()
    EXCLAMATION = auto()
    COLON = auto()
    SEMICOLON = auto()

    @staticmethod
    def from_espeak(terminator: int) -> "Optional[ClauseTerminator]":
        """Convert espeak-ng terminator to enum."""
        clause_terminator = terminator & 0x000FFFFF
        if clause_terminator == CLAUSE_PERIOD:
            return ClauseTerminator.PERIOD

        if clause_terminator == CLAUSE_COMMA:
            return ClauseTerminator.COMMA

        if clause_terminator == CLAUSE_QUESTION:
            return ClauseTerminator.QUESTION

        if clause_terminator == CLAUSE_EXCLAMATION:
            return ClauseTerminator.EXCLAMATION

        if clause_terminator == CLAUSE_COLON:
            return ClauseTerminator.COLON

        if clause_terminator == CLAUSE_SEMICOLON:
            return ClauseTerminator.SEMICOLON

        return None


class EspeakError(Exception):
    def __init__(self, message, error_code) -> None:
        super().__init__(message)
        self.error_code = error_code

    def __str__(self):
        base_msg = super().__str__()
        return f"{base_msg} (error_code={self.error_code})"
