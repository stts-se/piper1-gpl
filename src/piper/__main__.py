"""Piper main script."""

import argparse
import logging
import shutil
import sys
import tempfile
import time
import wave
from collections.abc import Iterable
from pathlib import Path

from . import PiperVoice
from .audio_playback import AudioPlayer

_FILE = Path(__file__)
_DIR = _FILE.parent
_LOGGER = logging.getLogger(_FILE.stem)


def main() -> None:
    """Run piper text-to-speech engine."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Path to Onnx model file")
    parser.add_argument("-c", "--config", help="Path to model config file")
    parser.add_argument(
        "-f",
        "--output-file",
        "--output_file",
        help="Path to output WAV file (default: stdout)",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        "--output_dir",
        help="Path to output directory (default: cwd)",
    )
    parser.add_argument(
        "--output-raw",
        "--output_raw",
        action="store_true",
        help="Stream raw audio to stdout",
    )
    #
    parser.add_argument("-s", "--speaker", type=int, help="Id of speaker (default: 0)")
    parser.add_argument(
        "--length-scale", "--length_scale", type=float, help="Phoneme length"
    )
    parser.add_argument(
        "--noise-scale", "--noise_scale", type=float, help="Generator noise"
    )
    parser.add_argument(
        "--noise-w", "--noise_w", type=float, help="Phoneme width noise"
    )
    #
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    #
    parser.add_argument(
        "--sentence-silence",
        "--sentence_silence",
        type=float,
        default=0.0,
        help="Seconds of silence after each sentence",
    )
    #
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        action="append",
        default=[str(Path.cwd())],
        help="Data directory to check for voice models (default: current directory)",
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args, texts = parser.parse_known_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    texts: Iterable[str]
    if texts:
        texts = [" ".join(texts)]
    else:
        texts = sys.stdin

    def lines() -> Iterable[str]:
        for line in texts:
            line = line.strip()
            if line:
                yield line

    model_path = Path(args.model)
    if not model_path.exists():
        # Look in data directories
        voice_name = args.model
        for data_dir in args.data_dir:
            maybe_model_path = Path(data_dir) / f"{voice_name}.onnx"
            _LOGGER.debug("Checking '%s'", maybe_model_path)
            if maybe_model_path.exists():
                model_path = maybe_model_path
                break

    if not model_path.exists():
        raise ValueError(
            f"Unable to find voice: {model_path} (use piper.download_voices)"
        )

    # Load voice
    _LOGGER.debug("Loading voice: '%s'", model_path)
    voice = PiperVoice.load(model_path, use_cuda=args.cuda)
    synthesize_args = {
        "speaker_id": args.speaker,
        "length_scale": args.length_scale,
        "noise_scale": args.noise_scale,
        "noise_w": args.noise_w,
        "sentence_silence": args.sentence_silence,
    }

    wav_file: wave.Wave_write
    if args.output_raw:
        # Write raw audio to stdout as its produced
        for line in lines():
            audio_stream = voice.synthesize_stream_raw(line, **synthesize_args)
            for audio_bytes in audio_stream:
                sys.stdout.buffer.write(audio_bytes)
                sys.stdout.buffer.flush()
    elif args.output_dir:
        # Write multiple WAV files to a directory, one per line
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for line in lines():
            wav_path = output_dir / f"{time.monotonic_ns()}.wav"
            wav_file = wave.open(str(wav_path), "wb")
            with wav_file:
                voice.synthesize(line, wav_file, **synthesize_args)

            _LOGGER.info("Wrote %s", wav_path)
    else:
        if args.output_file == "-":
            # Write WAV file to stdout
            with tempfile.NamedTemporaryFile("wb+", suffix=".wav") as temp_wav_file:
                wav_file = wave.open(temp_wav_file.name, "wb")
                with wav_file:
                    wav_file.setframerate(voice.config.sample_rate)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setnchannels(1)  # mono

                    for line in lines():
                        voice.synthesize(
                            line, wav_file, set_wav_params=False, **synthesize_args
                        )

                temp_wav_file.seek(0)
                shutil.copyfileobj(temp_wav_file, sys.stdout.buffer)
        elif (not args.output_file) and AudioPlayer.is_available():
            with AudioPlayer(voice.config.sample_rate) as player:
                for line in lines():
                    audio_stream = voice.synthesize_stream_raw(line, **synthesize_args)
                    for audio_bytes in audio_stream:
                        player.play(audio_bytes)
        else:
            # Write to WAV file
            if not args.output_file:
                _LOGGER.warning(
                    "Audio playback is not available (ffplay). Writing audio to output.wav."
                )
                args.output_file = "output.wav"

            wav_file = wave.open(args.output_file, "wb")
            with wav_file:
                wav_file.setframerate(voice.config.sample_rate)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setnchannels(1)  # mono

                for line in lines():
                    voice.synthesize(
                        line, wav_file, set_wav_params=False, **synthesize_args
                    )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
