"""Flask web server with HTTP API for Piper."""

import argparse
import io
import logging
import wave
from pathlib import Path

from flask import Flask, request

from . import PiperVoice, SynthesisConfig

_LOGGER = logging.getLogger()


def main() -> None:
    """Run HTTP server."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host")
    parser.add_argument("--port", type=int, default=5000, help="HTTP server port")
    #
    parser.add_argument("-m", "--model", required=True, help="Path to Onnx model file")
    #
    parser.add_argument("-s", "--speaker", type=int, help="Id of speaker (default: 0)")
    parser.add_argument(
        "--length-scale", "--length_scale", type=float, help="Phoneme length"
    )
    parser.add_argument(
        "--noise-scale", "--noise_scale", type=float, help="Generator noise"
    )
    parser.add_argument(
        "--noise-w-scale",
        "--noise_w_scale",
        "--noise-w",
        "--noise_w",
        type=float,
        help="Phoneme width noise",
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
        help="Data directory to check for downloaded models (default: current directory)",
    )
    #
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug(args)

    # Download voice if file doesn't exist
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
    voice = PiperVoice.load(model_path, use_cuda=args.cuda)
    syn_config = SynthesisConfig(
        speaker_id=args.speaker,
        length_scale=args.length_scale,
        noise_scale=args.noise_scale,
        noise_w_scale=args.noise_w_scale,
    )

    # 16-bit samples for silence
    silence_int16_bytes = bytes(
        int(voice.config.sample_rate * args.sentence_silence * 2)
    )

    # Create web server
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def app_synthesize() -> bytes:
        if request.method == "POST":
            text = request.data.decode("utf-8")
        else:
            text = request.args.get("text", "")

        text = text.strip()
        if not text:
            raise ValueError("No text provided")

        _LOGGER.debug("Synthesizing text: %s", text)
        with io.BytesIO() as wav_io:
            wav_file: wave.Wave_write = wave.open(wav_io, "wb")
            with wav_file:
                wav_params_set = False
                for i, audio_chunk in enumerate(voice.synthesize(text, syn_config)):
                    if not wav_params_set:
                        wav_file.setframerate(audio_chunk.sample_rate)
                        wav_file.setsampwidth(audio_chunk.sample_width)
                        wav_file.setnchannels(audio_chunk.sample_channels)
                        wav_params_set = True

                    if i > 0:
                        wav_file.writeframes(silence_int16_bytes)

                    wav_file.writeframes(audio_chunk.audio_int16_bytes)

            return wav_io.getvalue()

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
