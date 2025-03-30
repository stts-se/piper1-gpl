# Piper 1 (GPL)

A self-contained version of [Piper](https://github.com/rhasspy/piper), the fast and local neural text-to-speech engine.

This version directly embeds [espeak-ng][] and therefore has a [GPL license](COPYING).

## Installing

Install with:

``` sh
pip install piper
```

## Downloading Voices

List voices with:

``` sh
python3 -m piper.download_voices
```

Choose a voice ([samples here](https://rhasspy.github.io/piper-samples/)) and download. For example:

``` sh
python3 -m piper.download_voices en_US-lessac-medium
```

This will download to the current directory. Override with `--data-dir <DIR>`

## Running

After downloading the example voice above, run:

``` sh
echo 'This is a test.' | python3 -m piper -m en_US-lessac-medium -f test.wav
```

This will write `test.wav` with the sentence "This is a test."
If you have voices in a different directory, use `--data-dir <DIR>`

Running Piper this way is slow since it needs to load the model each time. Run the web server unless you need to stream audio (see `--output-raw` from `--help`).

## Web Server

Install the necessary dependencies:

``` sh
python3 -m pip install piper[http]
```


After downloading the example voice above, run:

``` sh
python3 -m piper.http_server -m en_US-lessac-medium
```

This will start an HTTP server on port 5000 (use `--host` and `--port` to override).
If you have voices in a different directory, use `--data-dir <DIR>`

Now you can get WAV files via HTTP:

``` sh
curl -X POST -H 'Content-Type: text/plain' -d 'This is a test.' -o test.wav localhost:5000
```

## Building Manually

We use [scikit-build-core](https://github.com/scikit-build/scikit-build-core) along with [cmake](https://cmake.org/) and [swig](https://www.swig.org/) to build a Python module that directly embeds [espeak-ng][].

To create a dev environment:

``` sh
git clone https://github.com/OHF-voice/piper1-gpl.git
cd piper1-gpl
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .[dev]
```

You can manually build wheels with:

``` sh
python3 -m build
```

<!-- Links -->
[espeak-ng]: https://github.com/espeak-ng/espeak-ng
