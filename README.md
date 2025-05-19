# Piper 1 (GPL)

A self-contained version of [Piper][piper], the fast and local neural text-to-speech engine.

This version directly embeds [espeak-ng][] and therefore has a [GPL license](COPYING).

## Installing

Install with:

``` sh
pip install piper1-tts
```

## Downloading Voices

List voices with:

``` sh
python3 -m piper.download_voices
```

Choose a voice ([samples here][samples]) and download. For example:

``` sh
python3 -m piper.download_voices en_US-lessac-medium
```

This will download to the current directory. Override with `--data-dir <DIR>`

## Running

After downloading the example voice above, run:

``` sh
python3 -m piper -m en_US-lessac-medium -f test.wav -- 'This is a test.'
```

This will write `test.wav` with the sentence "This is a test."
If you have voices in a different directory, use `--data-dir <DIR>`

If you have [ffplay][] installed, omit `-f` to hear the audio immediately:

``` sh
python3 -m piper -m en_US-lessac-medium -- 'This will play on your speakers.'
```

Running Piper this way is slow since it needs to load the model each time. Run the web server unless you need to stream audio (see `--output-raw` from `--help`).

## Web Server

Install the necessary dependencies:

``` sh
python3 -m pip install piper1-tts[http]
```


After downloading the example voice above, run:

``` sh
python3 -m piper.http_server -m en_US-lessac-medium
```

This will start an HTTP server on port 5000 (use `--host` and `--port` to override).
If you have voices in a different directory, use `--data-dir <DIR>`

Now you can get WAV files via HTTP:

``` sh
curl -X POST -H 'Content-Type: application/json' -d '{ "text": "This is a test." }' -o test.wav localhost:5000
```

The JSON data fields area:

* `text` (required) - text to synthesize
* `voice` (optional) - name of voice to use; defaults to `-m <VOICE>`
* `speaker` (optional) - name of speaker for multi-speaker voices
* `speaker_id` (optional) - id of speaker for multi-speaker voices; overrides `speaker`
* `length_scale` (optional) - speaking speed; defaults to 1
* `noise_scale` (optional) - speaking variability
* `noise_w_scale` (optional) - phoneme width variability

Get the available voices with:

``` sh
curl localhost:5000/voices
```


## Training New Voices

See [TRAINING.md](TRAINING.md)

## Building Manually

We use [scikit-build-core](https://github.com/scikit-build/scikit-build-core) along with [cmake](https://cmake.org/) and [swig](https://www.swig.org/) to build a Python module that directly embeds [espeak-ng][].

You will need the following system packages installed (`apt-get`):

* `build-essential`
* `cmake`
* `ninja-build`
* `swig`

To create a dev environment:

``` sh
git clone https://github.com/OHF-voice/piper1-gpl.git
cd piper1-gpl
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .[dev]
```

Next, run `script/dev_build` or manually build the extension:

``` sh
python3 setup.py build_ext --inplace
```

Now you should be able to use `script/run` or manually run Piper:

``` sh
python3 -m piper --help
```

You can manually build wheels with:

``` sh
python3 -m build
```

<!-- Links -->
[piper]: https://github.com/rhasspy/piper
[espeak-ng]: https://github.com/espeak-ng/espeak-ng
[samples]: https://rhasspy.github.io/piper-samples/
[ffplay]: https://ffmpeg.org/ffplay.html
