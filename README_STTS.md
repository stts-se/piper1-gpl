# Information about STTS' Piper fork

Used Piper version: 1.3.0

The updates we have made are intended to be backward compatible. A description of the updates can be found below.

## A. Training models with existing transcriptions

Affected file: `src/piper/train/vits/dataset.py`

1. If the specified config file already exists, it will not be overwritten

2. If the input csv file contains three columns instead of two, the third column will be used for the `phonemes` variable, instead of generated espeak phonemes.


## B. Default dynamo parameter for torch.onnx.export

Affected file: `src/piper/train/export_onnx.py`

We believe default value for `dynamo` in torch.onnx.export has changed between torch versions, causing an error if this value is unset

---

A detailed comparison can be found here: https://github.com/OHF-Voice/piper1-gpl/compare/main...stts-se:piper1-gpl:main
