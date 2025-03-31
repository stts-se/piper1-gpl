"""Needed so package data is included."""

from pathlib import Path

from skbuild import setup

MODULE_DIR = Path(__file__).parent / "src" / "piper"
ESPEAK_NG_DATA_DIR = MODULE_DIR / "espeak-ng-data"
ESPEAK_NG_DATA_FILES = [
    f.relative_to(MODULE_DIR) for f in ESPEAK_NG_DATA_DIR.rglob("*") if f.is_file()
]

setup(
    name="piper",
    version="1.3.0",
    description="Fast and local neural text-to-speech engine",
    license="Apache-2.0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="The Home Assistant Authors",
    author_email="hello@home-assistant.io",
    keywords=["home", "assistant", "tts", "text-to-speech"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.9",
    install_requires=[
        "onnxruntime>=1,<2",
    ],
    extras_require={
        "dev": [
            "black==24.8.0",
            "flake8==7.1.1",
            "mypy==1.14.0",
            "pylint==3.2.7",
            "pytest==8.3.4",
            "build==1.2.2",
            "scikit-build<1",
            "swig>=4,<5",
            "cmake>=3.18,<4",
            "ninja>=1,<2",
        ],
        "http": [
            "flask>=3,<4",
        ],
    },
    packages=["piper"],
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "piper": [str(p) for p in ESPEAK_NG_DATA_FILES],
    },
    cmake_install_dir="src/piper",
)
