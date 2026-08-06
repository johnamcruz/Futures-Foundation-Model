from setuptools import setup, find_packages

setup(
    name="futures-foundation-model",
    version="2.0.0",
    description="Futures-market foundation layer on pretrained Chronos-Bolt: frozen embeddings + strategy-pluggable training/eval pipelines",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url="https://github.com/johnamcruz/Futures-Foundation-Model",
    packages=find_packages(),
    python_requires=">=3.11",
    # Core install is torch-free (the parent process must never load torch —
    # see futures_foundation/foundation.py). Torch/Chronos run only inside
    # the embed subprocess; install them via the [foundation] extra.
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "ml-training-loop @ git+https://github.com/johnamcruz/ML-training-loop.git@3644eab5a753cba29d73e92991edcd85b8e2ca8f",
    ],
    extras_require={
        "foundation": [
            "torch>=2.0",
            "chronos-forecasting>=2.2",
            "transformers>=4.41,<5",
            "peft>=0.10",
            "safetensors>=0.4,<0.5",
        ],
        "heads": ["xgboost>=2.0", "joblib>=1.3"],
        "regime": ["hmmlearn>=0.3"],   # futures_foundation.regime market-state HMM
        "onnx": ["onnxmltools", "skl2onnx"],
        "data": ["databento>=0.76", "zstandard>=0.23"],
        "dev": ["pytest>=7.0", "black", "ruff", "hmmlearn>=0.3"],
    },
)
