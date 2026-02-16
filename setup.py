from setuptools import setup, find_packages

setup(
    name="futures-foundation-model",
    version="0.1.0",
    description="Pretrained transformer backbone for futures market structure and regime classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    url="https://github.com/YOUR_USERNAME/futures-foundation-model",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.30",
        "pandas>=2.0",
        "numpy>=1.24",
        "safetensors>=0.3",
        "scikit-learn>=1.3",
    ],
    extras_require={"dev": ["pytest>=7.0", "black", "ruff"]},
)