"""Setup script for EMG_PROSTUDIO."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="emg_prostudio",
    version="0.1.0",
    author="PRIMOCOSMOS",
    description="Professional EMG signal analysis studio for exercise monitoring",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PRIMOCOSMOS/EMG_PROSTUDIO",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "pywavelets>=1.1.1",
        "scikit-learn>=1.0.0",
        "PyQt5>=5.15.0",
        "pyqtgraph>=0.13.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "h5py>=3.7.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-cov>=3.0.0"],
        "dl": ["torch>=2.0.0", "torchvision>=0.15.0"],
    },
)
