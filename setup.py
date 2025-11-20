#!/usr/bin/env python3
"""
Setup script for QUITO: Large-Scale Time Series Training and Evaluation Library
"""

from setuptools import setup, find_packages
import os

# Read the README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="quito",
    version="0.1.0",
    description="Large-Scale Time Series Training and Evaluation Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="QUITO Team",
    author_email="quito@example.com",
    url="https://github.com/your-username/QUITO",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=3.8",
            "mypy>=0.900",
            "pre-commit>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "examples": [
            "jupyter>=1.0",
            "matplotlib>=3.5",
            "seaborn>=0.11",
            "plotly>=5.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="time-series, machine-learning, deep-learning, transformers, gan, vae",
    project_urls={
        "Bug Reports": "https://github.com/your-username/QUITO/issues",
        "Source": "https://github.com/your-username/QUITO",
        "Documentation": "https://quito.readthedocs.io/",
    },
) 