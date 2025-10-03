"""
Setup configuration for Rank1-LawLlama
MTEB (Massive Text Embedding Benchmark) ranking improvements for LawLlama
"""
import os
from setuptools import setup, find_packages

setup(
    name="rank1-lawllama",
    version="0.1.0",
    author="LawLlama Team",
    description="Rank1 for LawLlama - MTEB ranking improvements and evaluation tools",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="mteb_branch"),
    package_dir={"": "mteb_branch"},
    python_requires=">=3.8",
    install_requires=[
        # Add any required dependencies here
        # Common MTEB dependencies
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)