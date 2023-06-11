#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="surprise-stream",
    version="1.0",
    description="Stream GCL Dataset",
    author="Iordanis Fostiropoulos",
    author_email="mail@iordanis.me",
    url="https://iordanis.me/",
    python_requires=">3.10",
    long_description=open("README.md").read(),
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "autods==1.0",
    ],
    extras_require={
        "dev": [
            "mypy==1.2.0",
            "pytest==7.3.0",
            "pylint==2.17.2",
            "flake8==6.0.0",
            "black==23.3.0",
            "types-requests==2.28.11.17",
        ],
    },
)
