#!/usr/bin/env python

from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
      name='torch_lstm',
      version='0.1.0',
      packages=find_packages(exclude=["test"]),
    )
