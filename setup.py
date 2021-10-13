from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project implements a simple transliterator to convert words from the Ligurian language (written in the Grafîa Ofiçiâ) to their pronunciation (written with IPA symbols). Under the hood the transliterator uses a character level Encoder-Decoder architecture with Attention.',
    author='fleanend',
    license='MIT',
)
