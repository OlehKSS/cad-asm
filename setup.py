#! /usr/bin/env python
from setuptools import setup, find_packages

DISTNAME = 'ashapes'
DESCRIPTION = 'Active Shapes Method'
AUTHOR = 'Brianna Burton, Oleh Kozynets'
URL = 'https://github.com/OlehKSS/cad-asm'
LICENSE = 'GNU GPL'
DOWNLOAD_URL = 'https://github.com/OlehKSS/cad-asm'

EXTERNAL_PACKAGES = ('numpy')
setup(
    name=DISTNAME,
    author=AUTHOR,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version='0',
    download_url=DOWNLOAD_URL,
    long_description=open('README.md').read(),
    platforms='any',
    packages=find_packages(),
    install_requires=EXTERNAL_PACKAGES)