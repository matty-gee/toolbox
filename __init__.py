#!/usr/bin/env python3

import setuptools

setuptools.setup(
    name='tools',
    version='0.0.0', # increement the version number so can update w/ pip install
    author='Matthew Schafer',
    author_email='matthew.schafer@icahn.mssm.edu',
    description='Testing installation of Package',
    url='https://github.com/mmatty-gee/tools',
    license='MIT',
    packages=['toolbox'],
    install_requires=['requests'],
)