#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from setuptools import setup

with open("svj_ntuple_processing/include/VERSION", "r") as f:
    version = f.read().strip()

setup(
    name          = 'svj_ntuple_processing',
    version       = version,
    license       = 'BSD 3-Clause License',
    description   = 'Description text',
    url           = 'https://github.com/tklijnsma/svj_ntuple_processing.git',
    author        = 'Thomas Klijnsma',
    author_email  = 'tklijnsm@gmail.com',
    packages      = ['svj_ntuple_processing'],
    package_data  = {'svj_ntuple_processing': ['include/*']},
    include_package_data = True,
    zip_safe      = False,
    scripts       = [],
    install_requires=['uproot', 'awkward', 'seutils']
    )
