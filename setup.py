import sys
import os
import platform

from setuptools import setup
from setuptools.extension import Extension

with open('README.md') as f:
	long_description = f.read()

setup(
	name = 'lrr-annot',
	version = 1.0,
	description = 'Protein domain annotation tool',
	long_description = long_description,
	long_decription_content_type = 'text/markdown',
	author = 'Boyan Xu, Alois Cerbu, Daven Lim, Chris Tralie, Ksenia Krasileva',
	license = 'MIT',
	packages = ['lrr-annot'],
	install_requires = ['numpy', 'scipy', 'matplotlib', 'biopython']
	)