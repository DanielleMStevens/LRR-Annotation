import sys
import os
import platform

from setuptools import setup, find_packages

with open('README.md') as f:
	long_description = f.read()

setup(
	name = 'lrr_annot',
	version = 1.0,
	description = 'Protein domain annotation tool',
	long_description = long_description,
	long_description_content_type = 'text/markdown',
	author = 'Boyan Xu, Alois Cerbu, Daven Lim, Chris Tralie, Ksenia Krasileva',
	author_email = 'boxu@berkeley.edu',
	url = 'https://github.com/amcerbu/LRR-Annotation/',
	license = 'MIT',
	packages = find_packages(where = 'lrr_annot'),
	install_requires = ['numpy', 'scipy', 'matplotlib', 'biopython', 'jupyter', 'tqdm']
)