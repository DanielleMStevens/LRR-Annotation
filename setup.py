import sys
import os
import platform

from setuptools import setup, find_packages

with open('README.md') as f:
	long_description = f.read()

long_description = '''
# geom_lrr

Geometric methods for annotating solenoidal domains in protein structures. 

Example usage of the package can be found at the accompanying 
[GitHub repo](https://github.com/amcerbu/LRR-Annotation/).
'''

setup(
	name = 'geom_lrr',
	version = 1.01,
	description = 'Protein domain annotation tool',
	long_description = long_description,
	long_description_content_type = 'text/markdown',
	author = 'Boyan Xu, Alois Cerbu, Daven Lim, Chris Tralie, Ksenia Krasileva',
	author_email = 'boxu@berkeley.edu',
	url = 'https://github.com/amcerbu/LRR-Annotation/',
	license = 'MIT',
	packages = find_packages(where = 'geom_lrr'),
	install_requires = ['numpy', 'scipy', 'matplotlib', 'tqdm', 'biopython', 'jupyter', 'mayavi', 'wxPython']
)