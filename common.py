import os
from os import path
from os.path import isfile, join, dirname, isdir, exists

import numpy as np

from Bio.PDB import *
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from scipy import linalg, sparse, stats

import matplotlib.pyplot as plt


def compromise(a, b):
    X = np.array([a,b])
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    Y = u @ vh
    return [*Y]

