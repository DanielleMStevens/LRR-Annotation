import os
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy import sparse
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from tqdm import tqdm