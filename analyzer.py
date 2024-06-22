from common import *

class Analyzer:
	def __init__(self):
		pass

	# given two vectors a and b find orthonormal vectors closest to a and b with the same span
	def compromise(self, a, b):
		X = np.array([a,b])
		u, s, vh = np.linalg.svd(X, full_matrices=False)
		Y = u @ vh
		return [*Y]
	
