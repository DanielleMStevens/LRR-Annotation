from common import *

def compromise(a, b):
	"""Given a pair of n-dimensional vectors `a`, `b`, this function
	finds the (n x 2) matrix `Y` whose columns are orthonormal and such that
	the Frobenius norm of `(Y - [a b])` is minimized. This is the "best orthonormal
	approximation to `[a b]`

	Args:
		a (Numpy array): First vector
		b (Numpy array): Second vector

	Returns:
		list: A two-element list of orthonormal vectors
	"""
	X = np.array([a,b])
	u, s, vh = np.linalg.svd(X, full_matrices=False)
	Y = u @ vh
	return [*Y]

def median_slope(data, small = 150, big = 250):
	"""Computes the distribution of slopes of secant lines
	over a data curve (e.g. cumulative winding number)

	Args:
		data (Numpy array): Curve on which to compute slopes of secant lines
		small (int): Lower bound for run. Defaults to 150.
		big (int): Upper bound for run. Defaults to 250. 

	Returns:
		list: A two-element list consisting of the median slope, and `scores`, 
		the histogram of secant line slopes
	"""
	slopes = []
	weights = []
	for i in range(len(data) - small):
		for j in range(i + small, min(i + big, len(data))):
			s = (data[j]-data[i])/(j-i)
			slopes.append(s)
			reg = data[i:j] - s * np.arange(i,j)
			reg -= np.mean(reg)
			# weights.append(np.sqrt(j - i))
			weights.append((j - i) / (1 + np.sum(reg ** 2)))
	
	n_bins = int(np.sqrt(len(slopes)))
	scores = [0 for i in range(n_bins)]
	a = min(slopes)
	b = max(slopes) + 0.01

	for s, weight in zip(slopes, weights):
		bin_index = int(n_bins * (s - a) / (b - a))
		scores[bin_index] += weight

	return a + (np.argmax(scores) / n_bins) * (b - a), scores

def loss(winding, params, slope, penalties):
	l, r = params
	l = int(l)
	r = int(r)
	
	pre = np.array(winding[:l])
	pre -= np.mean(pre)
	
	mid = winding[l:r] - (slope * (np.arange(l, r)))
	mid -= np.mean(mid)
	
	post = np.array(winding[r:])
	if len(post): post -= np.mean(post)    

	return penalties[0] * np.sum(pre ** 2) + penalties[1] * np.sum(mid ** 2) + penalties[0] * np.sum(post ** 2)

class Analyzer:
	def __init__(self):
		self.structures = {}
		self.backbones = {}
		self.normal_bundles = {}
		self.flattened = {}
		self.windings = {}
		self.slopes = {}
		self.regressions = {}

	def load_structures(self, structures):
		"""Updates internal dictionary of three-dimensional protein structures,
		loaded, e.g., by a Loader object.

		Args:
			structures (dict): Dictionary of protein structures
		"""
		self.structures.update(structures)

	def compute_windings(self, smoothing = 20):
		"""Computes the normal bundle framing and cumulative winding number
		for each protein structure stored in the `structures` dictionary.
		The backbone, normal bundle, "flattened" curve (projection to the
		normal bundle), and cumulative winding number are stored to
		the respective member variables: `backbones`, `normal_bundles`, `flattened`,
		and `winding`.

		Args:
			smoothing (int, optional): Amount of smoothing to apply when computing the
			backbone curve. Defaults to 20.
		"""
		for key, structure in self.structures.items():
			X = gaussian_filter(structure, [1, 0]) # smoothed out structure
			dX = gaussian_filter(X, [1, 0], order = 1) # tangent of structure
			Y = gaussian_filter(X, [smoothing, 0]) # backbone
			dY = gaussian_filter(Y, [1, 0], order = 1) # tangent of backbone
			dZ = dY / np.sqrt(np.sum(dY ** 2, axis = 1))[:, np.newaxis] # normalized tangent

			# parallel transport along backbone
			# V[i] is an orthonormal basis for the orthogonal complement of dZ[i]
			V = np.zeros((len(dZ), 2, 3)) 
			V[0] = np.random.rand(2, 3)
			for i, z in enumerate(dZ):
				if i: V[i] = V[i-1]

				# remove projection onto z, the current tangent vector,
				# then enforce orthonormality
				V[i] -= np.outer(V[i] @ z, z)
				V[i] = compromise(*V[i])

			Q = np.zeros((len(dZ), 4))
			for i in range(len(Q)):
				Q[i] = [(X[i] - Y[i]) @ V[i,0], (X[i] - Y[i]) @ V[i,1], \
	    				(X[i] - Y[i]) @ dZ[i], dX[i] @ dZ[i]]
				
			s, c, q, dx = Q.T

			# differentiate the appropriate variables
			ds = gaussian_filter(s, 1, order = 1)
			dc = gaussian_filter(c, 1, order = 1)
			dq = gaussian_filter(q, 1, order = 1)
			r2 = s ** 2 + c ** 2

			# compute discrete integral
			summand = (c * ds - s * dc) / r2
			winding = np.cumsum(summand) / (2 * np.pi)
			winding *= np.sign(winding[-1] - winding[0])

			self.backbones[key] = Y
			self.normal_bundles[key] = V
			self.flattened[key] = np.array([s, c])
			self.windings[key] = winding

	def compute_regressions(self, penalties = [1, 1.5], learning_rate = 0.01, iterations = 10000):
		"""Computes piecewise-linear regressions (constant - slope = m - constant) over
		all cumulative winding curves stored in the `winding` dictionary. Writes the parameters
		of these regressions to the `parameters` and `slopes` dictionaries.

		Args:
			penalties (list, optional): Two-element list describing the relative penalties, in the loss function,
			of deviation. The first component refers to the non-coiling regions; the second to the coiling region.
			Defaults to [1, 1.5].
			learning_rate (float, optional): Scalar for gradient descent in parameter optimization. Defaults to 0.01.
			iterations (int, optional): Iterations of gradient descent. Defaults to 10000.
		"""
		for key, winding in self.windings.items():
			n = len(winding)

			parameters = np.array([n // 2, (3 * n) // 4]) # best-guess initialization
			gradient = np.zeros(2)
			delta = [*np.identity(2)]
			prev_grad = np.array(gradient)

			m, _ = median_slope(winding)
			self.slopes[key] = m

			for i in range(iterations):
				present = loss(winding, parameters, m, penalties)
				gradient = np.array([loss(winding, parameters + d, m, penalties) - present for d in delta])
				parameters = parameters - learning_rate * gradient

			if parameters[1] > 0.9 * n:
				parameters[1] = len(winding)

			self.regressions[key] = parameters

	def cache_geometry(self, directory, prefix = ''):
		with open(os.path.join(directory, prefix + 'backbones.pickle'), 'wb') as handle:
			pickle.dump(self.windings, handle, protocol = pickle.HIGHEST_PROTOCOL)
		
		with open(os.path.join(directory, prefix + 'normal_bundles.pickle'), 'wb') as handle:
			pickle.dump(self.normal_bundles, handle, protocol = pickle.HIGHEST_PROTOCOL)
		
		with open(os.path.join(directory, prefix + 'flattened.pickle'), 'wb') as handle:
			pickle.dump(self.flattened, handle, protocol = pickle.HIGHEST_PROTOCOL)

		with open(os.path.join(directory, prefix + 'windings.pickle'), 'wb') as handle:
			pickle.dump(self.windings, handle, protocol = pickle.HIGHEST_PROTOCOL)

	def retrieve_geometry(self, directory, prefix = ''):
		with open(os.path.join(directory, prefix + 'backbones.pickle'), 'rb') as handle:
			self.backbones.update(pickle.load(handle))
		
		with open(os.path.join(directory, prefix + 'normal_bundles.pickle'), 'rb') as handle:
			self.normal_bundles.update(pickle.load(handle))

		with open(os.path.join(directory, prefix + 'flattened.pickle'), 'rb') as handle:
			self.flattened.update(pickle.load(handle))
		
		with open(os.path.join(directory, prefix + 'windings.pickle'), 'rb') as handle:
			self.windings.update(pickle.load(handle))

	def cache_regressions(self, directory, prefix = ''):
		with open(os.path.join(directory, prefix + 'slopes.pickle'), 'wb') as handle:
			pickle.dump(self.slopes, handle, protocol = pickle.HIGHEST_PROTOCOL)

		with open(os.path.join(directory, prefix + 'regressions.pickle'), 'wb') as handle:
			pickle.dump(self.regressions, handle, protocol = pickle.HIGHEST_PROTOCOL)

	def retrieve_regressions(self, directory, prefix = ''):
		with open(os.path.join(directory, prefix + 'slopes.pickle'), 'rb') as handle:
			self.slopes.update(pickle.load(handle))

		with open(os.path.join(directory, prefix + 'regressions.pickle'), 'rb') as handle:
			self.regressions.update(pickle.load(handle))
