from common import *

"""Loads batches of PBD files from disk, extracts backbones, stores them
in a dictionary, labeled by filename. 
"""
class Loader:
	def __init__(self):
		self.structures = {}

	def load_batch(self, directory, prefix = ''):
		"""Loads batch of PDB files from specified directory and stores
		them in the self.structures dictionary, where they can be looked up
		by filename. Optionally a 

		Args:
			directory (str): Path to folder containing .pdb files
			prefix (str, optional): Prepended to keys when storing structures in
			dictionary (deals with conflicting filenames over multiple imports).
			Defaults to ''.
		"""
		parser = PDBParser()
		for filename in os.listdir(directory):
			if filename.endswith('.pdb'):
				path = os.path.join(directory, filename)
				key = os.path.splitext(filename)[0]
				chain = next(parser.get_structure(key, path).get_chains())
				self.structures[prefix + key] = np.array([np.array(list(residue["CA"].get_vector())) for residue in chain.get_residues()])	


	def load_single(self, directory, filename, prefix = ''):
		"""Loads single PDB file from specified path, stores it in the
		self.structures dictionary

		Args:
			directory (str): Directory containing .pdb file
			filename (str): Name of .pdb file
			prefix (str, optional): Prepended to keys when storing structures in
			dictionary (deals with conflicting filenames over multiple imports).
			Defaults to ''.
		"""
		parser = PDBParser()
		assert filename in os.listdir(directory)

		if filename.endswith('.pdb'):
			path = os.path.join(directory, filename)
			key = os.path.splitext(filename)[0]
			chain = next(parser.get_structure(key, path).get_chains())
			self.structures[prefix + key] = np.array([np.array(list(residue["CA"].get_vector())) for residue in chain.get_residues()])	

	def cache(self, directory, filename):
		"""Caches imported structure to directory

		Args:
			directory (str): Directory to save cache to
			filename (str): Name of cached export
		"""
		pass

	def retrieve(self, directory, filename):
		"""Retrieves cached structure data

		Args:
			directory (str): Directory to retrieve cache from
			filename (str): Name of cached import dictionary
		"""
		pass