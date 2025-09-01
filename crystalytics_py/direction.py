from crystal_structure import CrystalStructure
import numpy as np
import fractions
from utils import *
import some_math_operations as sm
import miscellaneous_routines as misc
import warnings
import collections
from scipy import optimize
import itertools
import operator
import copy
import sympy
from sympy.solvers.diophantine.diophantine import diop_linear
from sympy.core.containers import Tuple

class Direction:

    """
    Represents a set of directions in a crystal lattice and provides methods to compute
    lattice spacings and related properties along those directions.

    Inputs for instantiation
    ----------
    crystal_structure : CrystalStructure
        The crystal structure object containing primitive vectors and motifs.
    directions : np.ndarray
        Array of directions along which lattice spacing is requested. Shape (M, N).
    basis_directions : str, optional
        Basis in which the directions are defined. Either 'global_orthonormal' (default)
        or 'primitive_vector'.
    verbose : bool, optional
        If True, prints additional information during computations (default: False).

    Attributes
    ----------
    primitive_vectors : np.ndarray
        Primitive lattice vectors of the crystal structure.
    motifs : np.ndarray
        Motifs of the crystal structure.
    directions : np.ndarray
        Directions along which lattice spacing is computed.
    basis_directions : str
        Basis in which the directions are defined.
    lattice_spacings : np.ndarray
        Computed lattice spacings along the requested directions.
    shortest_lattice_vectors : np.ndarray
        Shortest lattice vectors (expressed in primitive vector basis) that are
        collinear with the requested directions.
    """

    """
    Inputs:
    
    directions: The directions along which the lattice spacing is requested.
                type = numpy ndarray (M, N)
                        N = number of dimensions (same as self.crystal_structure.primitive_vecs)
                        M = number of requested directions
                dtype = float

    ***Optional inputs***

    basis_direction: The basis vectors w.r.t. to which the components of the
                    requested directions are provided.
                    type = str
                    value = 'global_orthonormal' (default): The primitive lattice 
                                vectors and the desired direction must have their 
                                components defined w.r.t. same orthonormal basis.
                                            or
                            'primitive_vector': The primitive lattice vectors 
                                must have their components defined w.r.t. orthonormal 
                                basis and the desired directions are defined w.r.t. 
                                the primitive vectors. In this case, the directions
                                must be an integer array.
    max_length: It is the maximum search length for finding the shortest lattice vector 
                along any desired direction. Increasing this value increases the
                chance of finding the shortest lattice vector but also increases
                the computational cost.
                type = int or float
                value = default 50
    verbose: Permission to print messages during running of the method.
            type = bool
            value = default False
    """

    _max_lattice_vector_length = 50
    
    def __init__(self, crystal_structure, directions,
                 basis_directions = 'global_orthonormal',
                 verbose=False):

        """
        Initializes the Direction object with the given crystal structure and directions.

        Arguments
        ----------
        crystal_structure : CrystalStructure
            The crystal structure object containing primitive vectors and motifs.
        directions : np.ndarray (float)
            Array of directions along which lattice spacing is requested. Shape (M, N) where
            N is the number of dimensions (same as crystal_structure.primitive_vecs) and
            M is the number of requested directions.
        basis_directions : str, optional
            Basis in which the directions are defined. Either 'global_orthonormal' (default)
            or 'primitive_vector'.
        limit_denom : int, optional
            Maximum denominator for rational approximations (default: 50).
        verbose : bool, optional
            If True, prints additional information during computations (default: False).
        """
        
        self._crystal_structure = crystal_structure
        self._directions = directions
        self._basis_directions = basis_directions
        self.verbose = verbose
        self._compute_pipeline = []

    @property
    def primitive_vectors(self):
        return self._crystal_structure._primitive_vecs
    
    @property
    def motifs(self):
        return self._crystal_structure._motifs
    
    @property
    def directions(self):
        return self._directions
    
    @property
    def basis_directions(self):
        return self._basis_directions
    
    @classmethod
    def set_max_lattice_vector_length(cls, length):
        if isinstance(length, (int, float)) and length > 0:
            cls._max_lattice_vector_length = length
        else:
            raise ValueError("max_lattice_vector_length must be a positive number")

    def compute(self, compute_list='all'):
        self._execute_pipeline()

    def _execute_pipeline(self):
        compute_element_to_method_map = {
            'lattice_spacing': self._compute_lattice_spacing,
        }
        for element in self._compute_pipeline:
            compute_element_to_method_map[element]()

    def _compute_lattice_spacing(self):

        """
        Calculates the lattice spacing along each requested direction.

        The lattice spacing is defined as the length of the shortest lattice vector
        collinear with each direction. 

        Raises
        ------
        RuntimeError
            If any direction is a null vector or if no suitable lattice vector is found.

        Actions
        ------------
        Sets the following attributes:
            - self._dir_primitive : Directions expressed in primitive vector basis.
            - self._shortest_lattice_vectors : Shortest lattice vectors expressed in primitive vector basis.
            - self._cosine_deviations : Deviation from perfect collinearity.
            - self._lattice_spacings : Computed lattice spacings.

        If self.verbose is set True, the method prints warnings if any direction 
        does not have a perfectly collinear lattice vector.
        """

        if self.basis_directions == 'global_orthonormal':
            # Expressing the components of the desired directions in terms of the primitive lattice vectors
            # This is saved in self._dir_primitive for debugging purposes
            self._dir_primitive = np.linalg.solve(self.crystal_structure.primitive_vecs.T, self.directions.T).T
        else:
            self._dir_primitive = self.directions.copy()

        # None of dir_primitive vectors must be a null vector
        if any(np.all(np.isclose(self._dir_primitive, 0.0), axis = 1)):
            raise RuntimeError('None of directions must be a null vector. ' +
                                'self._dir_primitive = {0}'.format(self._dir_primitive))
        # Setting very small values in self._dir_primitive to zero
        self._dir_primitive[np.isclose(self._dir_primitive, 0)] = 0
    
        # Finding the shortest lattice vector along each desired direction (expressed in primitive vectors basis)
        # self._cosine_deviations gives the deviation from perfect collinearity for debugging purposes.
        # Note that this is not exactly cosine deviation per se and a value of 0 indicates perfect collinearity.
        # It is defined as 0.5*(1-cos(theta)), where theta is the angle between the desired direction
        # and the found shortest lattice vector.
        self._shortest_lattice_vectors, self._cosine_deviations = \
            shortest_collinear_vector_with_integer_components(self._dir_primitive,
                                                              max_length=self._max_lattice_vector_length)
        
        if self.verbose:
            if np.any(self._cosine_deviations > 1e-6):
                n_deviated = np.sum(self._cosine_deviations > 1e-6)
                print(f'Warning: Could not find integer lattice vectors "absolutely" collinear with {n_deviated} out of '+
                      f"{len(self.directions)} directions. \n Please check the self._cosine_deviations > 1e-6 for such directions")

        # lattice spacing(s) along desired direction(s)
        self._lattice_spacings = np.linalg.norm(np.dot(self.primitive_vecs.T, 
                                                       self._shortest_lattice_vectors.T), axis = 0)

    
    @property
    def lattice_spacings(self):
        if hasattr(self, '_lattice_spacings'):
            return self._lattice_spacings
        else:
            raise RuntimeError("lattice spacings have not been computed yet. \n"+
                               "Please run the compute() method first with "+
                               "lattice_spacings in the compute_list.")
    
    @property
    def shortest_lattice_vectors(self):
        if hasattr(self, '_shortest_lattice_vectors'):
            return self._shortest_lattice_vectors
        else:
            raise RuntimeError("shortest lattice vectors have not been computed yet. \n"+
                               "Please run the compute() method first with "+
                               "lattice_spacings in the compute_list.")


    def __setattr__(self, name, value):

        match name:

            case "crystal_structure":
                if isinstance(value, CrystalStructure):
                    self.__dict__[name] = value
                else:
                    raise ValueError("crystal_structure must be an instance of CrystalStructure")
                
            case "directions":
                if not (isinstance(value, np.ndarray) and 
                        (np.issubdtype(value.dtype, np.floating) or 
                         np.issubdtype(value.dtype, np.integer))):
                    raise ValueError("directions must be a numpy array of floats or integers")
                elif value.ndim != 2 or value.shape[1] != self.crystal_structure.primitive_vecs.shape[0]:
                    raise ValueError(f"directions must be a 2D array with shape (M, {self.crystal_structure.primitive_vecs.shape[0]}), "+
                                     "where M is the number of directions")
                elif hasattr(self, "basis_directions") and self.basis_directions == "primitive_vector":
                    if not np.issubdtype(value.dtype, np.integer):
                        raise ValueError("directions must be an integer numpy array " +
                                         "when basis_directions is 'primitive_vector'")
                else:
                    self.__dict__[name] = value

            case "basis_directions":
                # Ensure it's one of the allowed values
                if value not in ("global_orthonormal", "primitive_vector"):
                    raise ValueError(f"Invalid basis_directions: {value} "+"\n"+
                                     "Allowed values are 'global_orthonormal' or 'primitive_vector'")
                elif hasattr(self, "directions") and np.issubdtype(self.directions.dtype, np.floating):
                    if value == "primitive_vector":
                        raise ValueError("basis_directions cannot be 'primitive_vector' when " +
                                         "directions is an array of floats")
                else:
                    self.__dict__[name] = value

            case "limit_denom":
                # Ensure it's a positive integer
                if not isinstance(value, int) or value <= 0:
                    raise ValueError("limit_denom must be a positive integer")
                else:
                    self.__dict__[name] = value
            case "verbose":
                # Ensure it's a boolean
                if not isinstance(value, bool):
                    raise ValueError("verbose must be a boolean")
                else:
                    self.__dict__[name] = value
            case _:
                # Default behavior for other attributes
                self.__dict__[name] = value


