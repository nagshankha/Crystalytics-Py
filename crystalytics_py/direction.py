from crystal_structure import CrystalStructure
import numpy as np
import fractions
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
    def __init__(self, crystal_structure, directions,
                 basis_directions = 'global_orthonormal',
                 limit_denom = 100, verbose=False):
        
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
        limit_denom: To express a floating point number as fraction of numerator and 
                    denominator, limit_denom limits the largest value of denominator
                    possible
                    type = int
                    value = default 100
        verbose: Permission to print messages during running of the method.
                type = bool
                value = default False
        """

        self.crystal_structure = crystal_structure
        self.directions = directions
        self.basis_directions = basis_directions
        self.limit_denom = limit_denom
        self.verbose = verbose

    def compute(self, compute_list='all'):
        pass

    def _compute_lattice_spacing(self):

        """
        This method calculates the lattice spacing along any direction.
        It is the length of the smallest lattice vector along that direction. 
        
        Output: 

        1D numpy array (M,) of floats with lattice spacings corresponding to each requested
        direction
            
        
        Theory: Estimation of lattice spacing

        Let v1, v2, v3, ..., vN be N primitive lattice vectors for a N-D lattice. By defination,
        the vector L = (a1*v1) + (a2*v2) + ... + (aN*vN) would always join two lattice points
        if a_i are integers (negative, positive or zero). So we must choose a set {a} of 
        integers such that they the relatively prime (meaning, gcd(a',b',c') = 1) and the vector 
        L({a}) points towards the direction is which the lattice spacing is requested. 
        The magnitude of lattice vector L({a}) is the desired lattice spacing.
        
        """

        if self.basis_directions == 'global_orthonormal':
            # Expressing the components of the desired directions in terms of the primitive lattice vectors
            dir_primitive = np.linalg.solve(self.crystal_structure.primitive_vecs.T, self.directions.T).T
        else:
            dir_primitive = self.directions

        # None of dir_primitive vectors must be a null vector
        if any(np.all(np.isclose(dir_primitive, 0.0), axis = 1)):
            raise RuntimeError('None of directions must be a null vector. ' +
                                'dir_primitive = {0}'.format(dir_primitive))

        dir_primitive[np.isclose(dir_primitive, 0)] = 0
    
        # To get relatively prime set of integers {a} such that the vector L({a}) points in the desired direction, 
        # we need to do the following.
        # If the desired direction in terms of primitive lattice vectors (as stored in dir_primitive) is (m,n,l,...)
        # then we calculate lcm/gcd = lcm of denominators of m,n,l.../ gcd of numerators of m,n,l,..., and multiply this 
        # with each of m,n,l,... and the result would be the desired relatively prime set of integers;
        # however if there are square root elements in the primitive lattice vectors or directions (in global coordinates), 
        # they might be decimals but close to the desired integer values ---> 
        # So no significant problem is envisaged in calculation of lattice spacing

        # Creating numpy ufuncs of functions of the fractions class
        frac_array = np.frompyfunc(lambda x: fractions.Fraction(x).limit_denominator(limit_denom), 1, 1)
        num_array = np.frompyfunc(lambda x: x.numerator, 1, 1)
        den_array = np.frompyfunc(lambda x: x.denominator, 1, 1)

        # Using fractions.Fractions on dir_primitive might result in weird fractions, so we must perform this step first.
        # This step should also give relatively prime {a} but we would still do the above procedure to double check!
        gcd_est = np.array([misc.apply_func_stepwise(x, sm.gcd_general) for x in dir_primitive])
        if any(np.isclose(gcd_est, 0)):
            raise RuntimeError('Components of following input direction(s) {0} w.r.t. primitive lattice vectors '.format(
                                directions[np.isclose(gcd_est, 0)]) +
                                'possibly has irrational components\ndir_primitive = {0}\n'.format(dir_primitive)+
                                'estimated gcd = {0}'.format(gcd_est))
        else:
            dir_primitive = (dir_primitive.T/gcd_est).T

        # Creating fractions of every component of dir_primitive (absolute values) and 
        # extracting the corresponding numerators and denominators
        dir_primitive_frac = frac_array(abs(dir_primitive))
        dir_primitive_num = num_array(dir_primitive_frac).astype(int)
        dir_primitive_den = den_array(dir_primitive_frac).astype(int)

        # Visual check if the fractional representation of the dir_primitive vectors are correcting.
        # Correct fractional representation is a crucial step for getting accurate result. If one
        # feels that the fractional representation is incorrect, one can play around with the limit_denom parameter.
        if verbose:
            print("Visual check for fractional form of the absolute value of desired "+
                "direction(s) (in terms of primitive lattice vectors)")
            print("{0} = {1}".format(dir_primitive, dir_primitive_frac))

        # Calculating the lcm of denominators and gcd of numerators
        dir_primitive_den_lcm = np.lcm.reduce(dir_primitive_den, axis=1)
        dir_primitive_num_gcd = np.gcd.reduce(dir_primitive_num, axis=1)
    
        # lattice vector(s) along desired direction(s)
        dir_lattice_vecs = (dir_primitive_den_lcm.astype(float) / dir_primitive_num_gcd.astype(float) *
                            dir_primitive.T).T

        if verbose:
            print('Lattice vectors in the desired directions in terms of primitive vectors')
            print(dir_lattice_vecs)

        # lattice spacing(s) along desired direction(s)
        lattice_spacings = np.linalg.norm(np.dot(self.primitive_vecs.T, dir_lattice_vecs.T), axis = 0)

        return lattice_spacings, dir_lattice_vecs


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


