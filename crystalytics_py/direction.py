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

    # Maximum allowed length for searching shortest lattice vectors
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
            or 'primitive_vector'. If 'primitive_vector', the directions must be an integer array.
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
            - self._shortest_lattice_vectors : Shortest lattice vectors in those directions 
                                               expressed in primitive vector basis.
            - self._cosine_deviations : Deviation from perfect collinearity.
            - self._lattice_spacings : Computed lattice spacings.

        If self.verbose is set True, the method prints warnings if any direction 
        does not have a perfectly collinear lattice vector.
        """

        if self.basis_directions == 'global_orthonormal':
            # Expressing the components of the desired directions in terms of the primitive lattice vectors
            # This is saved in self._dir_primitive for debugging purposes
            self._dir_primitive = np.linalg.solve(self.primitive_vecs.T, self.directions.T).T
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

    def _compute_lattice_interplanar_spacing(self):

        """
        Calculates the lattice interplanar spacing along each requested direction. 
        There may be more than one plane of lattice sites within one lattice spacing
        along any direction; the number of such planes is called multiplicity. Due to
        translational symmetry of the lattice, all the planes will be equally spaced 
        along that direction; therefore the lattice interplanar spacing is simply 
        the lattice spacing divided by the multiplicity. 
        
        Outputs: 

        1D numpy array (M,) of floats with lattice interplanar spacings corresponding to each 
        requested direction

        list(len=M) of list(len=P-1) of 2D numpy array (L,N) of floats with smallest relative 
        displacements of lattice planes w.r.t. any one of them (only if return_relative_displacements=True). 
        M is the number of requested directions, P the number of lattice planes within one lattice 
        spacing in any direction (The corresponding list has P-1 entries since we are considering 
        relative displacements), L is the number of possible smallest relative displacement vectors 
        and N is the number of dimensions of the problem.
        
        1D numpy array (M,) of ints with multiplicity corresponding to each requested direction
        (only if return_multiplicity=True)

        1D numpy array (M,) of floats with lattice spacings corresponding to each requested
        direction (only if return_lattice_spacing=True)
            
        *************************************************
        Theory: Estimation of lattice interplanar spacing
        *************************************************

        Let L be the lattice vector in the desired direction such that L = (a1*v1) + (a2*v2) + ... + (aN*vN)
        where a1, a2, ..., aN are integer coefficients of the linear combination of primitive lattice vectors 
        v1, v2, ..., vN. Now the number of planes of lattice sites within one lattice spacing must equal the number 
        of possible values of L.((m1*v1) + (m2*v2) + ... + (mN*vN)) in the interval (0, lattice_spacing**2], 
        where '.' signifies dot product, and m_i are integers. The expression simplifies to:
        0 < m1*p1 + m2*p2 + ... + mN*pN <= lattice_spacing**2  --------> Ineq 1
        p1 = (a1*v1.v1) + (a2*v1.v2) + ... + (aN*v1.vN)
        p2 = (a1*v2.v1) + (a2*v2.v2) + ... + (aN*v2.vN)
        .
        .
        .
        pN = (a1*vN.v1) + (a2*vN.v2) + ... + (aN*vN.vN)
        
        We multiply Ineq 1 by a factor I such that we get
        0 < m1*p1_I + m2*p2_I + ... + mN*pN_I <= mul -------> Ineq 2
        p1_I, p2_I, ..., pN_I are setwise coprime integers and in same proportion as p1, p2, ..., pN.
        mul is also an integer and happens to be the number of stacking planes (the multiplicity) since
        it equals the number of possible values of (m1*p1_I + m2*p2_I + ... + mN*pN_I) in the inteval (0, mul].
        (This stems from the number theory theorem which states:
        "For any positive integers a and b, there exist integers x and y such that ax + by = gcd(a, b). 
            Furthermore, as x and y vary over all integers, ax + by attains all multiples and only multiples 
            of gcd(a, b)." In our case, p1_I, p2_I,... are the a,b and their gcd is 1. So basically the expression in 
            Ineq 2 will evaluate to multiples of 1, i.e. all integers depending on m_i chosen and mul is the multiplicity
            since its defines the upper limit via Ineq 2).

        So the task is to find the factor I, then mul equals I*lattice_spacing**2 and
        the lattice interplanar spacing (= lattice_spacing/mul) equals (I*lattice_spacing)**(-1)      

        """      

        # Gram matrix primitive lattice vectors
        gram_matrix = np.dot(self.primitive_vecs, self.primitive_vecs.T)
        
        # Estimating p1, p2, ... pN
        coeff = np.dot(gram_matrix, self.shortest_lattice_vectors.T).T

        integer_coeff, cosine_deviations = \
            shortest_collinear_vector_with_integer_components(coeff,
                                                              max_length=self._max_lattice_vector_length)

        I = coeff/integer_coeff
        if np.max(abs(I-np.round(I))) > 1e-6:
            sel = np.max(abs(I-np.round(I)), axis=1) > 1e-6
            raise RuntimeError(f"For directions:\n {self.directions[sel]} \n"+
                               f"cosine deviations of p vectors are \n {cosine_deviations[sel]}")
        elif not np.allclose(np.round(I).T-np.round(I)[:,0], 0.0):
            sel = np.invert(np.allclose(np.round(I).T-np.round(I)[:,0], 0.0, axis=0))
            raise RuntimeError(f"For directions:\n {self.directions[sel]} \n"+
                               f"the I is not unique for all dimensions \n {I[sel]}")
        else:
            I = np.round(I)[:,0]

        multiplicity = I * self.lattice_spacings**2
        if np.allclose(multiplicity, np.round(multiplicity)):
            self._multiplicity = multiplicity
        else:
            sel = np.invert(np.allclose(multiplicity, np.round(multiplicity)))
            raise RuntimeError(f"For directions:\n {self.directions[sel]} \n"+
                               f"Multiplicity is not an integer: \n {multiplicity[sel]}")

        self._lattice_interplanar_spacing = self.lattice_spacings/self._multiplicity

        mul = frac_array(mul)
        if any(den_array(mul).astype(int) != 1):
            raise RuntimeError('multiplicity must be an integer.. Check!!!')
        else:
            multiplicity = num_array(mul).astype(int)
    
        lattice_interplanar_spacings = lattice_spacings/multiplicity

        # Estimating smallest relative displacents of lattice planes w.r.t any one of them
        # if return_relative_displacements is True
        if return_relative_displacements:
            relative_displacements = [] # Initiating empty list
            for dir_count in np.arange(len(multiplicity)):
                # dir_count: index iterating over the requested directions
                # Function for evaluating expr "disp" for a particular requested direction
                disp_func = lambda x: (x - 
                            (np.dot(np.dot(self.primitive_vecs.T, x),
                                    np.dot(self.primitive_vecs.T, dir_lattice_vecs[dir_count]))/
                            lattice_spacings[dir_count]**2 * dir_lattice_vecs[dir_count]))
                relative_displacements.append([]) # appending an empty list to fill in entries for this
                                                # particular requested direction (index: dir_count)
                for p_count in np.arange(multiplicity[dir_count]-1):
                # p_count: index iterating over the lattice planes within one lattice spacing (except one)
                # along the particular requested direction (index: dir_count)
                coeff_norm = (coeff.T * coeff_den_lcm.astype(float) # coeff normalized by multiplying it's entries by the lcm of the 
                                / coeff_num_gcd.astype(float)).T      # denominators and dividing by the gcd of the numerators
                
                objective_func = lambda x: np.sum(np.outer(x,x)*metric_tensor) # Objective function for the minimization problem
                constraint_func = lambda x: np.sum(x*coeff_norm[dir_count])-p_count-1 # Constraint function equating to zero 
                                                                                        # for the minimization problem
                # Solving the optimization problem
                res = optimize.minimize(objective_func, np.zeros(len(self.primitive_vecs)),
                                    method='SLSQP', constraints = (
                                    {'type':'eq', 'fun': constraint_func}), options = {'ftol':1e-8})
                res.x[np.isclose(res.x, 0)] = 0
                # Since the objective function is convex the problem will have a unique minima (consider real variables)
                # However since our minimization problem needs the variable to be integer, there won't be a unique minima
                # unless the above minimization yield integers for all elements of the optimum vector.
                if np.allclose(res.x, np.round(res.x)):
                    # checks whether all elements of the optimal vector are integers
                    relative_displacements[dir_count].append(disp_func(res.x))
                else:
                    #**** Method 1 (worked correctly for fcc but might not work for some systems since it has some parameter that can be tuned!) ***
                    #check_int = np.isclose(res.x, np.round(res.x))
                    #res.x[check_int] = np.round(res.x)[check_int]
                    #print 'res.x', res.x 
                    #int_floor = np.floor(res.x)
                    #int_arr = np.array(list(itertools.product(*[np.arange(y, y+2) 
                    #          if np.invert(check_int)[i] else [y] for i,y 
                    #          in enumerate(int_floor)])))
                    #int_satisfy_contraint = np.array([np.isclose(constraint_func(y), 0) 
                    #                                  for y in int_arr])
                    #int_arr = int_arr[int_satisfy_contraint]
                    #**** Method 2 (should work for all cases) ***
                    int_arr = findingNearestIntSoln2LinDiophantineEq(res.x, coeff_norm[dir_count], p_count+1)
                    obj_values = np.array([objective_func(y) for y in int_arr])
                    int_arr = int_arr[np.isclose(obj_values, min(obj_values))]
                    relative_displacements[dir_count].append(
                            np.array([disp_func(y) for y in int_arr]))  
                if basis_relative_displacements == 'global_orthonormal':
                    relative_displacements[dir_count][p_count] = np.dot(
                    self.primitive_vecs.T, relative_displacements[dir_count][p_count].T).T
                relative_displacements[dir_count][p_count][np.isclose(
                                relative_displacements[dir_count][p_count], 0)] = 0

            return operator.itemgetter(*np.nonzero([True, True, return_multiplicity, return_lattice_spacing])[0])((
                            lattice_interplanar_spacings, relative_displacements, multiplicity, lattice_spacings))   

        else:
            return operator.itemgetter(*np.nonzero([True, return_multiplicity, return_lattice_spacing])[0])((
                            lattice_interplanar_spacings, multiplicity, lattice_spacings))  

    
    def _compute_relative_shift_of_lattice_planes(self):

        """
        Calculates the smallest relative shift vectors among consecutive lattice planes along each requested direction.

        ************************************************************************************************************
        Theory: Finding the smallest relative displacement vector(s) of lattice planes w.r.t. any one of them within 
                one lattice spacing along the requested direction
        ************************************************************************************************************

        For this we need to solve the following convex optimization problem with integer variables:
            
            Minimize sum_i sum_j m_i*m_j*(v_i.v_j) (where variable integers m_i and integer constants p_i_I are same as in Ineq 2)
            s.t.
                m1*p1_I + m2*p2_I + ... + mN*pN_I = 0 or 1 or ... (mul-1)
            (Depending on what you put on the right hand side of the constraint equation, 
            you get the displacement for that particular lattice plane) 

        Suppose one solution of its optimization is m1', m2', ..., mN'
        Then the corresponding displacement vector will be 
        (m1'*v1 + m2'*v2 + ... + mN'*vN) - ((L.(m1'*v1 + m2'*v2 + ... + mN'*vN)/|L|^2) L) -------> expr "disp"
        """

        pass


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


