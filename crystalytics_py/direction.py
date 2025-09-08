from crystal_structure import CrystalStructure
import numpy as np
from utils import *

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
        for element in self._compute_pipeline:
            self.__dict__["_compute_"+element]()

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
                               "lattice_spacing in the compute_list.")
    
    @property
    def shortest_lattice_vectors(self):
        if hasattr(self, '_shortest_lattice_vectors'):
            return self._shortest_lattice_vectors
        else:
            raise RuntimeError("shortest lattice vectors have not been computed yet. \n"+
                               "Please run the compute() method first with "+
                               "lattice_spacing in the compute_list.")

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
        Algorithm: Estimation of lattice interplanar spacing
        *************************************************

        Let L be the shortest lattice vector in the desired direction such that L = (a1*v1) + (a2*v2) + ... + (aN*vN)
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

        I = integer_coeff/coeff
        if not np.allclose(np.round(I).T-np.round(I)[:,0], 0.0):
            sel = np.invert(np.all(np.isclose(np.round(I).T-np.round(I)[:,0], 0.0), axis=0))
            raise RuntimeError(f"For directions:\n {self.directions[sel]} \n"+
                               f"the I is not unique for all dimensions \n {I[sel]}. \n"+
                               f"The cosine deviations for these directions are \n {cosine_deviations[sel]}")
        else:
            I = np.round(I)[:,0]

        multiplicity = I * self.lattice_spacings**2
        if np.allclose(multiplicity, np.round(multiplicity)):
            self._multiplicity = multiplicity.astype(int)
        else:
            sel = np.invert(np.isclose(multiplicity, np.round(multiplicity)))
            raise RuntimeError(f"For directions:\n {self.directions[sel]} \n"+
                               f"Multiplicity is not an integer: \n {multiplicity[sel]}")

        self._lattice_interplanar_spacings = self.lattice_spacings/self._multiplicity

        
    @property
    def lattice_interplanar_spacings(self):
        if hasattr(self, '_lattice_interplanar_spacings'):
            return self._lattice_interplanar_spacings
        else:
            raise RuntimeError("lattice interplanar spacings have not been computed yet. \n"+
                               "Please run the compute() method first with "+
                               "lattice_interplanar_spacing in the compute_list.")
    
    @property
    def multiplicity(self):
        if hasattr(self, '_multiplicity'):
            return self._multiplicity
        else:
            raise RuntimeError("multiplicity for the requested directions have not been computed yet. \n"+
                               "Please run the compute() method first with "+
                               "lattice_interplanar_spacing in the compute_list.")

    
    def _compute_relative_shift_of_lattice_planes(self):

        """
        Calculates the smallest relative shift vectors among consecutive lattice planes along each requested direction.

        ************************************************************************************************************
        Algorithm: Finding the smallest relative shift vector(s) of consecutive lattice planes
        ************************************************************************************************************

        Let A be the shortest lattice vector in the desired direction such that A = (a1*v1) + (a2*v2) + ... + (aN*vN)
        where a1, a2, ..., aN are integer coefficients of the linear combination of primitive lattice vectors 
        v1, v2, ..., vN. With mul being the multiplicity, A/mul is the shortest vector between two planes of 
        lattice sites in the desired direction.

        Let R = (r1*v1) + (r2*v2) + ... + (rN*vN) be the shortest relative shift vector(s) we are after. 
        One can easily deduce that (R + A/mul) is the shortest lattice vector between two consecutive lattice
        planes in the desired direction. This dictates {r_i + (a_i/mul)} to be setwise coprime integers 
        (Note: {r_i} and {a_i/mul} are floats).

        Let {c_i} be the set of {ceil(a_i/mul)} integers and {f_i} be the set of {floor(a_i/mul)} integer.
        Let D = [(d1_1, d2_1), (d1_2, d2_2), ..., (d1_i, d2_i), ..., (d1_N, d2_N)] be a list of N float 
        tuples; defined as d1_i = c_i-(a_i/mul) (hence positive, <1) and d2_i = f_i-(a_i/mul) (hence negative, >-1).

        Taking one fraction from each tuple in D, one can construction a N-dimensional vector which when added to 
        L/mul gives a vector joining lattice points across two consecutive lattice planes. There are 2^N such 
        vectors possible. Any of these vectors can be our desired R vector, that is, shortest relative shift vector. 
        We first need to enumerate this collection of 2^N vectors.
        (HERE ONE CAN EASILY SEE THAT THIS ALGORITHM HAS EXPONENTIAL COMPLEXITY WITH RESPECT TO DIMENSION N AND 
        IT IS MEANT FOR NO HIGHER THAN 3 DIMENSIONAL LATTICES)

        Each of these 2^N enumerated vectors serves as seed for generating more potential vector candidates for R as follows.

        Let's say one of these 2^N enumerated vector is (x1 v1 + x2 v2 + x3 v3 + ...).
        One can then generate infinitely many vectors from this seed vector as follows
        (I1+x1) v1 + (I2+x2) v2 + (I3+x3) v3 + ... where I1, I2, ... IN are integers.

        The magnitude of these vectors can then be written as 
        [x].T Q [x] + [I].T Q [I] + L [I]
        where [x] := [x1, x2, x3, ...].T
              [I] := [I1, I2, I3, ...].T
              Q := Gram matrix with the primitive vectors v1, v2, ...
              L := [2(x2 v1.v2 + x3 v1.v3 + ...), 2(x1 v1.v2 + v3 v2.v3 + ...), 2(x1 v1.v3 + x2 v2.v3 + ...), ...]

        The first two terms with Gram matrix Q are positive. And we must only consider those integers [I] which 
        results in negative ([I].T Q [I] + L [I]) since we are after the shortest relative shift vectors.

        These are all the integer vectors [I] for which vectors chol_mat[I] - chol_mat[I]* lie inside the Euclidean ball of radius 
        sqrt(0.25 L Q^-1 L.T), where Q = chol_mat.T chol_mat (Cholesky) and
        [I]* = -0.5 Q^-1 L.T

        So the enumeration is in two steps:
        1. Enumerate the 2^N vectors of fractions [x]
        2. For each of the above vector enumerate the admissible [I]

        Finally you will get a long list of vectors [x] + [I] ([I] depending on [x]), let's say x2
        From this list we remove the ones which when added to A/mul the resultant vectors are not setwise coprime.

        Out of the remaining vectors we search for the ones with the shortest length, that will be our collection of 
        shortest relative shift vector R

        """

        if np.shape(self.primitive_vectors)[1] > 3:
            print("Warning: This function is NOT recommended for dimension higher than 3, owing to exponential scaling")
            if input("Press 1 if you still want to continue, else press 0") == 0:
                return
            else:
                pass

        a = (self.shortest_lattice_vectors.T/self.multiplicity).T
        c = np.ceil(a); f = np.floor(a)
        d1 = c - a; d2 = f - a

        ### First we will enumerate the 2^N vectors
        
        # Generate all binary masks using division and modulus
        numbers = np.arange(2**N)[:, None]     # shape (2^N, 1)
        divisors = 2 ** np.arange(N)           # shape (N,)
        masks = (numbers // divisors) % 2      # shape (2^N, N)
        del number, divisors
        
        # Expand for broadcasting
        d1 = d1[:, None, :]                # (M, 1, N)
        d2 = d2[:, None, :]                # (M, 1, N)
        masks = masks[None, :, :]          # (1, 2^N, N)
        
        # Select from d1 or d2 depending on mask
        x = np.where(masks, d2, d1)  # (M, 2^N, N)
        del masks

        # Recalling the meaning of the dimension M and N
        # M := number of requested directions, that is, 
        #      len(self.directions)
        # N := Spatial dimension

        ### We will now enumerate the admissible integer vectors [I] 
        ### for each of the 2^N vectors enumerate above

        ## Gram matrix of primitive lattice vectors
        gram_matrix = np.dot(self.primitive_vecs, self.primitive_vecs.T)

        ## The L matrix
        G = gram_matrix.copy()
        np.fill_diagonal(G, 0)
        L = 2 * np.einsum('mjn,nk->mjk', x, G) 
        del G

        ## I_star
        # Precompute inverse of gram_matrix
        Ginv = np.linalg.inv(gram_matrix)
        I_star = -0.5 * np.einsum('mjn,nk->mjk', L, Ginv)

        ## Radius of the Eucleadian ball         
        # Use einsum to compute quadratic form: L @ Ginv @ L^T
        quad = np.einsum('mjn,nk,mjk->mj', L, Ginv, L)        
        # Final sqrt with factor 0.25
        Radius = np.sqrt(0.25 * quad)
        del quad, Ginv

        self._shortest_relative_shift_vectors = []
        n_directions = len(self.directions)
        eigs = np.linalg.eigvalsh(gram_matrix)
        # A sanity check
        if eigs[0] <= 0:
            raise RuntimeError("Gram matrix must be positive definite. \n"+
                               "Please check for possible bug in the code!")
        min_eig = eigs[0]

        # loop over the desired direction, that is, 
        # n_directions
        for i in range(n_directions):
            # zero relative shift if mutliplicity is 1
            if self.multiplicity[i]==1:
                self._shortest_relative_shift_vectors.append(
                    np.zeros((1,np.shape(self.directions)[1]))
                )
                continue

            centers = I_star[i]   # (P, N)
            shifts  = x[i]        # (P, N)
            r_vec   = radius[i]   # (P,)

            # bounding box for all j of this i
            max_disp = (r_vec / np.sqrt(min_eig)).max()
            low  = np.floor(centers - max_disp).min(axis=0).astype(int)
            high = np.ceil( centers + max_disp).max(axis=0).astype(int)

            # candidate integer grid for this i
            ranges = [np.arange(low[d], high[d] + 1) for d in range(N)]
            mesh = np.meshgrid(*ranges, indexing="ij")
            V = np.stack([m.ravel() for m in mesh], axis=-1)  # (K, N)

            # differences to all centers
            dif = V[None, :, :] - centers[:, None, :]         # (P, K, N)
            quad = np.einsum('pkn,nm,pkm->pk', dif, gram_matrix, dif)   # (P, K)
            # Notice to compute quad there is no need to 
            # explicitly perform Cholesky decomposition of gram_matrix

            # check condition
            mask = quad <= (r_vec[:, None] ** 2)              # (P, K)
            j_idx, k_idx = np.nonzero(mask)

            if j_idx.size == 0:
                raise RuntimeError('It is not possible to have no relative '+ 
                                   'shift vector between two consecutive '+
                                   'lattice plane when multiplicity is not 1. \n'+
                                   'Please check for possible bug in the code!')
            else:
                vecs = V[k_idx] + shifts[j_idx]
                vecs_int = vecs + (self.shortest_lattice_vectors[i]/self.multiplicity[i])
                if np.allclose(vecs_int, np.round(vecs_int)):
                    vecs_int = np.round(vecs_int).astype(int)
                    vecs_int = (vecs_int.T/np.gcd.reduce(vecs_int, axis=1)).T                    
                    _, vecs_inds = np.unique(vecs_int, return_index = True, axis=0)
                else:
                    raise RuntimeError("The relative shift vectors when added to "+
                                       "A/mul, A being the shortest lattice vector "+
                                       "in the desired direction, then the sum must be "+
                                       "an integer ---> which is not the case for "+
                                       f"direction {self.directions[i]}. \n" +
                                       "Please check for possible bug in the code!")
                self._shortest_relative_shift_vectors.append(vecs[vecs_inds])

        
    @property
    def shortest_relative_shift_vectors(self):
        if hasattr(self, '_shortest_relative_shift_vectors'):
            return self._shortest_relative_shift_vectors
        else:
            raise RuntimeError("shortest_relative_shift_vectors for lattice planes perpendicular "+
                               "to the requested directions have not been computed yet. \n"+
                               "Please run the compute() method first with "+
                               "relative_shift_of_lattice_planes in the compute_list.")


    def _compute_primitive_vectors_in_2d_sublattice(self):
        """
        Disclaimer: This function is ONLY applicable to 3D lattices.
        """

        from scipy.optimize import minimize_scalar

        if len(self.primitive_vectors) != 3:
            raise ValueError(
                f"This function is only applicable to 3D lattices, "
                f"but got {len(self.primitive_vectors)} primitive vectors."
            )

        # self.shortest_relative_shift_vectors*mul are lattice vectors in the 
        # 2d sublattice. Since they are all of the same length, let's choose 
        # the first one as a primitive vector for the 2D sublattice

        ## Gram matrix of primitive lattice vectors
        gram_matrix = np.dot(self.primitive_vecs, self.primitive_vecs.T)

        ## Transform vector from primitive vector basis to orthonormal basis
        def convert_primitive_to_orthonormal_basis(v):
            return np.sum(v*self.primitive_vectors.T, axis=1)

        ## Transform vector from orthonormal basis to primitive vector basis
        def convert_orthonormal_to_primitive_basis(v):
            return np.linalg.solve(self.primitive_vectors.T, v)

        self._primitive_vectors_in_2d_sublattice = []
        self._3d_to_2d_transformation_matrices = []

        for i in len(self.directions):

            ## Area of parallelogram enclosed by primitive vectors of 2d sublattice
            Area = (abs(np.linalg.det(self.primitive_vectors))/
                        self.lattice_interplanar_spacings[i])

            v1 = self.shortest_relative_shift_vectors[i][0]*self.multiplicity[i]

            # sanity check
            if np.allclose(v1, np.round(v1)):
                v1_orthonorm = convert_primitive_to_orthonormal_basis(v1)
                v1_unit_orthonorm = v1_orthonorm/np.linalg.norm(v1_orthonorm)
            else:
                raise RuntimeError("shortest relative shift vectors multiplied "+
                                   "by multiplicity are lattice vectors and "+
                                   "since they are expressed in terms of primitive "+
                                   "vectors their components needs to be integers, "+
                                   "which is not the case here. \n"+
                                   "Please check for possible bug in the code!")

            # The desired direction in orthonormal coordinates
            u_orthonorm = convert_primitive_to_orthonormal_basis(
                                    self.shortest_lattice_vectors[i])
            u_unit_orthonorm = u_orthonorm/np.linalg.norm(u_orthonorm)

            # Length of vector v1
            s1 = np.sum(np.outer(v1)*gram_matrix)

            # v2 generator
            def generate_v2(alpha):
                v2_unit = ((np.cos(alpha)*v1_unit_orthonorm) + 
                           (np.sin(alpha)*np.cross(u_unit_orthonorm, v1_unit_orthonorm)))
                # sanity check:
                assert np.isclose(np.linalg.norm(v2_unit), 1), "v2_unit must be a unit vector, check code for bug!"
                v2_prim = convert_orthonormal_to_primitive_basis(v2_unit)
                v2_prim_ints = shortest_collinear_vector_with_integer_components(v2_prim, 10)
                return v2_unit, v2_prim_ints

            # Loss function
            def loss(alpha):
                v2_prim_ints = generate_v2(alpha)[1]
                s2 = np.sum(np.outer(v2_prim_ints)*gram_matrix)
                return abs(Area - s1 * s2 * np.sin(alpha))

            # -----------------------------
            # Local optimization
            # -----------------------------
            res = minimize_scalar(
                loss,
                bounds=(1e-6, np.pi/2 + 1e-3),
                method="bounded",
                options={"xatol": 1e-6}
            )

            best_alpha = res.x
            best_loss = res.fun

            v2_unit, v2_prim_ints = generate_v2(best_alpha)
            v2_orthonorm = convert_primitive_to_orthonormal_basis(v2_prim_ints)
            v2_unit_orthonorm = v2_orthonorm/np.linalg.norm(v2_orthonorm)
            if np.isclose(best_loss, 0) and np.allclose(v2_unit, v2_unit_orthonorm):
                v2 = v2_prim_ints
            else:
                raise RuntimeError("Failed to find a primitive vector couple to v1")

            B = np.array([convert_primitive_to_orthonormal_basis(v1),
                          convert_primitive_to_orthonormal_basis(v2)])

            from lattice_utils import reduce_and_transform_lattice_vecs
            _, B_coords, Q, _ = reduce_and_transform_lattice_vecs(B)

            self._primitive_vectors_in_2d_sublattice.append(B_coords)
            self._3d_to_2d_transformation_matrices.append(Q)

        self._primitive_vectors_in_2d_sublattice = np.array(self._primitive_vectors_in_2d_sublattice)
        self._3d_to_2d_transformation_matrices = np.array(self._3d_to_2d_transformation_matrices)


    @property
    def primitive_vectors_in_2d_sublattice(self):
        if hasattr(self, '_primitive_vectors_in_2d_sublattice'):
            return self._primitive_vectors_in_2d_sublattice
        else:
            raise RuntimeError("primitive_vectors_in_2d_sublattice for lattice planes perpendicular "+
                               "to the requested directions have not been computed yet. \n"+
                               "Please run the compute() method first with "+
                               "primitive_vectors_in_2d_sublattice in the compute_list.")

    @property
    def transformation_matrices_3d_to_2d(self):
        if hasattr(self, '_3d_to_2d_transformation_matrices'):
            return self._3d_to_2d_transformation_matrices
        else:
            raise RuntimeError("Transformation matrices for expressing vectors with respect to "+
                               "orthonormal basis with two basis vectors in lattice plane perpendicular "+
                               "to the requested directions have not been computed yet. \n"+
                               "Please run the compute() method first with "+
                               "primitive_vectors_in_2d_sublattice in the compute_list.")








        








        


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


