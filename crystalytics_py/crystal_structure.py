import numpy as np
from .utils import *
import warnings

class CrystalStructure():
   
   """ 
   Crystal structure comprises of a lattice with motifs at each lattice site.
   A lattice is an infinite set of sites spatially so distributed such that 
   it satistfies translational periodicity. Translational periodicity can be
   understood as follows:
   Suppose we consider a vector joining two lattice sites (a lattice vector).
   Now if we translate any lattice site by this vector it will always reach
   another lattice site. And this condition remains valid for all lattice vectors.
   Now to describe a lattice we need three(for 3D)/two(for 2D) linearly independent
   lattice vectors which will form the edges of a minimum-volume(3D)/area(2D) 
   parallelopiped(3D)/parallelogram(2D) (primitive cell). This three lattice vectors 
   are called primitive vectors. The choice of primitive vectors is not unique, 
   however linear integer combination of any set of primitive vectors will construct 
   uniquely the same lattice. 
   Motifs are atoms or groups of atoms at the lattice site to complete the crystal 
   structure. The location of the atoms in the motif is expressed in fractional
   coordinates of the primitive lattice vectors.
   Therefore, if the chosen set of primitive vectors is (v1, v2, v3) and the set of
   j motifs be ((a1, b1, c1), (a2, b2, c2), ..., (aj, bj, cj)) where a, b, c are 
   fractional coordinates (0, 1] w.r.t. v1, v2, v3 respectively, then the set of 
   atomic sites A of the corresponding crystal structure is as follows:
   A = {{(I1+a1)*v1 + (I2+b1)*v2 + (I3+c1)*v3 : I1, I2, I3 are integers} U
        {(I1+a2)*v1 + (I2+b2)*v2 + (I3+c2)*v3 : I1, I2, I3 are integers} U
        .
        .
        .
        {(I1+aj)*v1 + (I2+bj)*v2 + (I3+cj)*v3 : I1, I2, I3 are integers}}

   This class deals will computing crystallographic quantities for any given crystal 
   structure described by a set of primitive lattice vectors and motifs. 

   Attributes:

   ***Member variables***
   primitive_vecs: Set of N primitive lattice vectors for N dimensional space
                   The primitive vector component are w.r.t. orthonormal basis
                   (Note: The methods of this class or any subclasses
                          are not gauranteed to work correctly for dimensions 
                          other than 2D and 3D)
                   type = numpy ndarray (N, N)
                          N = spatial dimensions                          
                   dtype = float
                   values = Any real number
                            Practically, it is better to factor out the scaling factor.
                            For example, for fcc:
                            [[0.5, 0.5, 0.0],
                             [0.0, 0.5, 0.5],
                             [0.5, 0.0, 0.5]] instead of lattice constant times the same.
   motifs: Set of M motifs' fractional coordinates w.r.t. the N primitive lattice vectors
           type = numpy ndarray (M, N)
                  M = number of motifs
                  If M = 1, then self.motif is [[0., 0., ... N times]]
           dtype = float
           values = in range [0, 1)

   ***Member methods***
   primitiveVecsnMotifsCheck:
         Checks whether the primitive vectors
   latticeSpacing: This method calculates the lattice spacing along any direction.
   latticeInterplanarSpacing: This method calculates the lattice interplanar spacing 
                              along any direction.
   latticeStackDisplacements:
   atomicInterplanarSpacing:
   atomicStackDisplacements:
   checkInputs: This method checks the validity of common 
                inputs of most of the methods of this class. 
   """

   def __init__(self, primitive_vecs, motifs=None, motif_types=None):
      """
      Constructor:

      To construct an instance of this class we need a set of primitive
      lattice vectors and fracional coordinates of the motifs w.r.t. those
      primitive lattice vectors.
      """
      self._primitive_vecs = primitive_vecs
      self._motifs = motifs
      self._motif_types = motif_types

   @property
   def primitive_vectors(self):
      return self._primitive_vecs
    
   @property
   def motifs(self):
      return self._motifs
   
   @property
   def motif_types(self):
      return self._motif_types


   def find_lattice_points_in_superlattice(self, 
                                           superlattice_generator_vector_directions,
                                           reference_basis = "global_orthonormal"):

      from .direction import Direction

      if len(superlattice_generator_vector_directions) != len(self._primitive_vecs):
         raise ValueError("Number of superlattice generator vectors must be "+
                          f"{len(self._primitive_vecs)}")

      D = Direction(self, superlattice_generator_vector_directions,
                    basis_directions=reference_basis)
      D._compute_lattice_spacing()

      gram_matrix = np.dot(self._primitive_vecs, 
                           self._primitive_vecs.T)

      if np.isclose(np.linalg.det(D.convert_primitive_to_orthonormal_basis(
                                    D.shortest_lattice_vectors)), 0.0):
         raise ValueError("The superlattice generator vectors are not "+
                          "linearly independent")
      
      M = abs(np.linalg.det(D.shortest_lattice_vectors))

      if M >=1 and np.isclose(M, np.round(M)):
         M = int(M)
      else:
         raise RuntimeError("The number of lattice points in the superlattice must "+
                            f"be an integer more than or equal to 1. But M = {M}. "+
                            "Bug Alert !!!")
      
      ranges = [np.arange(0, M)]*len(self._primitive_vecs)
      mesh = np.meshgrid(*ranges, indexing="ij")
      V = np.stack([m.ravel() for m in mesh], axis=-1)  # (M^2, N)

      W = np.dot(D.shortest_lattice_vectors.T, V.T).T/M
      W = W[np.all(np.isclose(W, np.round(W)), axis=1)]

      return W, D.shortest_lattice_vectors
   
   def get_superlattice_crystal_structure(self, 
                                          superlattice_generator_vector_directions,
                                          reference_basis = "global_orthonormal"):
      
      W, superlattice_generator_vectors = self.find_lattice_points_in_superlattice(
                                          superlattice_generator_vector_directions,
                                          reference_basis=reference_basis)
      superlattice_generator_vectors = np.dot(superlattice_generator_vectors, 
                                              self._primitive_vecs)
      
      new_motifs = W[:, None, :] + self._motifs[None, :, :]
      new_motifs = new_motifs.transpose(1, 0, 2).reshape(-1, W.shape[1])
      new_motifs = np.dot(new_motifs, self._primitive_vecs)
      new_motifs = np.dot(np.linalg.inv(superlattice_generator_vectors.T), 
                          new_motifs.T).T
      self._motif_types
      new_motif_types = np.repeat(self._motif_types, len(W)).tolist()

      return CrystalStructure(superlattice_generator_vectors,
                              new_motifs,
                              new_motif_types)

   ############################################################

   #####################
   ##### Methods #######
   #####################


   def latticeSpacing(self, directions, basis_directions = 'global_orthonormal',
                      limit_denom = 100, verbose=False):

      """
      This method calculates the lattice spacing along any direction.
      It is the length of the smallest lattice vector along that direction. 
      
      Inputs:
      
      directions: The directions along which the lattice spacing is requested.
                  type = numpy ndarray (M, N)
                         N = number of dimensions (same as self.primitive_vecs)
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
                                the primitive vectors.
      limit_denom: To express a floating point number as fraction of numerator and 
                   denominator, limit_denom limits the largest value of denominator
                   possible
                   type = int
                   value = default 100
      verbose: Permission to print messages during running of the method.
               type = bool
               value = default False


      Output: 

      1D numpy array (M,) of floats with lattice spacings corresponding to each requested
      direction
        
      
      Theory: Estimation of lattice spacing

      Let v1, v2, v3, ..., vN be N primitive lattice vectors for a N-D lattice. By defination,
      the vector L = (a1*v1) + (a2*v2) + ... + (aN*vN) would always join two lattice points
      if a_i are integers (negative, positive or zero). So we must choose a set {a} of 
      integers such that they the relatively prime (meaning, gcd(a',b',c') = 1) and the vector 
      L({a}) points towards the direction is which the lattice spacing 
      is requested. The magnitude of lattice vector L({a}) is the desired lattice spacing.
      
      """
      # Checking the validity of the inputs --- direction, basis_directions
      # and limit_denom
      directions = self.checkInputs(directions, basis_directions, limit_denom, verbose)

      if basis_directions == 'global_orthonormal':
         # Expressing the components of the desired directions in terms of the primitive lattice vectors
         dir_primitive = np.linalg.solve(self.primitive_vecs.T, directions.T).T
      else:
         dir_primitive = directions

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
      # however if there are square root elements in the primitive lattice vectors or directions, they might be decimals
      # but close to the desired integer values ---> So no significant problem is envisaged in calculation of lattice spacing

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
   
   ############################################################

   def latticeInterplanarSpacingnDisplacements(self, directions, basis_directions = 'global_orthonormal',
                                 limit_denom = 100, return_relative_displacements=True, 
                                 basis_relative_displacements = 'global_orthonormal',
                                 return_multiplicity=False, return_lattice_spacing=False, verbose=False):

      """
      This method primarily calculates the lattice interplanar spacing along any 
      direction by estimating the multiplicity. It also optionally calculates the 
      smallest relative displacements among lattice planes within one lattice spacing 
      along the requested direction.
      There may be more than one plane of lattice sites within one lattice spacing
      along any direction; the number of such planes is called multiplicity. Due to
      translational symmetry of the lattice, all the planes will be equally spaced 
      along that direction; therefore the lattice interplanar spacing is simply 
      the lattice spacing divided by the multiplicity. 
      
      Inputs:
      
      directions: The directions along which the lattice interplanar spacing are requested.
                  type = numpy ndarray (M, N)
                         N = number of dimensions (same as self.primitive_vecs)
                         M = number of requested directions
                  dtype = float

      ***Optional inputs***

      basis_direction: The basis vectors w.r.t. which the components of the
                       requested directions are provided.
                       type = str
                       value = 'global_orthonormal' (default): The primitive lattice 
                                vectors and the desired direction must have their 
                                components defined w.r.t. same orthonormal basis.
                                               or
                               'primitive_vector': The primitive lattice vectors 
                                must have their components defined w.r.t. orthonormal 
                                basis and the desired directions are defined w.r.t. 
                                the primitive vectors.
      limit_denom: To express a floating point number as fraction of numerator and 
                   denominator, limit_denom limits the largest value of denominator
                   possible
                   type = int
                   value = default 100
      return_relative_displacements: Permission to estimate and return the smallest relative 
                                     displacement vector(s) of the lattice planes w.r.t. any 
                                     one of them within one lattice spacing along the requested
                                     direction(s).
                                     type: bool
                                     value = default True
      basis_relative_displacements: The basis vectors w.r.t. which the components of the
                       relative displacement of lattice planes are outputted.
                       type = str
                       value = 'global_orthonormal' (default): The components are defined 
                                w.r.t. the same orthonormal basis as the primitive vectors.
                                               or
                               'primitive_vector': The primitive lattice vectors 
                                are the basis for defining the components of the relative 
                                displacements.
      return_multiplicity: Permission to output multiplicity
                           type = bool
                           value = default True
      return_lattice_spacing: Permission to output lattice spacing which is estimated for
                              calculating lattice interplanar spacing.
                              type = bool
                              value = default True
      verbose: Permission to print messages during running of the method.
               type = bool
               value = default False


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
      
      We multiply Ineq 1 by I = lcm of denominators of p_i / gcd of numerators of p_i
      This changes Ineq 1 to 
      0 < m1*p1_I + m2*p2_I + ... + mN*pN_I <= mul -------> Ineq 2
      p1_I, p2_I, ..., pN_I are integers relatively prime to each other and in same proportion as p1, p2, ..., pN.
      mul is also an integer and happens to be the number of stacking planes (the multiplicity) since
      it equals the number of possible values of (m1*p1_I + m2*p2_I + ... + mN*pN_I) in the inteval (0, mul].
      (This stems from the number theory theorem which states:
       "For any positive integers a and b, there exist integers x and y such that ax + by = gcd(a, b). 
        Furthermore, as x and y vary over all integers, ax + by attains all multiples and only multiples 
        of gcd(a, b)." In our case, p1_I, p2_I,... are the a,b and their gcd is 1. So basically the expression in 
        Ineq 2 will evaluate to multiples of 1, i.e. all integers depending on m_i chosen and mul is the multiplicity
        since its defines the upper limit via Ineq 2).

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

      # Checking the validity of the inputs --- direction, basis_directions
      # and limit_denom
      directions = self.checkInputs(directions, basis_directions, limit_denom, verbose)

      lattice_spacings = self.latticeSpacing(directions, basis_directions=basis_directions,
                      limit_denom=limit_denom, verbose=verbose)[0]

      if basis_directions == 'global_orthonormal':
         # Expressing the components of the desired directions in terms of the primitive lattice vectors
         dir_primitive = np.linalg.solve(self.primitive_vecs.T, directions.T).T
      else:
         dir_primitive = directions
      
      # Magnitude of dir_primitive(s) 
      mag_dir_primitive = np.linalg.norm(np.dot(dir_primitive, self.primitive_vecs), axis=1)
      # Lattice vectors in requested directions
      dir_lattice_vecs = (lattice_spacings/mag_dir_primitive*dir_primitive.T).T
      dir_lattice_vecs = np.round(dir_lattice_vecs).astype(int)
      

      # Metric tensor for the skew primitive lattice vectors
      metric_tensor = np.dot(self.primitive_vecs, self.primitive_vecs.T)
      
      # Creating numpy ufuncs of functions of the fractions class
      frac_array = np.frompyfunc(lambda x: fractions.Fraction(x).limit_denominator(limit_denom), 1, 1)
      num_array = np.frompyfunc(lambda x: x.numerator, 1, 1)
      den_array = np.frompyfunc(lambda x: x.denominator, 1, 1)
      
      # Estimating p1, p2, ... pN
      coeff = np.dot(metric_tensor, dir_lattice_vecs.T).T
      # Using fractions.Fractions on coeff might result in weird fractions, so we must perform this step first.
      gcd_est = abs(np.array([misc.apply_func_stepwise(x, sm.gcd_general) for x in coeff]))
      if any(np.isclose(gcd_est, 0)):
         raise RuntimeError('Components p = {p1, p2, ..., pN} for direction = {0}'.format(
                             directions[np.isclose(gcd_est, 0)]) +
                            'possibly has irrational components. p{0}'.format(
                             coeff[np.isclose(gcd_est, 0)]))
      else:
         coeff = (coeff.T/gcd_est).T
      coeff_frac = frac_array(coeff)
      coeff_num = num_array(coeff_frac).astype(int)
      coeff_den = den_array(coeff_frac).astype(int)
      coeff_den_lcm = np.lcm.reduce(abs(coeff_den), axis=1)
      coeff_num_gcd = np.gcd.reduce(abs(coeff_num), axis=1)
      mul = (lattice_spacings**2 * coeff_den_lcm.astype(float) 
                          / coeff_num_gcd.astype(float) / gcd_est ) 
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

   ############################################################
   
   def relativeDisplacementsLatticePlanes(self, directions, basis_directions = 'global_orthonormal',
                                     limit_denom=100, verbose=False):
      
      # Checking the validity of the inputs --- direction, basis_directions
      # and limit_denom
      directions = self.checkInputs(directions, basis_directions, limit_denom, verbose)
      
      multiplicity, spacing = self.latticeInterplanarSpacingnDisplacements(
                                   directions, return_relative_displacements=False,
                                   return_multiplicity=True, return_lattice_spacing=True)[-2:]
      
      dir_prim_vecs = self.latticeSpacing(directions)[-1]
      
      ratio_dir_prim_mul = (dir_prim_vecs.T/multiplicity).T
      floor_vals_diff = np.floor(ratio_dir_prim_mul) - ratio_dir_prim_mul  
      ceil_vals_diff = np.ceil(ratio_dir_prim_mul) - ratio_dir_prim_mul
   
      # Metric tensor for the primitive lattice vectors
      metric_tensor = np.dot(self.primitive_vecs, self.primitive_vecs.T)
   
      relative_displacements = []
      for i in np.arange(len(directions)):   
         rems = np.array(list(itertools.product(*zip(floor_vals_diff[i], ceil_vals_diff[i]))))
         vecs = np.dot(self.primitive_vecs.T, rems.T).T
         dot_vecs_dir = np.dot(vecs, np.dot(self.primitive_vecs.T, dir_prim_vecs[i]))
         disp_vecs = vecs[np.isclose(dot_vecs_dir, 0.0)]
         if len(disp_vecs) != 0:
            disp_vecs = np.unique(np.round(disp_vecs, decimals=8), axis=0)
            relative_displacements.append(disp_vecs)
            continue
         else:
            x,y,z = sympy.symbols('x y z')   
            t_0_vals = np.array([-3,-2,-1,0,1,2,3])
            # coefficients of the expr = np.sum(np.outer([x,y,z], dir_prim_vecs[i])*metric_tensor)
            # for x, y and z
            coeffs0 = np.sum(dir_prim_vecs[i]*metric_tensor, axis=1)
            if any(np.isclose(coeffs0, 0.0)):
               print('Coefficients = {0}'.format(coeffs0))
               raise RuntimeError('None of coefficients can be zero for the time being')
            coeff_orders = np.array(list(itertools.permutations([0,1,2])))
            disp_vecs = []
            for j in np.arange(len(vecs)):
               int_coeffs = get_miller_indices(np.r_[coeffs0, dot_vecs_dir[j]])
               coeffs = int_coeffs[:-1]; const = int_coeffs[-1]
               res_list = [Tuple(*np.array(diop_linear(np.dot([x,y,z], coeffs[c_ord])+const))[np.argsort(c_ord)])
                           for c_ord in coeff_orders]
               free_syms = list(set.union(*[res_val[count].free_symbols for res_val in res_list for count in [0,1,2]]))
               free_syms.sort(key=str)
               ints = np.array([res.subs(list(zip(free_syms, count))) for res in res_list 
                                for count in itertools.permutations(t_0_vals,2)]).astype(int)
               disp_vecs.append(vecs[j]+np.dot(self.primitive_vecs.T, ints.T).T)
            disp_vecs = np.vstack(disp_vecs)
            if not np.allclose(np.dot(disp_vecs, directions[i]), 0.0):
               print('dot product of relative displacements of plane in direction {0} = {1}'.format(
                      directions[i], np.dot(disp_vecs, directions[i])))
               print('Relative displacement vectors in direction {0}:\n {1}'.format(directions[i], disp_vecs))
               raise RuntimeError('Relative displacement vectors are not perpendicular to the direction {0}'.format(directions[i]))
            else:
               norm_disp_vecs = np.linalg.norm(disp_vecs, axis=1)
               disp_vecs = disp_vecs[np.isclose(norm_disp_vecs, np.min(norm_disp_vecs))]
               disp_vecs = np.unique(np.round(disp_vecs, decimals=8), axis=0)
               relative_displacements.append(disp_vecs)
               
      return relative_displacements
   
   
   ############################################################
   
   def get_subLattice_primitive_vecs_routine1(self, directions, basis_directions = 'global_orthonormal',
                                     limit_denom=100, verbose=False):
                                     
      ################ There are two routines of this method which are coded slightly differently 
      ################ but both of them give the correct solution(s). Since primitive vectors are not unique, 
      ################ the solution of this code by the two different routines might be different, but both solutions are correct!
                                     
      #Lattice vectors in the desired directions in terms of the primitive vectors
      dir_lattice_vecs = self.latticeSpacing(directions, basis_directions= basis_directions,
                                             limit_denom=limit_denom, verbose=verbose)[1]
                                             
      interplanar_spacings = self.latticeInterplanarSpacingnDisplacements(directions, return_relative_displacements=False, 
                             basis_directions= basis_directions, limit_denom=limit_denom, verbose=verbose)
      lat_vol = abs(np.dot(self.primitive_vecs[0], np.cross(self.primitive_vecs[1], self.primitive_vecs[2])))
      sublat_area = lat_vol/interplanar_spacings
      if verbose:
         print('lat_vol={0}'.format(lat_vol))
         print('sublat_area={0}'.format(sublat_area))
   
      coeff = np.dot(np.dot(self.primitive_vecs, self.primitive_vecs.T), dir_lattice_vecs.T).T
      
      # Creating numpy ufuncs of functions of the fractions class
      frac_array = np.frompyfunc(lambda x: fractions.Fraction(x).limit_denominator(limit_denom), 1, 1)
      num_array = np.frompyfunc(lambda x: x.numerator, 1, 1)
      den_array = np.frompyfunc(lambda x: x.denominator, 1, 1)
      
      # Creating fractions of every component of coeff and 
      # extracting the corresponding numerators and denominators
      coeff_frac = frac_array(coeff)
      coeff_num = num_array(coeff_frac).astype(int)
      coeff_den = den_array(coeff_frac).astype(int)
      
      x, y, z = sympy.symbols(['x', 'y', 'z'])
      
      primitive_vecs_sublattice_3D = []
      primitive_vecs_sublattice_2D = []
      
      for i in np.arange(len(coeff)):
         
         int_coeffs = np.array([ coeff_num[i,0]*coeff_den[i,1]*coeff_den[i,2], 
                                 coeff_num[i,1]*coeff_den[i,0]*coeff_den[i,2],
                                 coeff_num[i,2]*coeff_den[i,0]*coeff_den[i,1] ])
         expr = (int_coeffs[0]*x) + (int_coeffs[1]*y) + (int_coeffs[2]*z)
         res = diop_linear(expr)
         if any(np.array(res) == None):
            print("Result of Diophantine Eq: {0}".format(res))
            raise RuntimeError('Solution not found for the input direction {0}!'.format(directions[i]))
         if len(res) == 1:
            p_vars = list(res[0].free_symbols); p_vars.sort(key=str)
         elif len(res) == 2:
            p_vars = list(set.union(res[0].free_symbols, res[1].free_symbols)); p_vars.sort(key=str)
         elif len(res) == 3:
            p_vars = list(set.union(res[0].free_symbols, res[1].free_symbols, res[2].free_symbols))
            p_vars.sort(key=str)
         else:
            raise RuntimeError('Problem.. Please check!!!')    
         
         if np.any(int_coeffs == 0):
            if np.all(int_coeffs == 0):
               raise RuntimeError('The input direction must be a zero vector')
            res_list = list(res)
            for j in np.where(int_coeffs == 0)[0]:
               res_list.insert(j, 1)
            res = Tuple(*res_list)
               
         
         t0t1_arr = np.array([(1,0), (0,1), (-1,0), (0,-1), (1,1),
                              (-1,-1), (2,0), (0,2), (-2,0), (0,-2),
                              (2,1), (-2,1), (2,-1), (-2,-1),
                              (1,2), (-1,2), (1,-2), (-1,-2), (2,2), (-2,-2)])
           
         flag=0                
         for t0t1_1, t0t1_2 in itertools.combinations(t0t1_arr,2):
            ints1 = np.array(res.subs(list(zip(p_vars, t0t1_1)))).astype(int)
            ints2 = np.array(res.subs(list(zip(p_vars, t0t1_2)))).astype(int)
            tmp_vec1 = np.dot(self.primitive_vecs.T,ints1)
            tmp_vec2 = np.dot(self.primitive_vecs.T,ints2) 
            if np.allclose(np.linalg.norm(np.cross(tmp_vec1, tmp_vec2)), sublat_area[i]):
               if flag == 0:
                  cos_12 = np.dot(tmp_vec1, tmp_vec2)/np.linalg.norm(tmp_vec1)/np.linalg.norm(tmp_vec2)
                  vec1 = tmp_vec1; vec2 = tmp_vec2
               else:
                  cos_12_tmp = np.dot(tmp_vec1, tmp_vec2)/np.linalg.norm(tmp_vec1)/np.linalg.norm(tmp_vec2)
                  if np.isclose(abs(cos_12_tmp), abs(cos_12)):
                     if cos_12_tmp > cos_12:
                        cos_12 = cos_12_tmp
                        vec1 = tmp_vec1; vec2 = tmp_vec2
                  elif abs(cos_12_tmp) < abs(cos_12):
                     cos_12 = cos_12_tmp
                     vec1 = tmp_vec1; vec2 = tmp_vec2
               flag = flag + 1
            else:
               continue
            
         
         if flag == 0:
            raise RuntimeError('Primitive vectors not found for input direction {0}! '.format(directions[i]) +
                               'Increase the span of the array t0t1_arr')
         
         
         primitive_vecs_sublattice_3D.append( np.c_[vec1, vec2].T )  #w.r.t orthonormal basis, the same which defined self.primitive_vecs
         vec1_norm = np.linalg.norm(vec1); vec2_norm = np.linalg.norm(vec2)
         cos_theta = np.dot(vec1, vec2)/ vec1_norm/ vec2_norm
         vec1_2D = np.array([1,0.]); vec2_2D = np.array([cos_theta, np.sqrt(1-(cos_theta**2))])
         vec1_2D = vec1_2D*vec1_norm; vec2_2D = vec2_2D*vec2_norm
         primitive_vecs_sublattice_2D.append( np.c_[vec1_2D, vec2_2D].T )  #w.r.t 2D orthonormal basis, where one of the primitive vectors is
                                                                     # oriented along the x axis
      
      primitive_vecs_sublattice_3D = np.stack(primitive_vecs_sublattice_3D)
      primitive_vecs_sublattice_2D = np.stack(primitive_vecs_sublattice_2D)  
      
      return primitive_vecs_sublattice_3D, primitive_vecs_sublattice_2D
   
 
   ############################################################
   
   def get_subLattice_primitive_vecs_routine2(self, directions, basis_directions = 'global_orthonormal',
                                     limit_denom=100, verbose=False):
                                     
      ################ There are two routines of this method which are coded slightly differently 
      ################ but both of them give the correct solution(s). Since primitive vectors are not unique, 
      ################ the solution of this code by the two different routines might be different, but both solutions are correct!
                                     
                                     
      #Lattice vectors in the desired directions in terms of the primitive vectors
      dir_lattice_vecs = self.latticeSpacing(directions, basis_directions= basis_directions,
                                             limit_denom=limit_denom, verbose=verbose)[1]
                                             
      interplanar_spacings = self.latticeInterplanarSpacingnDisplacements(directions, return_relative_displacements=False, 
                             basis_directions= basis_directions, limit_denom=limit_denom, verbose=verbose)
      lat_vol = abs(np.dot(self.primitive_vecs[0], np.cross(self.primitive_vecs[1], self.primitive_vecs[2])))
      sublat_area = lat_vol/interplanar_spacings
      if verbose:
         print('lat_vol={0}'.format(lat_vol))
         print('sublat_area={0}'.format(sublat_area))
      
   
      coeff = np.dot(np.dot(self.primitive_vecs, self.primitive_vecs.T), dir_lattice_vecs.T).T
      
      # Creating numpy ufuncs of functions of the fractions class
      frac_array = np.frompyfunc(lambda x: fractions.Fraction(x).limit_denominator(limit_denom), 1, 1)
      num_array = np.frompyfunc(lambda x: x.numerator, 1, 1)
      den_array = np.frompyfunc(lambda x: x.denominator, 1, 1)
      
      # Creating fractions of every component of coeff and 
      # extracting the corresponding numerators and denominators
      coeff_frac = frac_array(coeff)
      coeff_num = num_array(coeff_frac).astype(int)
      coeff_den = den_array(coeff_frac).astype(int)
      
      x, y, z = sympy.symbols(['x', 'y', 'z'])
      
      primitive_vecs_sublattice_3D = []
      primitive_vecs_sublattice_2D = []
      
      
      # Looping over the different input directions
      for i in np.arange(len(coeff)):
      
         flag=0        
         int_coeffs = np.array([ coeff_num[i,0]*coeff_den[i,1]*coeff_den[i,2], 
                                 coeff_num[i,1]*coeff_den[i,0]*coeff_den[i,2],
                                 coeff_num[i,2]*coeff_den[i,0]*coeff_den[i,1] ])
                                 
         if np.count_nonzero(int_coeffs) == 0:
            raise RuntimeError('All the primitive vectors for the 3D lattice are perpendicular '+
                               'to the given direction {0}, which is weird '.format(directions[i])+
                               'since that would mean the 3 primitive vectors are coplanar')
         elif np.count_nonzero(int_coeffs) == 1:
            # Two of the primitive vectors for the 3D lattice are perpendicular to the given direction
            # and therefore will also be the primitive vectors for the 2D sublattice
            vec1, vec2 = self.primitive_vecs[int_coeffs==0]
            if not np.allclose(np.linalg.norm(np.cross(vec1, vec2)), sublat_area[i]):
               print('Vectors {0} and {1}'.format(vec1, vec2))
               print('cross-product magnitude = {0}'.format(np.linalg.norm(np.cross(vec1, vec2))))
               raise RuntimeError('The area of the parallelogram by the primitive vectors on the plane '+
                                  'must match sublat_area for direction {0}'.format(directions[i]))
            flag=1
         elif np.count_nonzero(int_coeffs) == 2:
            # One of the primitive vectors for the 3D lattice is perpendicular to the given direction
            # and therefore will also be the primitive vectors for the 2D sublattice
            vec1 = self.primitive_vecs[int_coeffs==0][0]
            expr = np.dot(int_coeffs, [x,y,z])
            res = diop_linear(expr)
            if any(np.array(res) == None):
               print("Result of Diophantine Eq: {0}".format(res))
               raise RuntimeError('Solution not found for the input direction {0}!'.format(directions[i]))
            p_vars = list(set.union(res[0].free_symbols, res[1].free_symbols)); p_vars.sort(key=str)
            if len(p_vars) !=1:
               raise RuntimeError('Number of free variables t* must be one for a linear equation of 2 variables '+
                                  'and all nonzero coefficients')
            for t0, t1 in itertools.combinations([-2, -1, 0, 1, 2]):
               ints = np.array(res.subs(list(zip(p_vars, [t0])))).astype(int)
               tmp_vec2 = np.dot(self.primitive_vecs[np.nonzero(int_coeffs)].T,ints)
               tmp_vec2 = tmp_vec2 + (t1*vec1)
               if np.allclose(np.linalg.norm(np.cross(vec1, tmp_vec2)), sublat_area[i]):
                  if flag == 0:
                     cos_12 = np.dot(vec1, tmp_vec2)/np.linalg.norm(vec1)/np.linalg.norm(tmp_vec2)
                     vec2 = tmp_vec2
                  else:
                     cos_12_tmp = np.dot(vec1, tmp_vec2)/np.linalg.norm(vec1)/np.linalg.norm(tmp_vec2)
                     if np.isclose(abs(cos_12_tmp), abs(cos_12)):
                        if cos_12_tmp > cos_12:
                           cos_12 = cos_12_tmp
                           vec2 = tmp_vec2
                     elif abs(cos_12_tmp) < abs(cos_12):
                        cos_12 = cos_12_tmp
                        vec2 = tmp_vec2
                  flag = flag + 1 
               else:
                  continue
                              
         else: 
            # None of the primitive vectors for the 3D lattice is perpendicular to the given direction
            coeff_order = np.array([[0,1,2], [0,2,1], [1,0,2], [2,0,1], [1,2,0], [2,1,0]])
            expr_list = [np.dot(int_coeffs[c_ord], [x,y,z]) for c_ord in coeff_order]
            res_list = [diop_linear(expr) for expr in expr_list]
            p_vars = list(set.union(res_list[0][0].free_symbols, res_list[0][1].free_symbols,
                                    res_list[0][2].free_symbols))
            p_vars.sort(key=str)
            if len(p_vars) !=2:
               raise RuntimeError('Number of free variables t* must be two for a linear equation of 3 variables '+
                                  'and all nonzero coefficients')
            
            t0t1_arr = np.array([(1,0), (0,1), (-1,0), (0,-1), (1,1),
                              (-1,-1), (2,0), (0,2), (-2,0), (0,-2),
                              (2,1), (-2,1), (2,-1), (-2,-1),
                              (1,2), (-1,2), (1,-2), (-1,-2), (2,2), (-2,-2)])
                              
            for t0t1_1, t0t1_2 in itertools.combinations(t0t1_arr,2):
               ints = np.stack( [np.array(res.subs(list(zip(p_vars, t0t1_1)))).astype(int)[np.argsort(coeff_order[iters])] for iters, res in enumerate(res_list)]
                              + [np.array(res.subs(list(zip(p_vars, t0t1_2)))).astype(int)[np.argsort(coeff_order[iters])] for iters, res in enumerate(res_list)] )
               vecs = np.dot(self.primitive_vecs.T,ints.T).T
               for tmp_vec1, tmp_vec2 in itertools.combinations(vecs, 2):
                  if np.allclose(np.linalg.norm(np.cross(tmp_vec1, tmp_vec2)), sublat_area[i]):
                     if flag == 0:
                        cos_12 = np.dot(tmp_vec1, tmp_vec2)/np.linalg.norm(tmp_vec1)/np.linalg.norm(tmp_vec2)
                        vec1 = tmp_vec1; vec2 = tmp_vec2
                     else:
                        cos_12_tmp = np.dot(tmp_vec1, tmp_vec2)/np.linalg.norm(tmp_vec1)/np.linalg.norm(tmp_vec2)
                        if np.isclose(abs(cos_12_tmp), abs(cos_12)):
                           if cos_12_tmp > cos_12:
                              cos_12 = cos_12_tmp
                              vec1 = tmp_vec1; vec2 = tmp_vec2
                        elif abs(cos_12_tmp) < abs(cos_12):
                           cos_12 = cos_12_tmp
                           vec1 = tmp_vec1; vec2 = tmp_vec2
                     flag = flag + 1
                  else:
                     continue
        
         
         if flag == 0:
            raise RuntimeError('Primitive vectors not found for input direction {0}! '.format(directions[i]) +
                               'Increase the span of the array t0t1_arr')
         
         
         primitive_vecs_sublattice_3D.append( np.c_[vec1, vec2].T )  #w.r.t orthonormal basis, the same which defined self.primitive_vecs
         vec1_norm = np.linalg.norm(vec1); vec2_norm = np.linalg.norm(vec2)
         cos_theta = np.dot(vec1, vec2)/ vec1_norm/ vec2_norm
         vec1_2D = np.array([1,0.]); vec2_2D = np.array([cos_theta, np.sqrt(1-(cos_theta**2))])
         vec1_2D = vec1_2D*vec1_norm; vec2_2D = vec2_2D*vec2_norm
         primitive_vecs_sublattice_2D.append( np.c_[vec1_2D, vec2_2D].T )  #w.r.t 2D orthonormal basis, where one of the primitive vectors is
                                                                     # oriented along the x axis
      
      primitive_vecs_sublattice_3D = np.stack(primitive_vecs_sublattice_3D)
      primitive_vecs_sublattice_2D = np.stack(primitive_vecs_sublattice_2D)  
      
      return primitive_vecs_sublattice_3D, primitive_vecs_sublattice_2D
   
   
   
   ############################################################

   def motifProjection(self, directions, basis_directions = 'global_orthonormal',
                       basis_project_vector = 'global_orthonormal'):

      """
      This method calculates the projection distances of motifs along the requested 
      direction(s) and also the projection vectors in the plane perpendicular to the 
      requested direction(s).
           
      Inputs:
      
      directions: The directions along which the motif projections are requested.
                  type = numpy ndarray (M, N)
                         N = number of dimensions (same as self.primitive_vecs)
                         M = number of requested directions
                  dtype = float

      ***Optional inputs***

      basis_direction: The basis vectors w.r.t. which the components of the
                       requested directions are provided.
                       type = str
                       value = 'global_orthonormal' (default): The primitive lattice 
                                vectors and the desired direction must have their 
                                components defined w.r.t. same orthonormal basis.
                                               or
                               'primitive_vector': The primitive lattice vectors 
                                must have their components defined w.r.t. orthonormal 
                                basis and the desired directions are defined w.r.t. 
                                the primitive vectors.
      
      basis_project_vector: The basis vectors w.r.t. which the components of the
                       motif projection vectors in the plane perpendicular to the 
                       requested direction(s) are outputted.
                       type = str
                       value = 'global_orthonormal' (default): The components are defined 
                                w.r.t. the same orthonormal basis as the primitive vectors.
                                               or
                               'primitive_vector': The primitive lattice vectors 
                                are the basis for defining the components of the relative 
                                displacements.
      

      Outputs: 

      2D numpy array (M,L) of floats with motif projections on each requested direction(s).
      M is the number of requested directions and L is the number of motifs. 

      3D numpy array (M,L,N) of floats with motif projection vectors along planes perpendicular
      to requested direction(s). N is the number of dimensions. 

      """

      # Checking input datatypes (and values)
      if not isinstance(directions, np.ndarray):
         raise TypeError('directions must be a numpy ndarray')
      if not isinstance(basis_directions, str):
         raise TypeError('basis_directions must be a string')
      if not np.in1d(basis_directions, ['global_orthonormal', 'primitive_vectors']):
         raise ValueError('basis_directions must be either "global_orthonormal" or '+
                          '"primitive_vectors"')
      if not isinstance(basis_project_vector, str):
         raise TypeError('basis_project_vector must be a string')
      if not np.in1d(basis_project_vector, ['global_orthonormal', 'primitive_vectors']):
         raise ValueError('basis_project_vector must be either "global_orthonormal" or '+
                          '"primitive_vectors"')
      # Checking shape of input matrices   
      if len(np.shape(directions)) == 1:
         directions = directions[None, :]
      elif len(np.shape(directions)) == 2:
         pass
      else:
         raise ValueError('directions must be either 1D or 2D numpy array')
      if np.shape(directions)[1] != np.shape(self.primitive_vecs)[0]:
         raise ValueError('directions must have the same number of dimension/columns ' + 
                          'as primitive_vecs')


      if basis_directions=='primitive_vectors':
         directions_orthonormal = np.dot(self.primitive_vecs.T, directions.T).T
      else:
         directions_orthonormal = directions
      motifs_orthonormal = np.dot(self.primitive_vecs.T, self.motifs.T).T
      projections = (np.dot(directions_orthonormal, motifs_orthonormal.T).T /
                     np.linalg.norm(directions_orthonormal, axis=1)).T # rows for directions and columns for motifs
      projection_vectors = (motifs_orthonormal[None,:] - (projections[:,:,None]*
                            ((directions_orthonormal.T/
                              np.linalg.norm(directions_orthonormal, axis=1)).T)[:,None,:]))
      if basis_project_vector == 'primitive_vectors':
         projection_vectors = np.array([np.linalg.solve(self.primitive_vecs.T, x.T).T
                              for x in projection_vectors])


      return projections, projection_vectors
      

   ############################################################
   
   def checkInputs(self, directions, basis_directions, limit_denom, verbose):

      """
      This method checks the validity of common inputs of most of the methods of this class. 
      """
      
      # Checking input datatypes (and values)
      if not isinstance(directions, np.ndarray):
         raise TypeError('directions must be a numpy ndarray')
      if not isinstance(basis_directions, str):
         raise TypeError('basis_directions must be a string')
      if not np.in1d(basis_directions, ['global_orthonormal', 'primitive_vectors']):
         raise ValueError('basis_directions must be either "global_orthonormal" or '+
                          '"primitive_vectors"')
      if not isinstance(limit_denom, int):
         raise TypeError('limit_denom must be an integer')
      if not isinstance(verbose, bool):
         raise TypeError('verbose must be a boolean')
      
      # Checking shape of input matrices   
      if len(np.shape(directions)) == 1:
         directions = directions[None, :]
      elif len(np.shape(directions)) == 2:
         pass
      else:
         raise ValueError('directions must be either 1D or 2D numpy array')
      if np.shape(directions)[1] != np.shape(self.primitive_vecs)[0]:
         raise ValueError('directions must have the same number of dimension/columns ' + 
                          'as primitive_vecs')

      return directions

   ############################################################
   
   def __setattr__(self, name, value):

      """motifs: fractional coordinates in the range [0, 1) defining the position of motifs w.r.t.
      primitive vectors. If only one motif is provided as 1D or 2D with one row or if motif = None,
      then motif is set to np.zeros(Ndims) where Ndims is the number of dimensions.
      If 2D then each row correspond to a different motif."""

      if (name == '_primitive_vecs'):
         if not isinstance(value, np.ndarray):
            raise TypeError('class CrystalStructure: The member "_primitive_vecs" must be a numpy ndarray')
         elif len(np.shape(value)) != 2:
            raise ValueError('class CrystalStructure: The member "_primitive_vecs" must be a 2D array with '+ 
                             'each row corresponding to a different lattice vector')
         elif np.shape(value)[0] != np.shape(value)[1]:
            raise ValueError('Number of primitive vectors must equal the number of dimensions. Therefore '+
                             'member "_primitive_vecs" must be a square matrix with number of rows/column '+
                             'equal to the number of dimensions')
         elif not inrange(np.shape(value)[0], 2, 3, ('closed', 'closed')):
            warnings.warn('class CrystalStructure: The routines in this class or subclasses does not gaurantee '+
                          'to work correctly for dimensions other than 2 or 3')
         else:
            if value.dtype == int:
               value = value.astype(float)               
            elif value.dtype != float:
               raise TypeError('The member "_primitive_vecs" must be of dtype float')
            # Checking whether the three lattice vectors are coplanar (which must not be).
            if np.isclose(np.linalg.det(value), 0.0):
               raise ValueError('class CrystalStructure: The lattice vectors are linearly dependent')
            else:
               self.__dict__[name] = value	 

      elif (name == '_motifs'):
         if value is None:
            self.__dict__[name] = np.zeros(np.shape(self._primitive_vecs)[0])[None, :]
            return
         elif not isinstance(value, np.ndarray):
            raise TypeError('class CrystalStructure: The member "_motifs" must be a numpy ndarray or None')
         else:
            if not np.all(inrange(value, 0, 1, ('closed', 'open'))):
               raise ValueError('class CrystalStructure: All fractional coordinates of every motif must be in range [0, 1)')
            elif (len(np.shape(value)) == 1) or ((len(np.shape(value)) == 2) and (np.shape(value)[0] == 1)): 
               self.__dict__[name] = np.zeros(np.shape(self._primitive_vecs)[0])[None, :]
               return
            elif (len(np.shape(value)) == 2) and (np.shape(value)[1] != np.shape(self._primitive_vecs)[0]):
               raise ValueError('class CrystalStructure: Dimension of primitive vectors and motifs does not match')
            elif value.dtype != float:
               raise TypeError('The member "_motifs" must be of dtype float')
            else:
               # so that no two or more motifs sit on each other
               value = np.unique(np.round(value,decimals=8), axis=0) # And we expect the difference in fractional 
                                                                     # coordinates is > 1e-8
               self.__dict__[name] = value
               if not all(np.isclose(value[0], 0.0)):
                  self.originShift4Motifs()
                  warnings.warn('The first motif was not at origin. So the motifs are translated '+
                                'so as to have the first motif at origin. The resulting set of '+
                                'motifs are {0}'.format(self._motifs))
                  return
               else:
                  return

      elif name == "_motif_types":
         self.__dict__[name] = value     

      else:
         raise NameError('class CrystalStructure: "' + name + '" is not a member of the this class. It must be ' +
                         'either "_primitive_vecs" or "_motifs" or "_motif_types".')

   ############################################################

   def originShift4Motifs(self):

      """ 
      For the set of motifs, where none of the motifs is at the origin (at a lattice site), 
      all the motifs are translated such that one motif is at the origin and the rest have
      positive fractional coordinates.
      """
      
      self._motifs -= self._motifs[0]
      self._motifs[self._motifs<(-2*np.finfo(float).eps)] += 1        
   

############################*******************###################################

class ReciprocalLattice(CrystalStructure):

   """Constructor"""
   def __init__(self, primitive_vecs, 
                defination = 'crystallographer'):
      CrystalStructure.__init__(self, primitive_vecs)
      self.defination = defination
      
      if np.shape(self.primitive_vecs)[0] != 3:
         raise ValueError('class ReciprocalLattice: To have reciprocal lattice, we need '+
                          'three primitive vectors in 3D')
      
      self.reciprocalLatticeVectors()

   ############################################################

   def reciprocalLatticeVectors(self):
      """Calculates the reciprocal lattice vectors"""

      self.real_space_primitive_vecs = self.primitive_vecs
      self.primitive_vecs = (np.array([
      np.cross(self.primitive_vecs[1], self.primitive_vecs[2]),
      np.cross(self.primitive_vecs[2], self.primitive_vecs[0]),
      np.cross(self.primitive_vecs[0], self.primitive_vecs[1])])/
      np.dot(self.primitive_vecs[0], np.cross(self.primitive_vecs[1], 
                                            self.primitive_vecs[2])))

      if self.defination == 'physicist':
         self.primitive_vecs *= 2*np.pi   


   ############################################################

   def latticeSpacing(self, directions, basis_directions = 'global_orthonormal',
                      limit_denom = 100, return_multiplicity = False, 
                      return_interplanar_spacing = False, verbose=False):

      res = self.latticeInterplanarSpacingnDisplacements(
         directions, basis_directions = basis_directions, limit_denom = limit_denom,
         return_multiplicity = return_multiplicity, return_lattice_spacing = return_interplanar_spacing,
         return_relative_displacements=False, verbose=verbose)

      res = list(res)
      res[0] = 1/res[0]
      if return_multiplicity and return_interplanar_spacing:
         res[2] = 1/res[2]
      elif return_interplanar_spacing:
         res[1] = 1/res[1]
      
      return tuple(res)

   
   ############################################################

   def latticeInterplanarSpacing(self, directions, basis_directions = 'global_orthonormal',
                      limit_denom = 100, verbose=False):

      res = self.latticeSpacing(directions, 
         basis_directions = basis_directions, limit_denom = limit_denom,
         verbose=verbose)

      return 1/res

   ############################################################

   
   def __setattr__(self, name, value):

      try:
         CrystalStructure.__setattr__(self, name, value)
      except NameError:
         pass
      
      if (name == 'defination'):
         if not isinstance(value, str):
            raise TypeError('class ReciprocalLattice: The member "defination" is a string')
         elif not np.in1d(value, ['crystallographer', 'physicist']).item():
            raise ValueError('class ReciprocalLattice: The member "defination" must be '+
                             'either "crystallographer" or "physicist". The latter differ by '+
                             'a multiplicative factor of 2*pi')
         else:
            self.__dict__[name] = value

      elif (name == 'reciprocal_lattice_vecs'):
         self.__dict__[name] = value
   


   
################################**********************################################################

class SymmetryOperations():

   def __init__(self, crys_struc):
      self.crys_struc = crys_struc

   ############################################################

   def __setattr__(self, name, value):

      if name == 'crys_struc':
         if not isinstance(value, CrystalStructure):
            raise ValueError('class SymmetryOperations: Member "crys_struc" must be an isntance of '+
                             'class CrystalStructure or its subclasses')
         else:
            self.__dict__[name] = copy.deepcopy(value)

   ############################################################

   def checkMirrorSymmetry(self, plane_normals, basis_directions='global_orthonormal', 
                           limit_denom=100, verbose=False):

      """
      This code checks for mirror symmetry across a plane passing through an atom. 
      The code takes the plane normals across which mirror symmetry
      needs to be examined as inputs.

      Theory:
   
      For a one-atom motif crystal structure, mirror symmetry exists across a plane
      passing through any atom/lattice site if and only if the multiplicity along
      the plane normal is 1 or 2.

      Proof:
   
      Let the primitive vectors be v1, v2, ..., vN. Lattice vector along the plane normal be
      N = (a1*v1) + (a2*v2) + ... + (aN*vN). Let any lattice vector be L = (m1*v1) + (m2*v2) + ... 
      + (mN*vN). Projection of L on N be P = I*k*n, where I is an integer, k is the lattice interplanar
      spacing along N and n is the unit vector along N. Since k = |L|/m (m is the multiplicity along N),
      P = I/m*N. Mirror image of a lattice site at L vector w.r.t. the site through which the plane passes,
      is given by vector L-2P = (m1 - 2I/m*a1)*v1 + (m2 - 2I/m*a2)*v2 + ... + (mN - 2I/m*aN)*vN. For L-2P
      to be a lattice point for all possible L, 2/m needs to be a integer which is possible if and only if
      m is 1 or 2.

      For more than one atom motif, the mirror symmetry along any plane (with multiplicity 1 or 2 along its
      normal) is preserved if and only if all vectors joining pairs of motif atoms are parallel to the plane
      being examined for mirror symmetry.

      """

      multiplicity = self.crys_struc.latticeInterplanarSpacingnDisplacements(plane_normals, 
                     basis_directions = basis_directions, limit_denom = limit_denom, 
                     return_relative_displacements=False, return_multiplicity=True, 
                     return_lattice_spacing=False, verbose=verbose)[1]
      
      # Checking whether multiplicity is 1 or 2
      mirror_symmetry_check = np.in1d(multiplicity, [1, 2])

      if np.shape(self.crys_struc.motifs)[0] != 1:
         # Checking whether the motif vectors are parallel to the requested plane
         projections = self.crys_struc.motifProjection(plane_normals, basis_directions = basis_directions)[0]
         mirror_symmetry_check[mirror_symmetry_check] = np.all(np.isclose(projections[mirror_symmetry_check], 0), axis=1) 

      return mirror_symmetry_check   


   ############################################################




 
################################**********************################################################


def findingNearestIntSoln2LinDiophantineEq(x, a, b):

   """
   Consider the equation (a1*x1) + (a2*x2) + ... + (aN*xN) = b --- Eqn 1
   where a1, a2, ..., aN are setwise coprime integers and b is 
   an integer.
   x1', x2', ..., xN' is a given real-number solution of Eqn 1. In this method,
   we are trying the integer solutions of Eqn 1. next to the given solution. By 
   next we mean solutions with integer values next to the real values in the 
   given solution in all dimensions (If any dimension of the given solution has
   integer value it is unchanged).

   Inputs:

   x: Given solution
      type = numpy ndarray (N,)
   a: Coefficients in Eqn 1 (setwise coprime integers)
      type = numpy ndarray (N,)
   b: right-hand side integer in Eqn 1
      type = int

   """

   # check inputs
   if not isinstance(x, np.ndarray):
      raise TypeError('The given solution "x" must be a numpy array')
   elif len(np.shape(x)) != 1:
      raise ValueError('The given solution "x" must be a 1D numpy array')
   elif len(x) < 2:
      raise ValueError('Number of dimensions must be at least 2')
   elif x.dtype == int:
      print("Needless to run this code since x was already an integer solution")
      return x
   elif x.dtype != float:
      raise ValueError('"x" must be a float type numpy array')
   elif all(np.isclose(x, np.round(x))):
      print("Needless to run this code since x was already an integer solution")
      return x.astype(int)
    

   if not isinstance(a, np.ndarray):
      raise TypeError('The coefficients of Eqn 1 "a" must be a numpy array')
   elif len(np.shape(a)) != 1:
      raise ValueError('The coefficients of Eqn 1 "a" must be a 1D numpy array')
   elif len(a) != len(x):
      raise ValueError('Number of dimensions of coefficients does not match'+
                       ' that of the given solution')
   elif a.dtype == float:
      if not all(np.isclose(a, np.round(a))):
         raise ValueError('Not all the coefficients are integers')
   elif a.dtype != int:
      raise ValueError('"a" must be a int type numpy array')
   elif abs(misc.apply_func_stepwise(a, sm.gcd_general)) != 1:
      raise ValueError('The coefficients are not setwise coprime')

   if not isinstance(b, np.integer):
      raise ValueError('The right-hand side of Eqn 1 "b" must be an integer')

   if not np.isclose(np.sum(x*a),b):
      raise ValueError('"x" is not a solution of Eqn 1')

   bool_int_x = np.isclose(x, np.round(x)) # boolean array with True for integer 
                                           # x values and False otherwise
   x[bool_int_x] = np.round(x)[bool_int_x]
   b_mod = b - np.sum(a[bool_int_x]*x[bool_int_x])
   bool_notint_x = np.invert(bool_int_x)
   notint_x = x[bool_notint_x]
   # coefficients corresponding to non-integer x 
   a2 = a[bool_notint_x]
   sign_a2 = np.sign(a2)
   a2 = abs(a2)
   min_a2 = min(a2)
   # sel first occurence of min_a2 in a2
   ind_min_a2 = np.nonzero(a2 == min_a2)[0][0]
   # a2 without first occurence of min_a2
   a2_rest = np.delete(a2, ind_min_a2)
   sign_a2_rest = np.delete(sign_a2, ind_min_a2)
   # x_rest without the entry at ind_min_a2
   notint_x_rest = np.delete(notint_x, ind_min_a2)
   floor_notint_x_rest = np.floor(notint_x_rest)
   # select a2_rest values which are multiples min_a2
   sel_multiple_a2 = np.isclose(a2_rest % min_a2, 0)
   
   #Create meshgrid of integers for dimensions of a2_rest whose values are not 
   # multiples of a2_min
   mesh_tup = np.meshgrid(*[np.arange(floor_notint_x_rest[i], floor_notint_x_rest[i]+2)
                 if sel_multiple_a2[i]
                 else np.arange(floor_notint_x_rest[i]-min_a2, floor_notint_x_rest[i]+min_a2+2)  
                 for i in np.arange(len(a2_rest))], indexing='ij', sparse=True)
   
   x_ind_min_a2 = 0
   for N in range(len(a2_rest)):
      x_ind_min_a2 = x_ind_min_a2 - (a2_rest[N]*sign_a2_rest[N]*mesh_tup[N])
   x_ind_min_a2 = x_ind_min_a2 + b_mod
   x_ind_min_a2 = x_ind_min_a2 / (sign_a2[ind_min_a2]*float(min_a2))
   
   # Select integers in x_ind_min_a2
   sel_ints = np.isclose(x_ind_min_a2, np.round(x_ind_min_a2))
   if not np.any(np.isclose):
      raise RuntimeError('No integers found in sel_ints. check!')
   else:
      sel_ints = np.nonzero(sel_ints)
      solns = np.zeros((len(sel_ints[0]), len(x)))
      solns[:,bool_int_x] = x[bool_int_x]
      inds_notint_x = np.nonzero(bool_notint_x)[0]
      inds_notint_x_rest = np.delete(inds_notint_x, ind_min_a2)
      for i in np.arange(len(mesh_tup)):
         solns[:, inds_notint_x_rest[i]] = mesh_tup[i].flatten()[sel_ints[i]]
      solns[:, inds_notint_x[ind_min_a2]] = x_ind_min_a2[sel_ints]

   if not np.all(np.isclose(np.sum(solns*a, axis=1), b)):
      raise RuntimeError('"solns does not satisfy the eqn ax=b"')

   return np.round(solns).astype(int)
         
  
################################**********************################################################     



def plot_lattice(primitive_vecs, extents):

   import matplotlib.pyplot as plt

   #extents is a (dim, 2) array which enumerates the range in each dimension
   #w.r.t to the primitive vectors. extent must be an integer array 

   if np.shape(primitive_vecs)[0] != np.shape(primitive_vecs)[1]:
      raise ValueError('Number of primitive vectors must equal the dimension')
   else:
      dim = len(primitive_vecs) 
      
   if extents.dtype != int:
      raise ValueError('extents must be an integer array')
      
   if np.shape(extents) != (dim, 2):
      raise ValueError('extents must be a (dim,2) array')
      
   mesh = np.meshgrid( *[ np.arange(ext[0], ext[1]) for ext in extents] )
   ints = np.array(list(zip( *[m.flatten() for m in mesh] )))
   points = np.dot(primitive_vecs.T, ints.T).T
   
   plt.scatter(*points.T)
   plt.gca().set_aspect('equal') 
   
   plt.show()
   
   
###############################################################################################################  
   

def get_miller_indices(directions, limit_denom = 100):

   if not isinstance(directions, np.ndarray):
      raise ValueError('"directions" must be a np.ndarray')
   elif (len(np.shape(directions)) > 2) or (len(np.shape(directions))==0):
      raise ValueError('"directions" must be either a 1D or a 2D array')
   elif len(np.shape(directions)) == 1:
      directions = directions[None,:]
   else:
      pass

   # Creating numpy ufuncs of functions of the fractions class
   frac_array = np.frompyfunc(lambda x: fractions.Fraction(x).limit_denominator(limit_denom), 1, 1)
   num_array = np.frompyfunc(lambda x: x.numerator, 1, 1)
   den_array = np.frompyfunc(lambda x: x.denominator, 1, 1)
   
   # Creating fractions of every component of directions and 
   # extracting the corresponding numerators and denominators
   dir_frac = frac_array(directions)
   dir_num = num_array(dir_frac).astype(int)
   dir_den = den_array(dir_frac).astype(int)
   
   miller_indices = np.c_[[directions[i]*np.lcm.reduce(dir_den[i])/np.gcd.reduce(dir_num[i])
                           for i in np.arange(len(directions))]] 
   
   if not np.allclose(miller_indices, np.round(miller_indices), rtol=0.0):
      warnings.warn('Miller indices not yet integers: {0}'.format(miller_indices))
   
   miller_indices = np.round(miller_indices).astype(int)
   
   if not np.all(np.gcd.reduce(miller_indices, axis=1) == 1):
      raise RuntimeError('Miller indices integers are not yet relatively prime')
   
   if np.shape(miller_indices)[0] == 1:
      miller_indices = miller_indices[0]
   
   return miller_indices

############################################################################################################### 





