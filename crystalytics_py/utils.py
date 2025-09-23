import numpy as np
from scipy import optimize

def shortest_collinear_vector_with_integer_components(v, max_length=50):

    """
    Finds the shortest integer vector(s) that are collinear with the given input vector(s).
    Basically, this function searches for setwise coprime integers forming vectors parallel to the 
    given input vectors.

    Parameters
    ----------
    v : np.ndarray
        Input vector(s). Can be a 1D array (single vector) or a 2D array (multiple vectors, shape (M, N)).
    max_length : int or float, optional
        Integer vectors with lengths less than max_length are searched for. Default is 50.

    Returns
    -------
    v_new : np.ndarray
        Shortest integer vector(s) collinear with the input vector(s).
    modified cosine deviation : np.ndarray
        Array of deviations from perfect collinearity for each vector.
        The deviation is defined as 0.5 * (1 - cos(theta)), where theta is the angle between
        the input vector and the found integer vector. A value of 0 indicates perfect collinearity.

    Raises
    ------
    ValueError
        If the input vector v is not a 1D or 2D array.
    RuntimeError
        If no suitable integer vector can be found or optimization fails 
        (the latter being unlikely and might indicate a bug in the code).

    Notes
    -----
    The function uses numerical optimization to find integer vectors that are collinear
    with the input vectors, minimizing the deviation from perfect collinearity.
    """

    if not isinstance(v, np.ndarray):
        raise ValueError("Input vector v must be a numpy array.")

    if len(v.shape) == 1:
        v = v[None,:]
    elif len(v.shape) == 2:
        pass
    else:
        raise ValueError("Input vector v must be 1D or 2D array.")
    
    if np.issubdtype(v.dtype, np.integer):
        v_new = (v.T/np.gcd.reduce(v, axis=1)).T
        return v_new.astype(int), np.zeros(len(v))

    u = (v.T/np.linalg.norm(v, axis=1)).T
    func1 = lambda x: np.max(abs((x*u.T)-np.round(x*u.T)))
    func2 = lambda x: np.max(abs((x[None,:,None]*u[:,None,:])-np.round(x[None,:,None]*u[:,None,:])), axis=-1)
    a = np.linspace(0, max_length, int(np.round(max_length/50*1000)))[1:]
    dev = func2(a)
    inds = np.nonzero(np.diff(np.sign(np.diff(dev, axis=1)), axis=1)>0)
    if np.all(np.unique(inds[0])==np.arange(len(u))):
        indices = np.cumsum(np.unique(inds[0], return_counts=True)[1])[:-1]  # drop last to avoid empty slice
        dev_min_inds_list = np.split(inds[1], indices)
    else:
        raise RuntimeError("Could not find integer vectors collinear with the following vectors: \n"+
                           f"{v[list(set(np.arange(len(u)))-set(inds[0]))]}")
    a_min = np.array([a[dev_min_inds_list[i][np.argmin(dev[i][dev_min_inds_list[i]])]] 
                      for i in range(len(u))])
    res = optimize.minimize(func1, x0=a_min, method='Nelder-Mead', tol=1e-6)
    if not res.success:      
       print(f"Warning: Optimization failed with message: \n"+res.message) 
    v_new = np.round(res.x*u.T).astype(int).T
    v_new = (v_new.T/np.gcd.reduce(v_new, axis=1)).T
    return v_new.astype(int), 0.5*(1-np.sum(u*v_new, axis=1)/np.linalg.norm(v, axis=1)) 
    

def inrange(vec, mini, maxi, bounds=('open', 'open')):

   if not hasattr(vec, '__iter__'):
      vec = np.array([vec])
   
   if not isinstance(vec, np.ndarray):
      vec = np.array(vec)
      
   if not (np.issubdtype(vec.dtype, np.integer) or np.issubdtype(vec.dtype, np.floating)):
      raise TypeError('Input must be either of type integers or floats')

   if not np.isnan(mini):
      if bounds[0] == 'open':
         sel1 = vec > mini
      elif bounds[0] == 'closed':
         sel1 = vec >= mini
      else:
         raise ValueError('elements of tuple bounds can take value either "open" or "closed"')
   else:
      sel1 = np.repeat(True, len(vec))

   if not np.isnan(maxi):
      if bounds[1] == 'open':
         sel2 = vec < maxi
      elif bounds[1] == 'closed':
         sel2 = vec <= maxi
      else:
         raise ValueError('elements of tuple bounds can take value either "open" or "closed"')
   else:
      sel2 = np.repeat(True, len(vec))

   return (sel1 & sel2)
