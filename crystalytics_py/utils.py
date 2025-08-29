import numpy as np
from scipy import optimize

def shortest_collinear_vector_with_integer_components(v, max_length=50):
    if len(v.shape) == 1:
        v = v[None,:]
    elif len(v.shape) == 2:
        pass
    else:
        raise ValueError("Input vector v must be 1D or 2D array.")
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
    res = optimize.minimize(func1, x0=a_min)
    if res.success:
        v_new = np.round(res.x*u.T).astype(int).T
        v_new = (v_new.T/np.gcd.reduce(v_new, axis=1)).T
        return v_new, 0.5*(1-np.sum(u*v_new, axis=1)/np.linalg.norm(v, axis=1))
    else:
        raise RuntimeError(f"Optimization failed with message: \n"+res.message)  