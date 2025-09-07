import numpy as np
from itertools import product
import warnings

# -------------------------
# Gauss reduction (M=2 case)
# -------------------------
def gauss_reduce(B):
    """
    Gauss reduction for 2D lattice basis.
    Input:
      B : (2,N) array, rows are generator vectors
    Returns:
      B_reduced : (2,N) array, reduced basis
      U : (2,2) unimodular integer matrix, such that B_reduced = U @ B
    """

    assert isinstance(B, np.ndarray) and len(B.shape)==2, "Input array B must be a 2D numpy array"
    assert B.shape[0] == 2, "This Gauss reduction function applies ONLY to 2d lattices"
    B = B.astype(float)
    U = np.eye(2, dtype=int)

    a = B[0].copy()
    b = B[1].copy()

    while True:
        if np.dot(b, b) < np.dot(a, a):
            a, b = b, a
            U[[0, 1]] = U[[1, 0]]   # swap rows of U
        mu = int(round(np.dot(a, b) / np.dot(a, a)))
        if mu == 0:
            break
        b = b - mu * a
        U[1] = U[1] - mu * U[0]

    B_reduced = np.array([a, b])
    return B_reduced, U


# -------------------------
# Minkowski enumeration (small M)
# -------------------------
def minkowski_reduce(B, search_radius=4):
    """
    Minkowski-style reduction by brute-force enumeration (row-wise).
    Input:
      B : (M, N) ndarray  -- row-wise lattice generators (M rows, ambient dim N)
      search_radius : int -- max absolute coefficient to enumerate (inclusive)
    Returns:
      B_reduced : (M, N) ndarray -- reduced row-wise generators (rows are vectors)
      U         : (M, M) ndarray (int) -- unimodular integer matrix such that B_reduced = U @ B
    Notes:
      - This is exact enumeration. Practical for M==2 or 3 with small search_radius.
      - If the routine fails to find M independent vectors, increase search_radius.
    """
    B = np.asarray(B, dtype=float)
    M, N = B.shape
    if M > N:
        raise ValueError("Number of generators M must be <= ambient dimension N.")

    coeffs = range(-search_radius, search_radius + 1)
    candidates = []

    # enumerate integer coefficient tuples k (length M) and compute v = k @ B (row)
    for k in product(coeffs, repeat=M):
        if all(ki == 0 for ki in k):
            continue
        k_row = np.array(k, dtype=int)           # shape (M,)
        v_row = k_row @ B                        # shape (N,)  (row vector)
        candidates.append((np.linalg.norm(v_row), k_row, v_row))

    # sort by length (ascending)
    candidates.sort(key=lambda x: x[0])

    chosen_rows = []   # list of row vectors (each shape (N,))
    U_rows = []        # list of integer rows (each shape (M,)) so that U @ B gives chosen_rows

    for _, k_row, v_row in candidates:
        if len(chosen_rows) == M:
            break
        if not chosen_rows:
            # no prior vectors -> accept the first shortest
            independent = True
        else:
            trial = np.array(chosen_rows + [v_row])   # shape (len+1, N)
            # independent if rank increases
            independent = np.linalg.matrix_rank(trial) > len(chosen_rows)
        if independent:
            chosen_rows.append(v_row)
            U_rows.append(k_row)

    if len(chosen_rows) < M:
        raise RuntimeError(
            "Failed to find M independent short vectors; increase search_radius."
        )

    # stack rows into arrays (row-wise)
    B_reduced = np.array(chosen_rows, dtype=float)   # (M, N)
    U = np.array(U_rows, dtype=int)                  # (M, M)

    # sanity check: B_reduced should equal U @ B (within numerical tolerance)
    recon = U @ B
    if not np.allclose(B_reduced, recon, atol=1e-10, rtol=1e-12):
        # this is usually numerical: issue a warning rather than error
        warnings.warn(
            "Numerical discrepancy: B_reduced != U @ B within tolerance. "
            "Max abs diff = {:.3e}".format(np.max(np.abs(B_reduced - recon)))
        )

    return B_reduced, U



# -------------------------
# Build orthonormal basis Q
# -------------------------
def build_Q_from_B(B_reduced, tol=1e-12):
    """
    Build orthonormal basis Q (N,N).
    The M rows of B_reduced define a subspace; 
    Q's first M columns span this vector subspace.
    """
    M, N = B_reduced.shape
    cols = []

    # Gram-Schmidt on transposed rows (so we work with column vectors in ambient space)
    for j in range(M):
        v = B_reduced[j].astype(float).copy()
        for u in cols:
            v -= np.dot(u, v) * u
        nrm = np.linalg.norm(v)
        if nrm > tol:
            cols.append(v / nrm)

    # fill to N with standard basis
    for i in range(N):
        if len(cols) == N:
            break
        e = np.zeros(N); e[i] = 1
        v = e.copy()
        for u in cols:
            v -= np.dot(u, v) * u
        nrm = np.linalg.norm(v)
        if nrm > tol:
            cols.append(v / nrm)

    Q = np.column_stack(cols[:N])  # (N,N)
    Q, _ = np.linalg.qr(Q)         # ensure orthonormal
    return Q


# -------------------------
# Top-level routine
# -------------------------
def reduce_and_transform_lattice_vecs(B, method="auto", search_radius=4):
    """
    Full pipeline, row convention everywhere.
    Input:
      B : (M,N) ndarray, rows are lattice generators
    Returns:
      B_reduced : (M,N) reduced generators
      B_coords  : (M,M) same generators expressed in intrinsic orthonormal basis
      Q         : (N,N) orthonormal basis matrix
      U         : (M,M) unimodular integer transform, s.t. B_reduced = U @ B
    """
    M, N = B.shape
    if M > N:
        raise ValueError("Need the number of lattice generator vectors to be less "+
                         "than equal to the dimension of the vector space")

    # reduce
    if method == "auto":
        if M == 2:
            B_reduced, U = gauss_reduce_rows(B)
        else:
            B_reduced, U = minkowski_reduce_rows(B, search_radius)
    elif method == "gauss":
        B_reduced, U = gauss_reduce_rows(B)
    elif method == "minkowski":
        B_reduced, U = minkowski_reduce_rows(B, search_radius)
    else:
        raise ValueError("Unknown method")

    # orthonormal basis
    Q = build_Q_from_Brows(B_reduced)

    # intrinsic coords: project rows of B_reduced into new basis
    padded_coords = B_reduced @ Q    # (M,N)
    B_coords = padded_coords[:, :M]  # (M,M)

    # sanity check
    if N > M:
        trailing = np.linalg.norm(padded_coords[:, M:])
        if trailing > 1e-8:
            warnings.warn(f"Trailing components not ~0: {trailing:.2e}")

    return B_reduced, B_coords, Q, U


# -------------------------
# Converters
# -------------------------
def ambient_to_intrinsic(v, Q, M):
    """Row vector v (1xN) -> intrinsic coords (1xM)."""
    v_new = v @ Q
    return v_new[:M]

def intrinsic_to_ambient(v, Q):
    """Intrinsic coords (1xM) -> ambient row (1xN)."""
    N = len(Q); M = v.shape[1]
    padded = np.zeros(N)
    padded[:M] = v
    return padded @ Q.T
