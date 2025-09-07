import numpy as np
from itertools import product

# -------------------------
# Low-level reducers (row-wise)
# -------------------------
def gauss_reduce_rows(B):
    """
    Gauss reduction for 2D lattice generators (rows).
    Input: B shape (2, N)
    Returns: B_reduced (2,N), coords (2,2), T_int (2,2)  where coords == T_int
    Note: T_int is the unimodular integer transform in the M-d lattice coordinates.
    """
    assert B.shape[0] == 2, "gauss_reduce_rows requires M==2"
    # Work with copies of row vectors (ambient coords)
    b1 = B[0].astype(float).copy()
    b2 = B[1].astype(float).copy()
    # unimodular integer transform in intrinsic coords
    T = np.eye(2, dtype=int)

    while True:
        if np.dot(b2, b2) < np.dot(b1, b1):
            b1, b2 = b2, b1
            T[[0,1]] = T[[1,0]]
        mu = int(round(np.dot(b1, b2) / np.dot(b1, b1)))
        if mu == 0:
            break
        b2 = b2 - mu * b1
        T[1] = T[1] - mu * T[0]

    B_reduced = np.vstack([b1, b2])
    B_coords = T.copy()
    return B_reduced, B_coords, T


def minkowski_reduce_rows(B, search_radius=4):
    """
    Minkowski-style reduction by brute-force enumeration (row-wise).
    Practical for small M (2 or 3) and small search_radius.
    Input: B shape (M, N)
    Returns: B_reduced (M, N), B_coords (M, M) where B_coords rows are integer
             coefficient rows w.r.t. original B rows (i.e., T).
    """
    M, N = B.shape
    coeffs = range(-search_radius, search_radius + 1)
    candidates = []

    # enumerate integer coefficient vectors k of length M (row combos k @ B)
    for k in product(coeffs, repeat=M):
        if all(ki == 0 for ki in k):
            continue
        k = np.array(k, dtype=int)
        v = k @ B   # row vector of shape (N,)
        candidates.append((np.linalg.norm(v), k, v))

    # sort by length in ambient space
    candidates.sort(key=lambda x: x[0])

    chosen_rows = []
    T_rows = []
    for _, k, v in candidates:
        if len(chosen_rows) == M:
            break
        # check linear independence among chosen row vectors
        trial = np.vstack(chosen_rows + [v]) if chosen_rows else v.reshape(1, -1)
        if np.linalg.matrix_rank(trial) > len(chosen_rows):
            chosen_rows.append(v)
            T_rows.append(k)

    if len(chosen_rows) < M:
        raise RuntimeError("Enumeration failed to find full-rank M vectors; increase search_radius")

    B_reduced = np.vstack(chosen_rows)    # (M, N)
    T = np.vstack(T_rows).astype(int)     # (M, M) integer matrix; rows are coefficients
    B_coords = T.copy()                   # intrinsic integer coordinates of reduced gens
    return B_reduced, B_coords, T


# -------------------------
# Helper: orthonormal basis build (T)
# -------------------------
def build_T_from_Breduced(B_reduced, normalize_first=True, tol=1e-12):
    """
    Build orthonormal N x N matrix T whose columns are the new orthonormal basis e'_j
    expressed in the original ambient basis, with the first column aligned with the
    first row of B_reduced (if normalize_first True). The first M columns span the
    subspace of the M row-generators.

    Input:
      B_reduced: (M, N) array (rows are generators in ambient basis)
    Returns:
      T: (N, N) orthonormal matrix (np.float64)
      padded_coords: (M, N) = B_reduced @ T   (should have zeros in columns M..N-1)
      B_coords: (M, M) = padded_coords[:, :M]
    """
    M, N = B_reduced.shape
    # Compute orthonormal basis of the row-space, with first vector aligned to first row
    cols = []  # will hold column vectors (each length N)

    # 1) first column = normalized first row (if possible)
    first = B_reduced[0].astype(float).copy()
    nrm = np.linalg.norm(first)
    if normalize_first and nrm > tol:
        e1 = first / nrm
        cols.append(e1)
    else:
        # fallback: pick largest row by norm
        norms = np.linalg.norm(B_reduced, axis=1)
        idx = int(np.argmax(norms))
        v = B_reduced[idx].astype(float)
        if np.linalg.norm(v) < tol:
            raise ValueError("B_reduced has near-zero rows")
        cols.append(v / np.linalg.norm(v))

    # 2) Gram-Schmidt on remaining rows to get up to M orthonormal columns
    for r in range(M):
        if len(cols) >= M:
            break
        if r == 0:
            continue
        v = B_reduced[r].astype(float).copy()
        # orthonormalize against existing cols
        for u in cols:
            v = v - np.dot(v, u) * u
        nm = np.linalg.norm(v)
        if nm > tol:
            cols.append(v / nm)

    # 3) If we still have fewer than M directions (rare), fill using SVD principal vectors
    if len(cols) < M:
        # take right singular vectors from SVD of B_reduced
        U, S, Vt = np.linalg.svd(B_reduced, full_matrices=False)
        # Vt is (min(M,N), N) but the right singular vectors are along Vt
        # take columns of V (rows of Vt) as candidates
        for row in Vt:
            cand = row.astype(float)
            # orthonormalize
            for u in cols:
                cand = cand - np.dot(cand, u) * u
            nm = np.linalg.norm(cand)
            if nm > tol:
                cols.append(cand / nm)
            if len(cols) >= M:
                break

    # 4) Now cols has M orthonormal vectors spanning the row-space; complete to N basis
    # Start with these M columns and extend
    Tcols = cols.copy()
    # Try standard basis vectors to complete
    for i in range(N):
        if len(Tcols) >= N:
            break
        e_std = np.zeros(N)
        e_std[i] = 1.0
        # orthonormalize
        w = e_std.copy()
        for u in Tcols:
            w = w - np.dot(w, u) * u
        nw = np.linalg.norm(w)
        if nw > tol:
            Tcols.append(w / nw)

    # If still not enough (very degenerate), use random vectors
    rng = np.random.default_rng(12345)
    while len(Tcols) < N:
        e_rand = rng.normal(size=N)
        w = e_rand.copy()
        for u in Tcols:
            w = w - np.dot(w, u) * u
        nw = np.linalg.norm(w)
        if nw > tol:
            Tcols.append(w / nw)

    # Create T (N x N) whose columns are the basis vectors
    T = np.column_stack(Tcols[:N])  # (N, N)
    # Ensure orthonormal (numerical)
    # re-orthonormalize via QR to be safe
    Q, R = np.linalg.qr(T)
    T = Q

    # compute padded coords and intrinsic coords
    padded_coords = B_reduced @ T        # shape (M, N)
    B_coords = padded_coords[:, :M].copy()  # (M, M)
    # Ideally padded_coords[:, M:] should be near zero (numerical)
    return T, padded_coords, B_coords


# -------------------------
# Main top-level reducer and transformer
# -------------------------
def reduce_and_transform(B, method="auto", search_radius=4):
    """
    Reduce row-wise lattice generators B (M,N) and build transform T and intrinsic coords.

    Returns:
       B_reduced : (M, N)  -- reduced generators in ambient basis
       B_coords  : (M, M)  -- reduced generators expressed in intrinsic orthonormal basis
       T         : (N, N)  -- orthonormal transform mapping ambient -> new basis columns
    """
    B = np.asarray(B, dtype=float)
    M, N = B.shape
    if M > N:
        raise ValueError("M (number of generators) must be <= N (ambient dim)")

    # 1) reduction in the row-wise lattice generating space
    if method == "auto":
        if M == 2:
            B_reduced, int_coords, T_int = gauss_reduce_rows(B)
        elif M <= 3:
            B_reduced, int_coords, T_int = minkowski_reduce_rows(B, search_radius=search_radius)
        else:
            # for M>3 recommend using external LLL package; fallback: use Minkowski enumeration (expensive!)
            B_reduced, int_coords, T_int = minkowski_reduce_rows(B, search_radius=search_radius)
    elif method == "gauss":
        if M != 2:
            raise ValueError("gauss method only for M==2")
        B_reduced, int_coords, T_int = gauss_reduce_rows(B)
    elif method == "minkowski":
        B_reduced, int_coords, T_int = minkowski_reduce_rows(B, search_radius=search_radius)
    else:
        raise ValueError("unknown method")

    # 2) Build orthonormal transform T (N x N) with first M columns spanning row-space
    T, padded_coords, B_coords = build_T_from_Breduced(B_reduced)

    # Small numerical cleanup: enforce near-zero trailing columns
    trailing_norm = np.linalg.norm(padded_coords[:, M:])
    if trailing_norm > 1e-8:
        # warn but continue
        # trailing components should be zero theoretically
        print(f"Warning: trailing components norm = {trailing_norm:.3e} (should be ~0).")

    return B_reduced, B_coords, T


# -------------------------
# Conversion helper functions
# -------------------------
def ambient_to_intrinsic(v_row, T, M):
    """
    Map ambient row-vector v_row (shape (N,)) to intrinsic coords (length M).
    Uses new basis columns in T (N x N). Returns first M components of v_row @ T.
    """
    v_row = np.asarray(v_row, dtype=float).ravel()
    N = T.shape[0]
    if v_row.size != N:
        raise ValueError("v_row length mismatch with T")
    coords_padded = v_row @ T   # shape (N,)
    return coords_padded[:M].copy()


def intrinsic_to_ambient(c_row, T, M, N):
    """
    Map intrinsic coordinates (length M) back to ambient row vector (length N).
    Pad with zeros and multiply by T^T (since T is orthonormal).
    """
    c_row = np.asarray(c_row, dtype=float).ravel()
    if c_row.size != M:
        raise ValueError("c_row length mismatch")
    padded = np.zeros(N, dtype=float)
    padded[:M] = c_row
    # B_reduced = padded @ T.T  (because earlier padded = B_reduced @ T)
    return padded @ T.T


# -------------------------
# Quick demo (if run as script)
# -------------------------
if __name__ == "__main__":
    # Example: M=2, N=4 ambient
    B = np.array([
        [2.0, 0.0, 1.0, 0.0],
        [0.5, 1.0, 0.0, 1.0]
    ])
    Breduced, Bcoords, T = reduce_and_transform(B, method="auto", search_radius=4)
    print("B_reduced (M,N):\n", Breduced)
    print("B_coords (M,M):\n", Bcoords)
    print("T (N,N):\n", T)
    # Test roundtrip for first generator:
    padded = np.hstack([Bcoords[0], np.zeros(T.shape[0] - Bcoords.shape[1])])
    rec = padded @ T.T
    print("Roundtrip ambient (reconstructed):\n", rec)
    print("Original reduced row 0:\n", Breduced[0])
    # Test ambient->intrinsic mapping
    x = Breduced[0]
    coords = ambient_to_intrinsic(x, T, M=2)
    print("Coords from ambient->intrinsic:", coords)
    # Test back
    back = intrinsic_to_ambient(coords, T, M=2, N=4)
    print("Back to ambient:", back)
