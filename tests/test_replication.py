import numpy as np
import pytest

from crystalytics_py.crystal_structure import CrystalStructure


def _rotated_sc_supercell(sc):
    """
    Construct the smallest SC superlattice whose cell vectors are along

        [ 1  1  0]
        [-1  1  0]
        [ 0  0  1]

    The transformation from the primitive SC cell is therefore

        S = [[ 1,  1, 0],
             [-1,  1, 0],
             [ 0,  0, 1]]

    with |det(S)| = 2.
    """
    return sc.get_superlattice_crystal_structure(
        np.array(
            [
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )


def test_superlattice_preserves_explicit_parent(sc):
    child = _rotated_sc_supercell(sc)

    assert child.parent_crystal_structure is sc
    assert child.is_superlattice
    assert child.n_parent_lattice_points == 2

    np.testing.assert_array_equal(
        np.rint(child.parent_lattice_transform).astype(int),
        np.array(
            [
                [1, 1, 0],
                [-1, 1, 0],
                [0, 0, 1],
            ]
        ),
    )


def test_primitive_cell_replication_plane_intervals_are_one(sc):
    """
    Until automatic primitive-parent identification is implemented,
    a CrystalStructure without an explicit parent is treated as primitive.
    """
    np.testing.assert_array_equal(
        sc.replication_plane_intervals,
        [1, 1, 1],
    )


def test_rotated_sc_supercell_replication_plane_intervals(sc):
    """
    The rotated SC cell contains two lattice-plane intervals across
    each of its two in-plane cell vectors and one along [001].
    """
    child = _rotated_sc_supercell(sc)

    np.testing.assert_array_equal(
        child.replication_plane_intervals,
        [2, 2, 1],
    )


def test_full_replication_returns_positions_types_and_cell(sc):
    child = _rotated_sc_supercell(sc)

    positions, atom_types, cell = child.replicate(
        repetitions=[2, 1, 1]
    )

    expected_cell = child.primitive_vectors.copy()
    expected_cell[0] *= 2.0

    np.testing.assert_allclose(
        cell,
        expected_cell,
    )

    # The minimal rotated cell contains two SC lattice points.
    # Replicating it twice along the first cell vector gives four.
    assert len(positions) == 4
    assert len(atom_types) == 4

    # All generated atoms must lie inside the returned half-open
    # periodic cell.
    fractional_positions = np.linalg.solve(
        cell.T,
        positions.T,
    ).T

    assert np.all(
        fractional_positions >= -1e-12
    )

    assert np.all(
        fractional_positions < 1.0 + 1e-12
    )


def test_partial_replication_uses_plane_intervals(sc):
    """
    For the rotated SC cell,

        K = [2, 2, 1].

    Therefore p_0 = 1 corresponds to half of one complete current-cell
    replication along cell vector 0.
    """
    child = _rotated_sc_supercell(sc)

    positions, atom_types, cell = child.replicate(
        repetitions=[1, 1, 1],
        partial_plane_steps=[1, 0, 0],
    )

    expected_cell = child.primitive_vectors.copy()
    expected_cell[0] *= 1.5

    np.testing.assert_allclose(
        cell,
        expected_cell,
    )

    assert len(positions) == 3
    assert len(atom_types) == 3


def test_partial_replication_bounds_are_enforced(sc):
    child = _rotated_sc_supercell(sc)

    # K_0 = 2, therefore the valid values are only p_0 = 0 or 1.
    with pytest.raises(
        ValueError,
        match="replication_plane_intervals",
    ):
        child.replicate(
            repetitions=[1, 1, 1],
            partial_plane_steps=[2, 0, 0],
        )


def test_zero_full_repetitions_allowed_when_partial_extent_is_positive(sc):
    """
    A partial cell smaller than the smallest exact superlattice is allowed.

    With K_0 = 2,

        repetitions[0] = 0
        partial_plane_steps[0] = 1

    gives a scale factor of 1/2 along the first cell vector.
    """
    child = _rotated_sc_supercell(sc)

    positions, atom_types, cell = child.replicate(
        repetitions=[0, 1, 1],
        partial_plane_steps=[1, 0, 0],
    )

    expected_cell = child.primitive_vectors.copy()
    expected_cell[0] *= 0.5

    np.testing.assert_allclose(
        cell,
        expected_cell,
    )

    assert len(positions) == 1
    assert len(atom_types) == 1


def test_zero_extent_is_rejected(sc):
    with pytest.raises(
        ValueError,
        match="positive extent",
    ):
        sc.replicate(
            repetitions=[0, 1, 1],
            partial_plane_steps=[0, 0, 0],
        )


def test_multi_atom_parent_replication_is_deferred():
    """
    Stage 3 deliberately supports only the single-site parent case.
    """
    crystal = CrystalStructure(
        np.eye(3, dtype=float),
        motif_positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
            ],
            dtype=float,
        ),
        motif_types=[1, 2],
    )

    with pytest.raises(
        NotImplementedError,
        match="single-site parent motif",
    ):
        crystal.replicate(
            repetitions=[2, 2, 2]
        )

def _skew_sc_supercell(sc):
    """
    SC supercell with transformation

        S = [[2, 1, 0],
             [0, 1, 0],
             [0, 0, 1]]

    The cell is non-orthogonal and contains two parent lattice sites.
    """
    return sc.get_superlattice_crystal_structure(
        np.array(
            [
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )

def test_skew_sc_supercell_geometry(sc):
    child = _skew_sc_supercell(sc)

    expected_cell = np.array(
        [
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        child.primitive_vectors,
        expected_cell,
    )

    assert child.is_superlattice
    assert child.n_parent_lattice_points == 2

    np.testing.assert_array_equal(
        np.rint(
            child.parent_lattice_transform
        ).astype(int),
        np.array(
            [
                [2, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    )

def test_skew_sc_supercell_has_expected_lattice_sites(sc):
    child = _skew_sc_supercell(sc)

    expected_fractional = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )

    actual = child.motifs["pos"]

    assert len(actual) == 2

    for expected in expected_fractional:
        assert np.any(
            np.all(
                np.isclose(
                    actual,
                    expected,
                    atol=1e-12,
                ),
                axis=1,
            )
        )

def test_skew_sc_replication_plane_intervals(sc):
    child = _skew_sc_supercell(sc)

    np.testing.assert_array_equal(
        child.replication_plane_intervals,
        [2, 2, 1],
    )

    # Independent geometrical verification.
    a1, a2, a3 = child.primitive_vectors

    n1 = np.cross(a2, a3)
    n1 /= np.linalg.norm(n1)

    H1 = abs(np.dot(a1, n1))
    d1 = 1.0

    assert H1 / d1 == pytest.approx(2.0)

    n2 = np.cross(a3, a1)
    n2 /= np.linalg.norm(n2)

    H2 = abs(np.dot(a2, n2))
    d2 = 1.0 / np.sqrt(5.0)

    assert H2 / d2 == pytest.approx(2.0)

    n3 = np.cross(a1, a2)
    n3 /= np.linalg.norm(n3)

    H3 = abs(np.dot(a3, n3))
    d3 = 1.0

    assert H3 / d3 == pytest.approx(1.0)

def test_skew_sc_partial_replication_exact_positions(sc):
    child = _skew_sc_supercell(sc)

    positions, atom_types, cell = child.replicate(
        repetitions=[1, 1, 1],
        partial_plane_steps=[1, 0, 0],
    )

    expected_cell = np.array(
        [
            [3.0, 1.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        cell,
        expected_cell,
    )

    expected_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )

    assert len(positions) == 3

    for expected in expected_positions:
        assert np.any(
            np.all(
                np.isclose(
                    positions,
                    expected,
                    atol=1e-12,
                ),
                axis=1,
            )
        )

def test_partial_step_adds_exactly_one_plane_spacing(sc):
    child = _skew_sc_supercell(sc)

    _, _, new_cell = child.replicate(
        repetitions=[1, 1, 1],
        partial_plane_steps=[1, 0, 0],
    )

    old_a1, old_a2, old_a3 = (
        child.primitive_vectors
    )

    new_a1 = new_cell[0]

    normal = np.cross(old_a2, old_a3)
    normal /= np.linalg.norm(normal)

    H_old = abs(
        np.dot(old_a1, normal)
    )

    H_new = abs(
        np.dot(new_a1, normal)
    )

    d = 1.0

    assert H_new - H_old == pytest.approx(d)

def test_skew_sc_full_replication_preserves_parent_lattice(sc):
    child = _skew_sc_supercell(sc)

    positions, atom_types, cell = child.replicate(
        repetitions=[2, 3, 1]
    )

    expected_cell = np.array(
        [
            [4.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        cell,
        expected_cell,
    )

    assert len(positions) == 12

    np.testing.assert_allclose(
        positions,
        np.rint(positions),
        atol=1e-12,
    )

def _assert_no_periodic_duplicates(
    positions,
    cell,
    atol=1e-10,
):
    fractional = np.linalg.solve(
        cell.T,
        positions.T,
    ).T

    wrapped = fractional % 1.0

    for i in range(len(wrapped)):
        for j in range(i + 1, len(wrapped)):
            assert not np.allclose(
                wrapped[i],
                wrapped[j],
                atol=atol,
            )

def test_replication_plane_intervals_are_metric_independent():
    parent_vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.3, 1.1, 0.0],
            [0.2, 0.1, 0.9],
        ]
    )

    parent = CrystalStructure(
        parent_vectors
    )

    S = np.array(
        [
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    child = CrystalStructure(
        S @ parent_vectors,
        motif_positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
            ]
        ),
        motif_types=[
            1,
            1,
        ],
        parent_crystal_structure=parent,
        parent_motif_indices=[
            0,
            0,
        ],
    )

    np.testing.assert_array_equal(
        child.replication_plane_intervals,
        [2, 2, 1],
    )

