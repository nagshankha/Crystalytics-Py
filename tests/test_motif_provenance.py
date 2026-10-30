import numpy as np
import pytest

from crystalytics_py.crystal_structure import (
    CrystalStructure,
)


_ROTATED_SUPERCELL_DIRECTIONS = np.array(
    [
        [1, 1, 0],
        [-1, 1, 0],
        [0, 0, 1],
    ],
    dtype=int,
)


def _two_motif_sc():
    """
    Construct a simple-cubic crystal with two chemically indistinguishable
    motifs so provenance cannot be inferred from motif type.
    """

    return CrystalStructure(
        np.eye(
            3,
            dtype=float,
        ),
        motif_positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
            ],
            dtype=float,
        ),
        motif_types=[
            7,
            7,
        ],
    )


def _rotated_supercell(
    crystal,
):
    """
    Construct the index-two [110]/[-110]/[001] superlattice relative
    to the supplied crystal's primitive-vector basis.
    """

    return (
        crystal
        .get_superlattice_crystal_structure(
            _ROTATED_SUPERCELL_DIRECTIONS,
            reference_basis="primitive_vector",
        )
    )


def test_root_crystal_has_identity_root_motif_indices():
    """
    A crystal without a parent should identify itself as the root and map
    every motif to the same-index motif of that root.
    """

    crystal = _two_motif_sc()

    assert (
        crystal.root_crystal_structure
        is crystal
    )

    assert (
        crystal.parent_crystal_structure
        is None
    )

    assert (
        crystal.parent_motif_indices
        is None
    )

    np.testing.assert_array_equal(
        crystal.root_motif_indices,
        np.array(
            [0, 1],
            dtype=int,
        ),
    )


def test_one_motif_superlattice_tracks_immediate_parent_motifs(
    sc,
):
    """
    Every motif generated from a one-motif parent should carry immediate
    parent motif index zero.
    """

    supercrystal = (
        _rotated_supercell(
            sc
        )
    )

    assert (
        supercrystal.parent_crystal_structure
        is sc
    )

    assert (
        len(
            supercrystal.motifs
        )
        == 2
    )

    np.testing.assert_array_equal(
        supercrystal.parent_motif_indices,
        np.array(
            [0, 0],
            dtype=int,
        ),
    )

    np.testing.assert_array_equal(
        supercrystal.root_motif_indices,
        np.array(
            [0, 0],
            dtype=int,
        ),
    )


def test_multi_motif_superlattice_tracks_parent_motif_indices():
    """
    A superlattice of an M-motif parent should contain M*N provenance
    entries and each parent motif should occur once per parent lattice site.
    """

    parent = _two_motif_sc()

    supercrystal = (
        _rotated_supercell(
            parent
        )
    )

    assert (
        supercrystal.parent_crystal_structure
        is parent
    )

    assert (
        len(
            supercrystal.motifs
        )
        == 4
    )

    assert (
        supercrystal.parent_motif_indices.shape
        == (4,)
    )

    np.testing.assert_array_equal(
        np.bincount(
            supercrystal.parent_motif_indices,
            minlength=2,
        ),
        np.array(
            [2, 2],
            dtype=int,
        ),
    )

    # Both parent motifs deliberately have the same type.
    # Provenance therefore cannot have been inferred from motif type.
    np.testing.assert_array_equal(
        supercrystal.motifs[
            "type"
        ],
        np.full(
            4,
            7,
            dtype=int,
        ),
    )


def test_parent_motif_indices_remain_aligned_after_motif_sorting():
    """
    Motif provenance should undergo exactly the same normalization and
    np.unique sorting as the corresponding superlattice motif positions.
    """

    parent = _two_motif_sc()

    supercrystal = (
        _rotated_supercell(
            parent
        )
    )

    expected_positions = np.array(
        [
            [0.000, 0.000, 0.0],
            [0.125, 0.875, 0.0],
            [0.500, 0.500, 0.0],
            [0.625, 0.375, 0.0],
        ],
        dtype=float,
    )

    expected_parent_indices = np.array(
        [
            0,
            1,
            0,
            1,
        ],
        dtype=int,
    )

    np.testing.assert_allclose(
        supercrystal.motifs[
            "pos"
        ],
        expected_positions,
    )

    np.testing.assert_array_equal(
        supercrystal.parent_motif_indices,
        expected_parent_indices,
    )


def test_nested_superlattices_preserve_immediate_and_root_provenance():
    """
    Nested A -> B -> C construction should preserve immediate-parent
    references while composing motif provenance correctly back to A.
    """

    crystal_a = _two_motif_sc()

    crystal_b = (
        _rotated_supercell(
            crystal_a
        )
    )

    crystal_c = (
        _rotated_supercell(
            crystal_b
        )
    )

    assert (
        crystal_b.parent_crystal_structure
        is crystal_a
    )

    assert (
        crystal_c.parent_crystal_structure
        is crystal_b
    )

    assert (
        crystal_c.root_crystal_structure
        is crystal_a
    )

    np.testing.assert_array_equal(
        crystal_b.parent_motif_indices,
        np.array(
            [
                0,
                1,
                0,
                1,
            ],
            dtype=int,
        ),
    )

    np.testing.assert_array_equal(
        crystal_c.parent_motif_indices,
        np.array(
            [
                0,
                3,
                2,
                1,
                2,
                1,
                0,
                3,
            ],
            dtype=int,
        ),
    )

    np.testing.assert_array_equal(
        crystal_c.root_motif_indices,
        np.array(
            [
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                1,
            ],
            dtype=int,
        ),
    )


def test_parent_requires_parent_motif_indices():
    """
    A CrystalStructure with an explicit parent should require an
    immediate-parent motif provenance mapping.
    """

    parent = _two_motif_sc()

    with pytest.raises(
        ValueError,
        match="parent_motif_indices",
    ):
        CrystalStructure(
            np.eye(
                3,
                dtype=float,
            ),
            motif_positions=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                ],
                dtype=float,
            ),
            motif_types=[
                7,
                7,
            ],
            parent_crystal_structure=parent,
        )

def test_parent_motif_indices_require_parent():
    """
    Parent motif provenance should not be accepted when no parent
    CrystalStructure exists to give those indices meaning.
    """

    with pytest.raises(
        ValueError,
        match="parent_crystal_structure",
    ):
        CrystalStructure(
            np.eye(
                3,
                dtype=float,
            ),
            parent_motif_indices=[
                0,
            ],
        )

def test_direct_child_with_explicit_provenance_composes_root_mapping():
    """
    A directly constructed child should support root provenance when an
    unambiguous immediate-parent motif mapping is explicitly supplied.
    """

    parent = _two_motif_sc()

    child = CrystalStructure(
        np.eye(
            3,
            dtype=float,
        ),
        motif_positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=float,
        ),
        motif_types=[
            7,
            7,
        ],
        parent_crystal_structure=parent,
        parent_motif_indices=[
            1,
            0,
        ],
    )

    np.testing.assert_array_equal(
        child.parent_motif_indices,
        np.array(
            [
                1,
                0,
            ],
            dtype=int,
        ),
    )

    np.testing.assert_array_equal(
        child.root_motif_indices,
        np.array(
            [
                1,
                0,
            ],
            dtype=int,
        ),
    )


def test_ambiguous_duplicate_motif_provenance_is_rejected():
    """
    If motif normalization collapses two motifs to one but they reference
    different parent motifs, construction should fail rather than choose
    provenance arbitrarily.
    """

    parent = _two_motif_sc()

    with pytest.raises(
        ValueError,
        match="ambiguous",
    ):
        CrystalStructure(
            np.eye(
                3,
                dtype=float,
            ),
            motif_positions=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=float,
            ),
            motif_types=[
                7,
                7,
            ],
            parent_crystal_structure=parent,
            parent_motif_indices=[
                0,
                1,
            ],
        )


def test_direct_self_parent_cycle_is_rejected():
    """
    Assigning a CrystalStructure as its own immediate parent should be
    rejected at the point of parent assignment.
    """

    crystal = CrystalStructure(
        np.eye(
            3,
            dtype=float,
        )
    )

    with pytest.raises(
        ValueError,
        match="Circular parent hierarchy",
    ):
        crystal._parent_crystal_structure = (
            crystal
        )


def test_indirect_parent_cycle_is_rejected():
    """
    Assigning a parent that would close an A -> B -> C -> A ancestry loop
    should be rejected using object identity.
    """

    crystal_a = CrystalStructure(
        np.eye(
            3,
            dtype=float,
        )
    )

    crystal_b = CrystalStructure(
        np.eye(
            3,
            dtype=float,
        ),
        parent_crystal_structure=crystal_a,
        parent_motif_indices=[
            0,
        ],
    )

    crystal_c = CrystalStructure(
        np.eye(
            3,
            dtype=float,
        ),
        parent_crystal_structure=crystal_b,
        parent_motif_indices=[
            0,
        ],
    )

    with pytest.raises(
        ValueError,
        match="Circular parent hierarchy",
    ):
        crystal_a._parent_crystal_structure = (
            crystal_c
        )


def test_root_traversal_defensively_detects_corrupted_cycle():
    """
    Root traversal should still detect a cycle if internal state is corrupted
    by bypassing the validated parent-assignment path.
    """

    crystal_a = CrystalStructure(
        np.eye(
            3,
            dtype=float,
        )
    )

    crystal_b = CrystalStructure(
        np.eye(
            3,
            dtype=float,
        )
    )

    # Deliberately bypass __setattr__ to emulate corrupted or unusual
    # object construction.
    crystal_a.__dict__[
        "_parent_crystal_structure"
    ] = crystal_b

    crystal_b.__dict__[
        "_parent_crystal_structure"
    ] = crystal_a

    with pytest.raises(
        RuntimeError,
        match="Circular parent hierarchy",
    ):
        _ = (
            crystal_a
            .root_crystal_structure
        )

    with pytest.raises(
        RuntimeError,
        match="Circular parent hierarchy",
    ):
        _ = (
            crystal_a
            .root_motif_indices
        )


def test_replicate_returns_aligned_motif_indices_after_partial_filtering(
    sc,
):
    """
    Optional replicate provenance should remain aligned after a partial-cell
    boundary filters out some generated motif instances.
    """

    crystal = (
        _rotated_supercell(
            sc
        )
    )

    (
        positions,
        atom_types,
        cell_vectors,
        motif_indices,
    ) = crystal.replicate(
        repetitions=[
            1,
            1,
            1,
        ],
        partial_plane_steps=[
            1,
            0,
            0,
        ],
        return_motif_indices=True,
    )

    assert (
        len(positions)
        == len(atom_types)
        == len(motif_indices)
        == 3
    )

    np.testing.assert_array_equal(
        motif_indices,
        np.array(
            [
                0,
                1,
                0,
            ],
            dtype=int,
        ),
    )

    # Every generated position must differ from the source motif by
    # an integer translation of the current crystal cell.
    fractional_positions = (
        np.linalg.solve(
            crystal.primitive_vectors.T,
            positions.T,
        ).T
    )

    source_motif_positions = (
        crystal.motifs[
            "pos"
        ][
            motif_indices
        ]
    )

    translations = (
        fractional_positions
        - source_motif_positions
    )

    np.testing.assert_allclose(
        translations,
        np.rint(
            translations
        ),
        atol=1e-12,
    )

    # This crystal ultimately originates from the one-motif SC root.
    np.testing.assert_array_equal(
        crystal.root_motif_indices[
            motif_indices
        ],
        np.zeros(
            len(motif_indices),
            dtype=int,
        ),
    )

    assert (
        cell_vectors.shape
        == (3, 3)
    )


def test_replicate_preserves_three_value_return_by_default(
    sc,
):
    """
    Calling replicate without requesting motif provenance should retain the
    existing three-value return contract.
    """

    crystal = (
        _rotated_supercell(
            sc
        )
    )

    result = crystal.replicate(
        repetitions=[
            2,
            1,
            1,
        ]
    )

    assert isinstance(
        result,
        tuple,
    )

    assert len(
        result
    ) == 3

    (
        positions,
        atom_types,
        cell_vectors,
    ) = result

    assert (
        len(positions)
        == len(atom_types)
    )

    assert (
        cell_vectors.shape
        == (3, 3)
    )