from __future__ import annotations

from zoo.adapters.similarity import cosine, pairwise_matrix
from zoo.adapters.synth import make, make_zoo


def test_make_shape() -> None:
    a = make("t1", d_in=32, d_out=32, rank=8, alpha=16)
    assert a.A["q_proj"].shape == (8, 32)
    assert a.B["q_proj"].shape == (32, 8)


def test_effective_delta_shape() -> None:
    a = make("t1", d_in=16, d_out=24, rank=4, alpha=8)
    d = a.effective_delta()
    assert d["q_proj"].shape == (24, 16)


def test_cosine_self_one() -> None:
    a = make("t1", d_in=16, d_out=16)
    assert abs(cosine(a, a) - 1.0) < 1e-9


def test_pairwise_matrix_symmetric() -> None:
    zoo = make_zoo(n=4)
    m = pairwise_matrix(zoo)
    assert m.shape == (4, 4)
    for i in range(4):
        for j in range(4):
            assert abs(m[i, j] - m[j, i]) < 1e-9


def test_pairwise_diag_is_one() -> None:
    zoo = make_zoo(n=4)
    m = pairwise_matrix(zoo)
    n = m.shape[0]
    # all on-diagonal entries are self-cosine = 1
    for i in range(n):
        assert abs(m[i, i] - 1.0) < 1e-9
