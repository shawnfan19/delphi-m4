"""Self-checks for the dynamic-DPP set-valued TPP (delphi/model/tpp.py).

The non-trivial logic is the DPP set log-prob  log det(L_S) - log det(L + I)
and its low-rank normaliser. We pin it down against an *independent* reference
(the marginal-kernel atom formula used in the paper's reference code) and the
exact DPP property that subset probabilities sum to 1, then check the class's
mark/time decomposition and same-age cluster masking.

Run: ``python delphi/test/test_dynamic_dpp.py``  (or via pytest).
"""

import math
from itertools import chain, combinations

import torch
import torch.nn.functional as F

from delphi.model.tpp import DPPSetHead, DynamicDPPTPP

DT = torch.float64


def _L_from(q: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Quality-diversity L-ensemble kernel  L_ij = q_i q_j <e_i,e_j> (unit norm)."""
    phi = q.unsqueeze(-1) * F.normalize(E, dim=-1)
    return phi @ phi.transpose(-1, -2)


def _ref_set_logprob(L: torch.Tensor, subset) -> torch.Tensor:
    """log P(set = subset) via the marginal kernel K = I - (L+I)^{-1}.

    Independent of the shipped log det(L_S) - log det(L+I) implementation:
    uses  det(I_B K + I_{B^c}(I - K))  (the reference repo's formula).
    """
    V = L.shape[0]
    eye = torch.eye(V, dtype=L.dtype)
    K = eye - torch.linalg.inv(L + eye)
    b = torch.zeros(V, dtype=L.dtype)
    if subset:
        b[list(subset)] = 1.0
    i_b, i_bbar = torch.diag(b), torch.diag(1 - b)
    return torch.logdet(i_b @ K + i_bbar @ (eye - K))


def _fixed_head(vocab_size: int, n_embd: int, q: torch.Tensor) -> DPPSetHead:
    """A head whose quality output is a fixed q (weights 0, bias = log q)."""
    head = DPPSetHead(n_embd=n_embd, vocab_size=vocab_size).to(DT)
    with torch.no_grad():
        head.quality.weight.zero_()
        head.quality.bias.copy_(torch.log(q))
    return head


def _tpp(head, E, exclude):
    """A DynamicDPPTPP whose history is the single reserved token 0 at age 1.

    Token 0 is the codebase's padding/exempt sentinel: ``have_occurred`` folds
    the exempt prior token into column 0, so token 0 reads as already-occurred
    (q -> 0). We mirror that here and never test token 0 as a real mark.
    """
    return DynamicDPPTPP(
        hidden_states=torch.zeros(1, 1, E.shape[1], dtype=DT),
        head=head,
        embedding=E,
        timesteps=torch.tensor([[1.0]], dtype=DT),
        tokens=torch.tensor([[0]]),
        exclude=torch.tensor(exclude, dtype=torch.long),
        terminate_except=torch.tensor([0]),
        time_unit=1.0,
    )


def _effective_q(q: torch.Tensor, excluded=()) -> torch.Tensor:
    """q as the class sees it: reserved token 0 and excluded tokens forced to 0."""
    q = q.clone()
    q[0] = 0.0  # reserved/occurred sentinel
    for k in excluded:
        q[k] = 0.0
    return q


def test_normalization_sums_to_one():
    """Exact DPP: sum over all 2^V subsets of P(set) == 1."""
    torch.manual_seed(0)
    V, d = 4, 3
    q = torch.rand(V, dtype=DT) + 0.5
    E = torch.randn(V, d, dtype=DT)
    L = _L_from(q, E)
    powerset = chain.from_iterable(combinations(range(V), r) for r in range(V + 1))
    total = sum(_ref_set_logprob(L, set(s)).exp() for s in powerset)
    assert torch.allclose(total, torch.tensor(1.0, dtype=DT), atol=1e-8), total


def test_class_matches_reference_on_nonempty_sets():
    """Shipped log det(L_S) - log det(L+I) == independent K-based reference."""
    torch.manual_seed(1)
    V, d = 5, 6  # token 0 reserved; real marks are 1..V-1
    q = torch.rand(V, dtype=DT) + 0.5
    E = torch.randn(V, d, dtype=DT)
    L = _L_from(_effective_q(q), E)
    head = _fixed_head(V, d, q)
    tpp = _tpp(head, E, exclude=[])

    for r in range(1, V):
        for s in combinations(range(1, V), r):
            toks = torch.tensor([list(s)])
            ages = torch.full((1, len(s)), 2.0, dtype=DT)
            got = tpp.log_p_marks(toks, ages)[0, 0]  # col 0 = cluster rep
            want = _ref_set_logprob(L, set(s))
            assert torch.allclose(got, want, atol=1e-5), (s, got.item(), want.item())


def test_excluded_token_is_empty_set():
    """A target that is an excluded token -> empty disease set -> -log det(L+I)."""
    torch.manual_seed(2)
    V, d = 5, 6
    q = torch.rand(V, dtype=DT) + 0.5
    E = torch.randn(V, d, dtype=DT)
    head = _fixed_head(V, d, q)
    # exclude token 3 (and token 0 is the reserved sentinel): both q -> 0
    tpp = _tpp(head, E, exclude=[3])
    L_kept = _L_from(_effective_q(q, excluded=[3]), E)

    got = tpp.log_p_marks(torch.tensor([[3]]), torch.tensor([[2.0]], dtype=DT))[0, 0]
    want = _ref_set_logprob(L_kept, set())  # == -log det(L_kept + I)
    assert torch.allclose(got, want, atol=1e-5), (got.item(), want.item())


def test_decomposition_and_cluster_masking():
    """joint == marks + times where finite; same-age continuation is NaN."""
    torch.manual_seed(3)
    V, d = 6, 4
    q = torch.rand(V, dtype=DT) + 0.5
    E = torch.randn(V, d, dtype=DT)
    head = _fixed_head(V, d, q)
    with torch.no_grad():  # nonzero, finite timing
        head.total_intensity.bias.fill_(0.5)
    tpp = _tpp(head, E, exclude=[])

    # one row: tokens 2 & 4 co-occur at age 2 (a cluster of size 2)
    x1 = torch.tensor([[2, 4]])
    t1 = torch.tensor([[2.0, 2.0]], dtype=DT)

    joint = tpp.log_likelihood(x1, t1)
    marks = tpp.log_p_marks(x1, t1)
    times = tpp.log_p_times(t1)

    assert torch.isnan(joint[0, 1]) and torch.isnan(marks[0, 1])  # continuation
    assert torch.isfinite(joint[0, 0])  # cluster rep scored once
    assert torch.allclose(joint[0, 0], marks[0, 0] + times[0, 0], atol=1e-10)


def test_intensity_is_lambda_times_marginal_inclusion():
    """intensity_m = lambda* * K_mm, and K_mm == sum over sets containing m."""
    torch.manual_seed(4)
    V, d = 5, 6  # token 0 reserved (occurred); real marks 1..V-1
    q = torch.rand(V, dtype=DT) + 0.5
    E = torch.randn(V, d, dtype=DT)
    L = _L_from(_effective_q(q), E)  # token-0 row/col is 0
    eye = torch.eye(V, dtype=DT)
    K = eye - torch.linalg.inv(L + eye)  # analytic marginal kernel
    Z = torch.logdet(L + eye).exp()

    # brute force: P(m in X) = sum_{S ni m} det(L_S) / det(L + I) == K_mm
    def brute(m):
        tot = torch.zeros((), dtype=DT)
        for r in range(1, V + 1):
            for s in combinations(range(V), r):
                if m in s:
                    j = torch.tensor(s)
                    tot = tot + torch.det(L[j][:, j])
        return tot / Z

    for m in range(V):
        assert torch.allclose(K[m, m], brute(m), atol=1e-6), m

    # class output with lambda* pinned to 1 -> intensity == K_mm
    head = _fixed_head(V, d, q)
    with torch.no_grad():
        head.total_intensity.weight.zero_()
        head.total_intensity.bias.fill_(math.log(math.e - 1.0))  # softplus -> 1
    tpp = _tpp(head, E, exclude=[])
    inten, _ = tpp.intensity(torch.tensor([[2.0]], dtype=DT))  # (1, 1, V)
    assert torch.allclose(inten[0, 0], torch.diagonal(K), atol=1e-5), inten[0, 0]
    assert inten[0, 0, 0].item() == 0.0  # token 0 reserved/occurred -> 0 intensity


def test_intensity_at_matches_intensity_gather():
    """intensity_at(t, m) == intensity(t)[..., m], with no-history queries NaN."""
    torch.manual_seed(5)
    V, d = 6, 5
    q = torch.rand(V, dtype=DT) + 0.5
    E = torch.randn(V, d, dtype=DT)
    head = _fixed_head(V, d, q)
    with torch.no_grad():
        head.total_intensity.bias.fill_(0.3)  # nonzero lambda*
    tpp = _tpp(head, E, exclude=[])

    # three valid query points (after the single history event at age 1) plus
    # one invalid point before it; distinct queried tokens per column
    t = torch.tensor([[2.0, 3.0, 5.0, 0.5]], dtype=DT)
    tokens = torch.tensor([[1, 3, 4, 2]])

    full, _ = tpp.intensity(t)  # (1, 4, V)
    gathered = full.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)  # (1, 4)
    at, nt = tpp.intensity_at(t, tokens)

    valid = ~torch.isnan(gathered)
    assert valid.tolist() == [[True, True, True, False]]  # last query has no history
    assert torch.allclose(at[valid], gathered[valid], atol=1e-6)
    assert torch.isnan(at[~valid]).all()  # invalid query -> NaN intensity
    assert (nt[~valid] == -1e4).all()  # invalid query -> -1e4 nearest_t


def test_intensity_gathers_per_position_kernel():
    """With multiple history events, a query uses the kernel of the event just
    before it (correct occurred-set) via the precompute + gather; also exercises
    the position-chunk loop (_POS_BLOCK forced to 1)."""
    torch.manual_seed(6)
    V, d = 7, 6  # marks 1..6 (token 0 kept here -- history tokens are 1,2,3)
    qbase = torch.rand(V, dtype=DT) + 0.5
    E = torch.randn(V, d, dtype=DT)
    head = _fixed_head(V, d, qbase)  # quality = qbase at every position
    with torch.no_grad():
        head.total_intensity.weight.zero_()
        head.total_intensity.bias.fill_(math.log(math.e - 1.0))  # lambda* = 1
    tpp = DynamicDPPTPP(
        hidden_states=torch.randn(1, 3, d, dtype=DT),  # irrelevant: head ignores h
        head=head,
        embedding=E,
        timesteps=torch.tensor([[1.0, 2.0, 3.0]], dtype=DT),
        tokens=torch.tensor([[1, 2, 3]]),  # events token1@1, token2@2, token3@3
        exclude=torch.tensor([], dtype=torch.long),
        terminate_except=torch.tensor([0]),
        time_unit=1.0,
    )
    tpp._POS_BLOCK = 1  # force the position-chunk loop (3 single-position blocks)

    # queries between events -> prediction positions 0, 1, 2
    t = torch.tensor([[1.5, 2.5, 3.5]], dtype=DT)
    inten, _ = tpp.intensity(t)  # (1, 3, V); lambda* == 1 so == K diagonal

    # occurred set just before each query is the cumulative history tokens
    for j, occ in enumerate([{1}, {1, 2}, {1, 2, 3}]):
        qe = qbase.clone()
        for k in occ:
            qe[k] = 0.0  # self-terminated -> dropped from the kernel
        L = _L_from(qe, E)
        eye = torch.eye(V, dtype=DT)
        K = eye - torch.linalg.inv(L + eye)
        assert torch.allclose(inten[0, j], torch.diagonal(K), atol=1e-6), j


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
