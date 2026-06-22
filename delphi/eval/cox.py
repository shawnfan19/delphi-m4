"""Ridge-penalized Cox proportional-hazards regression.

Thin wrapper over scikit-survival's :class:`CoxPHSurvivalAnalysis`, adapting the
repo's ``(occurrence_time, censorship_time)`` array convention to
scikit-survival's structured ``(event, time)`` target. Intended as a linear
baseline for disease-risk ranking, to compare against Delphi.

``scikit-survival`` is an optional, heavy dependency (it pulls ``numpy>=2`` and
``scikit-learn>=1.8``). It is imported here and deliberately *not* re-exported
from ``delphi.eval.__init__`` so the rest of the eval package stays importable
without it.
"""

# ponytail: sksurv does the Cox/ridge math + tie handling; this module only owns
# the (occurrence, censorship) -> (event, time) contract and a thin fit/predict.

import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv


def to_surv(occurrence_times: np.ndarray, censorship_times: np.ndarray) -> np.ndarray:
    """Combine occurrence/censorship arrays into a scikit-survival target.

    Occurrence-authoritative contract: an event occurred iff its occurrence time
    is not NaN; the observed time is the occurrence time for those subjects and
    the censorship time otherwise. Both inputs are ``(N,)`` and NaN marks "no
    event" in ``occurrence_times``.
    """
    occ = np.asarray(occurrence_times, dtype=float)
    cens = np.asarray(censorship_times, dtype=float)
    if occ.ndim != 1 or occ.shape != cens.shape:
        raise ValueError(
            f"expected matching 1-D arrays, got {occ.shape} and {cens.shape}"
        )

    event = ~np.isnan(occ)
    time = np.where(event, occ, cens)
    if not event.any():
        raise ValueError("no events (all occurrence times are NaN) — cannot fit Cox")
    if not np.isfinite(time).all():
        raise ValueError(
            "non-finite observed time (a censored row with NaN/inf censorship?)"
        )
    return Surv.from_arrays(event=event, time=time)


class CoxRidge:
    """Ridge (L2)-penalized Cox proportional-hazards fitter.

    Covariates are used as given — no internal standardization — so pass
    pre-scaled features (ridge is scale-sensitive). ``alpha`` is the ridge
    penalty weight; ``ties`` selects Efron (default) or Breslow tie handling.
    """

    def __init__(self, alpha: float = 1.0, ties: str = "efron"):
        self.model = CoxPHSurvivalAnalysis(alpha=alpha, ties=ties)

    def fit(
        self,
        X: np.ndarray,
        occurrence_times: np.ndarray,
        censorship_times: np.ndarray,
    ) -> "CoxRidge":
        X = np.asarray(X, dtype=float)
        y = to_surv(occurrence_times, censorship_times)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but {y.shape[0]} time entries")
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Risk score (linear predictor ``beta^T x``); higher => earlier event."""
        return self.model.predict(np.asarray(X, dtype=float))

    @property
    def coef_(self) -> np.ndarray:
        return self.model.coef_


def _demo():
    """Self-check: recover known coefficients, cross-check vs statsmodels, ridge shrinks."""
    from scipy.stats import spearmanr
    from statsmodels.duration.hazard_regression import PHReg

    rng = np.random.default_rng(0)
    n, p = 4000, 3
    beta = np.array([1.0, -0.5, 0.25])
    X = rng.standard_normal((n, p))
    # exponential survival: rate = exp(X beta) => higher X beta => earlier event
    t_event = rng.exponential(scale=np.exp(-X @ beta))
    t_cens = rng.exponential(scale=5.0, size=n)
    event = t_event <= t_cens
    occ = np.where(event, t_event, np.nan)
    cens = t_cens
    assert 0.05 < (~event).mean() < 0.6, "self-check needs a sane censoring fraction"

    # contract: hand example (detect fields by dtype, don't assume order)
    y = to_surv(np.array([5.0, np.nan, 8.0]), np.array([9.0, 7.0, 10.0]))
    ev = next(nm for nm in y.dtype.names if y[nm].dtype == bool)
    tm = next(nm for nm in y.dtype.names if nm != ev)
    assert list(y[ev]) == [True, False, True]
    assert list(y[tm]) == [5.0, 7.0, 8.0]

    # plumbing: unpenalized sksurv ~ statsmodels MLE (same ties)
    cox0 = CoxRidge(alpha=0.0, ties="efron").fit(X, occ, cens)
    obs_time = np.where(event, t_event, t_cens)
    sm = PHReg(obs_time, X, status=event.astype(int), ties="efron").fit()
    assert np.allclose(cox0.coef_, sm.params, atol=0.02), (cox0.coef_, sm.params)

    # recovery: signs and rough magnitude
    assert np.sign(cox0.coef_).tolist() == np.sign(beta).tolist()
    assert np.allclose(cox0.coef_, beta, atol=0.15), (cox0.coef_, beta)

    # ridge shrinks the coefficient norm
    cox_r = CoxRidge(alpha=1e4, ties="efron").fit(X, occ, cens)
    assert np.linalg.norm(cox_r.coef_) < np.linalg.norm(cox0.coef_)

    # discrimination: higher risk score => earlier event among the uncensored
    rho, _ = spearmanr(cox0.predict(X)[event], t_event[event])
    assert rho < -0.3, rho

    print(
        "cox.py self-check OK | coef~",
        np.round(cox0.coef_, 3),
        "| sm~",
        np.round(sm.params, 3),
        "| censored=%.2f" % (~event).mean(),
    )


if __name__ == "__main__":
    _demo()
