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

    @classmethod
    def tuned(
        cls,
        X: np.ndarray,
        occurrence_times: np.ndarray,
        censorship_times: np.ndarray,
        *,
        alphas,
        cv_folds: int,
        ties: str = "efron",
        n_jobs: int = 1,
    ) -> tuple["CoxRidge", float]:
        """Fit with ``alpha`` chosen by stratified k-fold CV on the training data.

        CV scores by Harrell's c-index (sksurv's default scorer), folds stratified
        on the event indicator so each has events, then refits on all of ``X`` with
        the best ``alpha``. Returns the fitted ``CoxRidge`` and the chosen ``alpha``.
        """
        from sklearn.model_selection import GridSearchCV, StratifiedKFold

        X = np.asarray(X, dtype=float)
        y = to_surv(occurrence_times, censorship_times)
        event = ~np.isnan(np.asarray(occurrence_times, dtype=float))
        splits = list(
            StratifiedKFold(cv_folds, shuffle=True, random_state=0).split(X, event)
        )
        search = GridSearchCV(
            CoxPHSurvivalAnalysis(ties=ties),
            {"alpha": list(alphas)},
            cv=splits,
            n_jobs=n_jobs,
        ).fit(X, y)
        best = cls(ties=ties)
        best.model = search.best_estimator_  # already refit on all of X
        return best, float(search.best_params_["alpha"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Risk score (linear predictor ``beta^T x``); higher => earlier event."""
        return self.model.predict(np.asarray(X, dtype=float))

    @property
    def coef_(self) -> np.ndarray:
        return self.model.coef_
