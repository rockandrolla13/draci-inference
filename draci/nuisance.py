"""
Nuisance estimators for DR-ACI simulations.

Unified API:
  - **Class-based** (LinearNuisance, MLNuisance, XGBoostNuisance,
    get_nuisance_estimator): .fit() returns NuisanceFunctions.
  - **Functional** (fit_nuisances): dispatches to class-based API,
    returns NuisanceFunctions.
  - Legacy functions (fit_nuisance_xgboost, fit_nuisance_linear) retained
    for backward compat; they delegate to the class-based API.
"""

import numpy as np
from typing import Callable, NamedTuple, Optional


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    """Reshape 1D array to column vector; pass 2D+ through unchanged."""
    return X.reshape(-1, 1) if X.ndim == 1 else X


# =========================================================================
# Unified return type
# =========================================================================

class NuisanceFunctions(NamedTuple):
    """Container for estimated nuisance functions."""
    e_hat: Callable[[np.ndarray], np.ndarray]
    mu0_hat: Callable[[np.ndarray], np.ndarray]
    mu1_hat: Callable[[np.ndarray], np.ndarray]
    tau_hat: Optional[Callable[[np.ndarray], np.ndarray]] = None


# Deprecated alias -- points to the same type
NuisanceTuple = NuisanceFunctions


class LinearNuisance:
    """Linear nuisance estimator (logistic propensity + OLS outcomes).

    Parameters
    ----------
    clip_propensity : tuple[float, float]
        Propensity clipping bounds for overlap (A3).
    """

    def __init__(self, clip_propensity: tuple[float, float] = (0.05, 0.95)):
        self.clip_propensity = clip_propensity

    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> NuisanceFunctions:
        from scipy.optimize import minimize

        lo, hi = self.clip_propensity
        X_2d = _ensure_2d(X)
        X_aug = np.column_stack([np.ones(len(X_2d)), X_2d])
        d = X_aug.shape[1]

        # Logistic regression for propensity
        def neg_loglik(beta):
            logits = np.clip(X_aug @ beta, -30, 30)
            return -np.mean(W * logits - np.log(1 + np.exp(logits)))

        res = minimize(neg_loglik, np.zeros(d), method='L-BFGS-B')
        beta_e = res.x

        def e_hat(X_new):
            X_new_2d = _ensure_2d(X_new)
            X_a = np.column_stack([np.ones(len(X_new_2d)), X_new_2d])
            logits = np.clip(X_a @ beta_e, -30, 30)
            return np.clip(1.0 / (1.0 + np.exp(-logits)), lo, hi)

        # OLS per treatment arm
        idx1 = W == 1
        idx0 = W == 0
        X1_aug = np.column_stack([np.ones(idx1.sum()), X_2d[idx1]])
        X0_aug = np.column_stack([np.ones(idx0.sum()), X_2d[idx0]])
        beta1 = np.linalg.lstsq(X1_aug, Y[idx1], rcond=None)[0]
        beta0 = np.linalg.lstsq(X0_aug, Y[idx0], rcond=None)[0]

        def mu1_hat(X_new):
            X_new_2d = _ensure_2d(X_new)
            return np.column_stack([np.ones(len(X_new_2d)), X_new_2d]) @ beta1

        def mu0_hat(X_new):
            X_new_2d = _ensure_2d(X_new)
            return np.column_stack([np.ones(len(X_new_2d)), X_new_2d]) @ beta0

        def tau_hat(X_new):
            return mu1_hat(X_new) - mu0_hat(X_new)

        return NuisanceFunctions(e_hat=e_hat, mu0_hat=mu0_hat,
                                 mu1_hat=mu1_hat, tau_hat=tau_hat)


class MLNuisance:
    """ML nuisance estimator (Random Forest propensity + outcomes).

    Parameters
    ----------
    clip_propensity : tuple[float, float]
        Propensity clipping bounds for overlap (A3).
    """

    def __init__(self, clip_propensity: tuple[float, float] = (0.05, 0.95)):
        self.clip_propensity = clip_propensity

    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> NuisanceFunctions:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        lo, hi = self.clip_propensity
        X_2d = _ensure_2d(X)

        # Propensity via RF classifier
        prop_model = RandomForestClassifier(
            n_estimators=100, max_depth=3, random_state=0, n_jobs=1,
        )
        prop_model.fit(X_2d, W)

        def e_hat(X_new):
            X_new_2d = _ensure_2d(X_new)
            return np.clip(prop_model.predict_proba(X_new_2d)[:, 1], lo, hi)

        # Outcome models per arm
        idx1 = W == 1
        idx0 = W == 0

        mu1_model = RandomForestRegressor(
            n_estimators=100, max_depth=3, random_state=0, n_jobs=1,
        )
        mu1_model.fit(X_2d[idx1], Y[idx1])

        mu0_model = RandomForestRegressor(
            n_estimators=100, max_depth=3, random_state=0, n_jobs=1,
        )
        mu0_model.fit(X_2d[idx0], Y[idx0])

        def mu1_hat(X_new):
            return mu1_model.predict(_ensure_2d(X_new))

        def mu0_hat(X_new):
            return mu0_model.predict(_ensure_2d(X_new))

        def tau_hat(X_new):
            return mu1_hat(X_new) - mu0_hat(X_new)

        return NuisanceFunctions(e_hat=e_hat, mu0_hat=mu0_hat,
                                 mu1_hat=mu1_hat, tau_hat=tau_hat)


class XGBoostNuisance:
    """XGBoost nuisance estimator (XGB propensity + outcome, RF CATE).

    Parameters
    ----------
    clip_propensity : tuple[float, float]
        Propensity clipping bounds for overlap (A3).
    """

    def __init__(self, clip_propensity: tuple[float, float] = (0.05, 0.95)):
        self.clip_propensity = clip_propensity

    def fit(self, X: np.ndarray, W: np.ndarray, Y: np.ndarray) -> NuisanceFunctions:
        from xgboost import XGBClassifier, XGBRegressor
        from sklearn.ensemble import RandomForestRegressor

        lo, hi = self.clip_propensity
        X_2d = _ensure_2d(X)

        # --- Propensity ---
        prop_model = XGBClassifier(
            max_depth=3, n_estimators=100, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss',
            verbosity=0, n_jobs=1,
        )
        prop_model.fit(X_2d, W)

        def e_hat(X_new):
            return np.clip(prop_model.predict_proba(_ensure_2d(X_new))[:, 1],
                           lo, hi)

        # --- Outcome models (separate per arm) ---
        idx1 = W == 1
        idx0 = W == 0

        mu1_model = XGBRegressor(
            max_depth=3, n_estimators=100, learning_rate=0.1,
            verbosity=0, n_jobs=1,
        )
        mu1_model.fit(X_2d[idx1], Y[idx1])

        mu0_model = XGBRegressor(
            max_depth=3, n_estimators=100, learning_rate=0.1,
            verbosity=0, n_jobs=1,
        )
        mu0_model.fit(X_2d[idx0], Y[idx0])

        def mu1_hat(X_new):
            return mu1_model.predict(_ensure_2d(X_new))

        def mu0_hat(X_new):
            return mu0_model.predict(_ensure_2d(X_new))

        # --- CATE: RF on plug-in pseudo-outcome ---
        tau_pseudo = mu1_hat(X_2d) - mu0_hat(X_2d)
        tau_model = RandomForestRegressor(
            n_estimators=100, max_depth=5, n_jobs=1, random_state=0,
        )
        tau_model.fit(X_2d, tau_pseudo)

        def tau_hat(X_new):
            return tau_model.predict(_ensure_2d(X_new))

        return NuisanceFunctions(e_hat=e_hat, mu0_hat=mu0_hat,
                                 mu1_hat=mu1_hat, tau_hat=tau_hat)


def get_nuisance_estimator(name: str, **kwargs):
    """Factory for nuisance estimator instances.

    Parameters
    ----------
    name : str
        'linear', 'ml' (RandomForest), or 'xgboost'.
    **kwargs
        Passed to the estimator constructor (e.g. clip_propensity).
    """
    if name == "linear":
        return LinearNuisance(**kwargs)
    if name == "ml":
        return MLNuisance(**kwargs)
    if name == "xgboost":
        return XGBoostNuisance(**kwargs)
    raise ValueError(f"Unknown nuisance estimator: {name}")


# =========================================================================
# Functional API — thin wrappers around class-based API (backward compat)
# =========================================================================

def fit_nuisance_xgboost(
    X_train: np.ndarray,
    W_train: np.ndarray,
    Y_train: np.ndarray,
    clip_propensity: tuple[float, float] = (0.05, 0.95),
) -> NuisanceFunctions:
    """XGBoost propensity + outcome, RandomForest CATE.

    Delegates to XGBoostNuisance.fit(). Retained for backward compat.
    """
    return XGBoostNuisance(clip_propensity=clip_propensity).fit(X_train, W_train, Y_train)


def fit_nuisance_linear(
    X_train: np.ndarray,
    W_train: np.ndarray,
    Y_train: np.ndarray,
    clip_propensity: tuple[float, float] = (0.05, 0.95),
) -> NuisanceFunctions:
    """Linear nuisance estimators (fast fallback for testing).

    Delegates to LinearNuisance.fit(). Retained for backward compat.
    """
    return LinearNuisance(clip_propensity=clip_propensity).fit(X_train, W_train, Y_train)


def fit_nuisances(
    X_train: np.ndarray,
    W_train: np.ndarray,
    Y_train: np.ndarray,
    method: str = 'xgboost',
    clip_propensity: tuple[float, float] = (0.05, 0.95),
) -> NuisanceFunctions:
    """Dispatch to the appropriate nuisance estimator.

    Parameters
    ----------
    method : str
        'xgboost' (production), 'ml' (RandomForest), or 'linear' (OLS).

    Returns
    -------
    NuisanceFunctions
        NamedTuple with fields (e_hat, mu0_hat, mu1_hat, tau_hat).
    """
    estimator = get_nuisance_estimator(method, clip_propensity=clip_propensity)
    return estimator.fit(X_train, W_train, Y_train)
