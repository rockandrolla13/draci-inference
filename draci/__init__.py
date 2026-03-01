"""DR-ACI: Doubly Robust Adaptive Conformal Inference under mixing."""

__version__ = "0.1.0"

from draci.conformal import (
    ConformalResult,
    dr_score,
    dr_pseudo_outcome,
    naive_score,
    vs_dr_score,
    dr_aci,
    vs_dr_aci,
    split_conformal,
    nexcp,
    aci,
    eci,
    block_cp,
    hac,
)
from draci.nuisance import (
    NuisanceTuple,
    NuisanceFunctions,
    fit_nuisances,
    fit_nuisance_xgboost,
    fit_nuisance_linear,
    get_nuisance_estimator,
    LinearNuisance,
    MLNuisance,
    XGBoostNuisance,
)
from draci.dgp import (
    DGPData,
    AR1DGP,
    get_dgp,
    generate_data,
)
from draci.methods import METHODS_REGISTRY, METHODS, METHOD_LABELS, METHOD_COLORS, METHOD_MARKERS

__all__ = [
    "ConformalResult",
    "dr_score", "dr_pseudo_outcome", "naive_score", "vs_dr_score",
    "dr_aci", "vs_dr_aci", "split_conformal", "nexcp", "aci", "eci",
    "block_cp", "hac",
    "NuisanceTuple", "NuisanceFunctions",
    "fit_nuisances", "fit_nuisance_xgboost", "fit_nuisance_linear",
    "get_nuisance_estimator",
    "LinearNuisance", "MLNuisance", "XGBoostNuisance",
    "DGPData", "AR1DGP", "get_dgp", "generate_data",
    "METHODS_REGISTRY", "METHODS", "METHOD_LABELS", "METHOD_COLORS", "METHOD_MARKERS",
]
