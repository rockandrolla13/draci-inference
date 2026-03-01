"""Centralized method metadata registry for DR-ACI experiments.

Single source of truth for method keys, display labels, colors, and markers.
Adding method #10 requires: 1 function in conformal.py + 1 entry here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodInfo:
    key: str           # e.g. "dr_aci"
    label: str         # e.g. "DR-ACI"
    color: str         # hex color for plots
    marker: str        # matplotlib marker
    score_type: str    # "dr" | "naive" | "vs_dr" | "oracle"


METHODS_REGISTRY: dict[str, MethodInfo] = {
    "dr_aci":        MethodInfo("dr_aci",        "DR-ACI",              "#2166ac", "o",  "dr"),
    "vs_dr_aci":     MethodInfo("vs_dr_aci",     "VS-DR-ACI",           "#4393c3", "v",  "vs_dr"),
    "split":         MethodInfo("split",         "Split conformal",     "#d6604d", "s",  "naive"),
    "nexcp":         MethodInfo("nexcp",         "NExCP",               "#f4a582", "D",  "dr"),
    "aci_no_dr":     MethodInfo("aci_no_dr",     "ACI (non-DR)",        "#92c5de", "^",  "naive"),
    "eci":           MethodInfo("eci",           "ECI",                 "#b2182b", "p",  "naive"),
    "block_cp":      MethodInfo("block_cp",      "Block CP",            "#fddbc7", "h",  "naive"),
    "hac":           MethodInfo("hac",           r"HAC (Newey--West)",  "#878787", "*",  "naive"),
    "oracle":        MethodInfo("oracle",        "Oracle DR-ACI",       "#4d4d4d", "X",  "oracle"),
}

# Convenience accessors (replace duplicated dicts in config files)
METHODS = list(METHODS_REGISTRY.keys())
METHOD_LABELS = {k: v.label for k, v in METHODS_REGISTRY.items()}
METHOD_COLORS = {k: v.color for k, v in METHODS_REGISTRY.items()}
METHOD_MARKERS = {k: v.marker for k, v in METHODS_REGISTRY.items()}
