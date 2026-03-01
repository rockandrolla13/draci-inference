# draci-inference

Doubly Robust Adaptive Conformal Inference (DR-ACI) for individual treatment effect estimation under beta-mixing temporal dependence.

## Installation

```bash
pip install -e .                  # core library
pip install -e ".[xgboost]"       # + XGBoost nuisance estimator
pip install -e ".[empirical]"     # + pandas/matplotlib for experiments
pip install -e ".[all]"           # everything
```

## Quick start

```python
from draci.dgp import generate_data
from draci.nuisance import fit_nuisances
from draci.conformal import dr_aci

# Generate AR(1) + GARCH data with heterogeneous treatment effects
data = generate_data(T=1000, rho=0.5)

# Fit nuisance functions (propensity, outcome models)
nf = fit_nuisances(data.X, data.W, data.Y, method="linear")

# Run DR-ACI conformal inference
result = dr_aci(data.Y, data.W, nf, alpha=0.10, gamma=0.005)
print(f"Coverage: {result.coverage:.3f}, Avg width: {result.avg_width:.3f}")
```

## Package structure

```
draci/              Core library (9 conformal methods, 3 nuisance backends, DGP)
simulation/         Monte Carlo coverage study (reproduces paper Figure 1, Table 1)
empirical/          M-ELO exchange data experiments (5 experiments)
tests/              Unit tests for core library
data/               Data download scripts + documentation
```

## Running experiments

```bash
# Simulation (quick mode: 10 MC trials, ~2 min)
python -m simulation --quick

# Simulation (full: 500 MC trials, ~4 hours)
python -m simulation

# Empirical (requires MELO_DATA_DIR env var)
export MELO_DATA_DIR=/path/to/dynamic-melo/data/processed
python -m empirical --experiment all --dev
```

## Methods

| Key | Method | Score type |
|-----|--------|-----------|
| `dr_aci` | DR-ACI (proposed) | Doubly robust |
| `vs_dr_aci` | Variance-standardized DR-ACI | Doubly robust |
| `split` | Split conformal | Naive |
| `nexcp` | NExCP | Doubly robust |
| `aci` | ACI (Gibbs & Candes) | Naive |
| `aci_no_dr` | ACI without DR scores | Naive |
| `eci` | ECI | Naive |
| `block_cp` | Block conformal | Naive |
| `hac` | HAC intervals | Naive |
| `oracle` | Oracle (known CATE) | Oracle |

## License

MIT
