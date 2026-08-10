"""DGP parameter configurations for all 9 data-generating processes."""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class DGPConfig:
    dgp_id: int
    K: int                              # number of regimes
    parameterization: str               # 'diagonal' or 'off_diagonal'
    common_variance: bool               # if True, all regimes share sigma^2
    transition_model: str               # 'Constant', 'ModelI', 'ModelII', 'ModelIII'
    mu: List[float]                     # regime means
    sigma2: List[float]                 # regime variances (single element if common)
    baseline_diag: Optional[List[float]] = None      # diagonal probs for diagonal param
    baseline_off_diag: Optional[List[float]] = None  # off-diag probs for off-diag param
    A: Optional[List[float]] = None    # TVTP score coefficients
    B: Optional[List[float]] = None    # GAS persistence coefficients
    estimation_models: List[str] = field(default_factory=list)
    dgp_x_sigma: float = 1.0           # std dev for X_t in ModelII DGPs

    def get_omega(self) -> np.ndarray:
        """Return baseline logit parameters (omega) derived from baseline probabilities."""
        if self.parameterization == 'diagonal':
            probs = np.array(self.baseline_diag)
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            return np.log(probs / (1 - probs))
        else:
            probs = np.array(self.baseline_off_diag)
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            return np.log(probs / (1 - probs))

    def get_sigma2_full(self) -> np.ndarray:
        """Return per-regime variance array of length K."""
        if self.common_variance:
            return np.full(self.K, self.sigma2[0])
        return np.array(self.sigma2)


# ---------------------------------------------------------------------------
# DGP 1: K=2, diagonal, common variance, Constant transitions
# ---------------------------------------------------------------------------
DGP1 = DGPConfig(
    dgp_id=1,
    K=2,
    parameterization='diagonal',
    common_variance=True,
    transition_model='Constant',
    mu=[-1.0, 1.0],
    sigma2=[0.5],
    baseline_diag=[0.80, 0.90],
    estimation_models=['Constant'],
)

# ---------------------------------------------------------------------------
# DGP 2: K=2, diagonal, common variance, Model I (y_{t-1} driver)
# ---------------------------------------------------------------------------
DGP2 = DGPConfig(
    dgp_id=2,
    K=2,
    parameterization='diagonal',
    common_variance=True,
    transition_model='ModelI',
    mu=[-1.0, 1.0],
    sigma2=[0.5],
    baseline_diag=[0.80, 0.90],
    A=[0.15, -0.10],
    estimation_models=['Constant', 'ModelI', 'ModelII', 'ModelIII'],
)

# ---------------------------------------------------------------------------
# DGP 3: K=2, diagonal, common variance, Model II (X_{t-1} driver)
# ---------------------------------------------------------------------------
DGP3 = DGPConfig(
    dgp_id=3,
    K=2,
    parameterization='diagonal',
    common_variance=True,
    transition_model='ModelII',
    mu=[-1.0, 1.0],
    sigma2=[0.5],
    baseline_diag=[0.80, 0.90],
    A=[0.20, -0.20],
    estimation_models=['Constant', 'ModelI', 'ModelII', 'ModelIII'],
)

# ---------------------------------------------------------------------------
# DGP 4: K=2, diagonal, common variance, Model III (GAS)
# ---------------------------------------------------------------------------
DGP4 = DGPConfig(
    dgp_id=4,
    K=2,
    parameterization='diagonal',
    common_variance=True,
    transition_model='ModelIII',
    mu=[-1.0, 1.0],
    sigma2=[0.5],
    baseline_diag=[0.80, 0.90],
    A=[0.10, -0.10],
    B=[0.90, 0.85],
    estimation_models=['Constant', 'ModelI', 'ModelII', 'ModelIII'],
)

# ---------------------------------------------------------------------------
# DGP 5: K=2, off_diagonal, regime-specific variances, Model I
# ---------------------------------------------------------------------------
DGP5 = DGPConfig(
    dgp_id=5,
    K=2,
    parameterization='off_diagonal',
    common_variance=False,
    transition_model='ModelI',
    mu=[-1.0, 1.0],
    sigma2=[0.3, 0.7],
    baseline_off_diag=[0.20, 0.15],   # [pi_12, pi_21]
    A=[0.10, -0.10],
    estimation_models=['Constant', 'ModelI'],
)

# ---------------------------------------------------------------------------
# DGP 6: K=3, off_diagonal, regime-specific variances, Constant
# ---------------------------------------------------------------------------
DGP6 = DGPConfig(
    dgp_id=6,
    K=3,
    parameterization='off_diagonal',
    common_variance=False,
    transition_model='Constant',
    mu=[-2.0, 0.0, 2.0],
    sigma2=[0.3, 0.5, 0.8],
    baseline_off_diag=[0.08, 0.08, 0.10, 0.10, 0.06, 0.06],
    # [pi_12, pi_13, pi_21, pi_23, pi_31, pi_32]
    estimation_models=['Constant'],
)

# ---------------------------------------------------------------------------
# DGP 7: K=3, off_diagonal, regime-specific variances, Model I
# ---------------------------------------------------------------------------
DGP7 = DGPConfig(
    dgp_id=7,
    K=3,
    parameterization='off_diagonal',
    common_variance=False,
    transition_model='ModelI',
    mu=[-2.0, 0.0, 2.0],
    sigma2=[0.3, 0.5, 0.8],
    baseline_off_diag=[0.08, 0.08, 0.10, 0.10, 0.06, 0.06],
    A=[0.05, -0.03, 0.04, -0.04, 0.03, -0.05],
    estimation_models=['Constant', 'ModelI', 'ModelII', 'ModelIII'],
)

# ---------------------------------------------------------------------------
# DGP 8: K=3, off_diagonal, regime-specific variances, Model II
# ---------------------------------------------------------------------------
DGP8 = DGPConfig(
    dgp_id=8,
    K=3,
    parameterization='off_diagonal',
    common_variance=False,
    transition_model='ModelII',
    mu=[-2.0, 0.0, 2.0],
    sigma2=[0.3, 0.5, 0.8],
    baseline_off_diag=[0.08, 0.08, 0.10, 0.10, 0.06, 0.06],
    A=[0.08, -0.04, 0.05, -0.06, 0.04, -0.07],
    estimation_models=['Constant', 'ModelI', 'ModelII', 'ModelIII'],
)

# ---------------------------------------------------------------------------
# DGP 9: K=3, off_diagonal, regime-specific variances, Model III (GAS)
# ---------------------------------------------------------------------------
DGP9 = DGPConfig(
    dgp_id=9,
    K=3,
    parameterization='off_diagonal',
    common_variance=False,
    transition_model='ModelIII',
    mu=[-2.0, 0.0, 2.0],
    sigma2=[0.3, 0.5, 0.8],
    baseline_off_diag=[0.08, 0.08, 0.10, 0.10, 0.06, 0.06],
    A=[0.03] * 6,
    B=[0.85] * 6,
    estimation_models=['Constant', 'ModelI', 'ModelII', 'ModelIII'],
)

# Ordered list of all DGPs
ALL_DGPS: List[DGPConfig] = [DGP1, DGP2, DGP3, DGP4, DGP5, DGP6, DGP7, DGP8, DGP9]

# Convenience lookup by ID
DGP_BY_ID = {dgp.dgp_id: dgp for dgp in ALL_DGPS}


def get_all_dgps() -> dict:
    """Return all DGP configurations keyed by dgp_id (1–9)."""
    return DGP_BY_ID
