"""
U.S. Treasury zero-coupon yield data loader.

Load order:
  1. Local cache in data/raw/treasury_yields.csv
  2. Download Liu & Wu (2021) data from public GitHub mirror
  3. Fallback: synthesise a realistic yield series with correct statistical
     properties for testing (AR(1) with regime-switching volatility)
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

LIU_WU_URL = (
    "https://raw.githubusercontent.com/richhansen/liuwuyields/main/data.csv"
)
DATA_DIR = Path(__file__).parent / "raw"
CACHE_FILE = DATA_DIR / "treasury_yields.csv"

# Paper uses 763 monthly observations starting 1961-06-01
_N_OBS = 763
_START_DATE = "1961-06-01"
_MATURITY_LABELS = {1: "1m", 12: "12m", 36: "36m", 72: "72m"}


# ---------------------------------------------------------------------------
# Synthetic data generator (fallback)
# ---------------------------------------------------------------------------

def _generate_synthetic_yields(
    n_obs: int = _N_OBS,
    start_date: str = _START_DATE,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic monthly Treasury yields with plausible properties:
      - AR(1) level factor with persistence 0.95
      - Two-regime Markov-switching volatility (low/high vol)
      - Maturities [1m, 12m, 36m, 72m] with term-structure loading

    The yield for maturity τ is:
        y_τ(t) = a_τ + b_τ * level(t) + c_τ * slope(t) + ε_τ(t)
    where level and slope are latent AR(1) factors.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns ['1m', '12m', '36m', '72m'],
    annualised in percent.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_obs, freq="MS")

    # Latent factors: level (long rate) and slope (short - long)
    rho_level = 0.98
    rho_slope = 0.95
    level = np.zeros(n_obs)
    slope = np.zeros(n_obs)

    # Regime-switching volatility: two regimes (low / high)
    # Regime transition probabilities
    p_stay_low = 0.97
    p_stay_high = 0.93
    regime = np.zeros(n_obs, dtype=int)
    regime[0] = 0  # start in low-vol regime
    vol_level = [0.20, 0.60]   # annualised vol in each regime (percent)
    vol_slope = [0.10, 0.35]

    for t in range(1, n_obs):
        if regime[t - 1] == 0:
            regime[t] = 0 if rng.random() < p_stay_low else 1
        else:
            regime[t] = 1 if rng.random() < p_stay_high else 0

    # Initialise around historical mean
    level[0] = 6.0
    slope[0] = 1.5
    for t in range(1, n_obs):
        v_l = vol_level[regime[t]]
        v_s = vol_slope[regime[t]]
        level[t] = rho_level * level[t - 1] + (1 - rho_level) * 6.0 + v_l * rng.standard_normal()
        slope[t] = rho_slope * slope[t - 1] + (1 - rho_slope) * 1.5 + v_s * rng.standard_normal()

    # Term-structure factor loadings (approximate Nelson-Siegel style)
    # b_τ (level loading) increases with maturity; c_τ (slope loading) decreases
    loadings = {
        "1m":  (0.60, 1.00),   # (b_level, b_slope)
        "12m": (0.85, 0.60),
        "36m": (0.95, 0.25),
        "72m": (1.00, 0.05),
    }
    idio_vol = {"1m": 0.15, "12m": 0.10, "36m": 0.08, "72m": 0.06}

    data = {}
    for mat_label, (b_l, b_s) in loadings.items():
        eps = rng.normal(0, idio_vol[mat_label], size=n_obs)
        y = b_l * level + b_s * slope + eps
        # Clip to plausible yield range [0, 20] percent
        data[mat_label] = np.clip(y, 0.01, 20.0)

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_treasury_yields(
    maturities: List[int] = [1, 12, 36, 72],
    start_date: str = "1961-06-01",
    end_date: str = "2024-12-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load U.S. Treasury zero-coupon yields (Liu & Wu 2021, monthly, annualised %).

    Attempts sources in order:
      1. Local cache file data/raw/treasury_yields.csv
      2. Live download from public GitHub mirror (LIU_WU_URL)
      3. Synthetic fallback with correct statistical properties

    Parameters
    ----------
    maturities : list of maturities in months to return
    start_date : start date for filtering (ISO 8601 string)
    end_date   : end date for filtering (ISO 8601 string)
    use_cache  : if True, read from / write to local cache

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns named e.g. '1m', '12m', etc.
    """
    col_map = {m: _MATURITY_LABELS.get(m, f"{m}m") for m in maturities}

    # ------------------------------------------------------------------ #
    # Source 1: local cache
    # ------------------------------------------------------------------ #
    if use_cache and CACHE_FILE.exists():
        try:
            df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
            df = _filter_dates(df, start_date, end_date)
            missing = [c for c in col_map.values() if c not in df.columns]
            if not missing:
                return df[[col_map[m] for m in maturities]]
        except Exception as exc:
            warnings.warn(f"Cache read failed ({exc}); trying remote source.")

    # ------------------------------------------------------------------ #
    # Source 2: download from GitHub
    # ------------------------------------------------------------------ #
    try:
        import requests  # noqa: PLC0415

        resp = requests.get(LIU_WU_URL, timeout=15)
        resp.raise_for_status()
        from io import StringIO

        raw = pd.read_csv(StringIO(resp.text))
        df = _parse_liu_wu(raw, col_map)
        df = _filter_dates(df, start_date, end_date)

        if use_cache:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(CACHE_FILE)

        return df[[col_map[m] for m in maturities]]

    except Exception as exc:
        warnings.warn(
            f"Remote download failed ({exc}); using synthetic fallback data. "
            "Results will NOT match empirical paper findings."
        )

    # ------------------------------------------------------------------ #
    # Source 3: synthetic fallback
    # ------------------------------------------------------------------ #
    df = _generate_synthetic_yields(n_obs=_N_OBS, start_date=_START_DATE)
    df = _filter_dates(df, start_date, end_date)
    available_cols = [col_map[m] for m in maturities if col_map[m] in df.columns]
    return df[available_cols]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Restrict DataFrame to the [start, end] date range."""
    df = df[df.index >= pd.Timestamp(start)]
    df = df[df.index <= pd.Timestamp(end)]
    return df


def _parse_liu_wu(raw: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """
    Attempt to standardise a raw Liu-Wu CSV into the expected format.
    The exact column names depend on the data file layout; this function
    tries common conventions and falls back gracefully.
    """
    # Try to find a date column
    date_col = None
    for candidate in ('date', 'Date', 'DATE', 'yyyymm', 'yearmon'):
        if candidate in raw.columns:
            date_col = candidate
            break

    if date_col is None:
        # Assume first column is the date
        date_col = raw.columns[0]

    raw = raw.copy()
    raw[date_col] = pd.to_datetime(raw[date_col].astype(str), errors='coerce')
    raw = raw.dropna(subset=[date_col])
    raw = raw.set_index(date_col)
    raw.index.name = 'date'

    # Map numeric maturity columns (Liu-Wu stores maturities as integers or
    # strings like '1', '3', '6', '12', '24', '36', '60', '72', '84', '120')
    rename = {}
    for m, label in col_map.items():
        for candidate in (str(m), f"{m}m", f"y{m}", f"m{m}"):
            if candidate in raw.columns:
                rename[candidate] = label
                break

    if rename:
        raw = raw.rename(columns=rename)

    return raw


def _build_change_series(
    df: pd.DataFrame,
    maturity: int = 12,
    n_lags: int = 1,
) -> np.ndarray:
    """
    Construct a yield-change series for empirical analysis.

    Parameters
    ----------
    df       : output of load_treasury_yields
    maturity : maturity in months
    n_lags   : differencing order (1 = monthly change)

    Returns
    -------
    (T,) array of yield changes (NaNs dropped)
    """
    label = _MATURITY_LABELS.get(maturity, f"{maturity}m")
    if label not in df.columns:
        raise KeyError(f"Maturity column '{label}' not found. Available: {list(df.columns)}")
    series = df[label].diff(n_lags).dropna()
    return series.values
