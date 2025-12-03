"""
QUITO: Unified Time Series Dataset Quality Toolkit

This module provides comprehensive tools for evaluating time series dataset quality:
- Per-series metrics: forecastability, seasonality strength, missingness, effective length, CV, ADF
- Dataset-level summaries: weighted and unweighted aggregations
- Cross-dataset comparison: QualityScore composite metric for ranking datasets

References:
- Large-Time-Series-Model repo: https://github.com/thuml/Large-Time-Series-Model
- Timer/UTSD paper: https://arxiv.org/pdf/2402.02368

Core Metrics:
- Forecastability: 1 - spectral_entropy_welch (normalized to [0,1])
- Seasonality strength: from STL decomposition (if period provided)
- Missingness: fraction of NaN values
- Effective length: count of non-NaN values
- QualityScore: 0.45*Forecast + 0.25*Season + 0.15*(1-Missing) + 0.15*LengthNorm
"""

import os
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union
import warnings
from scipy.signal import welch

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("pandas not available. Parquet loading will fail.")

# Setup logger
logger = logging.getLogger(__name__)


# --- Optional deps
try:
    from statsmodels.tsa.seasonal import STL
    STL_AVAILABLE = True
except Exception:
    STL_AVAILABLE = False
    warnings.warn("statsmodels not available. Seasonality strength won't be computed. pip install statsmodels")

try:
    from arch.unitroot import ADF
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False
    warnings.warn("arch not available. ADF won't be computed. pip install arch")

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

_EPS = 1e-12
ArrayLike = Union[np.ndarray, List[float]]


# ------------------------- Hurst Exponent -------------------------

def hurst_exponent(x: np.ndarray, min_window: int = 20, max_window: Optional[int] = None) -> float:
    """
    Compute Hurst exponent using R/S (Rescaled Range) analysis.
    
    The Hurst exponent H measures long-range dependence:
    - H < 0.5: Anti-persistent (mean-reverting)
    - H = 0.5: Random walk (no memory)  
    - H > 0.5: Persistent (trending)
    
    Args:
        x: 1D time series array
        min_window: Minimum window size for R/S calculation
        max_window: Maximum window size (default: len(x) // 2)
    
    Returns:
        Hurst exponent in [0, 1], or NaN if computation fails
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    
    n = len(x)
    if n < min_window * 4:
        return np.nan
    
    if max_window is None:
        max_window = n // 2
    
    max_window = min(max_window, n // 2)
    if max_window < min_window:
        return np.nan
    
    # Generate window sizes - use factors of n for better coverage
    # Start from min_window and go up by multiplying by ~sqrt(2)
    window_sizes = []
    divisors = [d for d in range(min_window, max_window + 1) if n % d == 0 or n // d >= 2]
    
    if len(divisors) < 3:
        # Fallback: use geometric progression
        size = min_window
        while size <= max_window:
            window_sizes.append(int(size))
            size *= 1.4  # ~sqrt(2)
        window_sizes = list(set(window_sizes))  # Remove duplicates
    else:
        window_sizes = divisors
    
    window_sizes = sorted(window_sizes)
    if len(window_sizes) < 3:
        return np.nan
    
    rs_values = []
    for ws in window_sizes:
        rs_list = []
        num_windows = n // ws
        
        if num_windows < 1:
            continue
            
        for i in range(num_windows):
            chunk = x[i * ws:(i + 1) * ws]
            if len(chunk) < ws:
                continue
                
            mean_chunk = np.mean(chunk)
            
            # Cumulative deviation from mean
            y = np.cumsum(chunk - mean_chunk)
            
            # Range
            r = np.max(y) - np.min(y)
            
            # Standard deviation (use ddof=0 for population std)
            s = np.std(chunk, ddof=0)
            
            if s > _EPS and r > 0:
                rs_list.append(r / s)
        
        if rs_list:
            rs_values.append((ws, np.mean(rs_list)))
    
    if len(rs_values) < 3:
        return np.nan
    
    # Linear regression in log-log space: log(R/S) = H * log(n) + c
    log_n = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])
    
    # Simple linear regression with proper normalization
    try:
        # Use least squares
        A = np.vstack([log_n, np.ones(len(log_n))]).T
        slope, _ = np.linalg.lstsq(A, log_rs, rcond=None)[0]
        # Clamp to [0, 1] range
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return np.nan


# ------------------------- Utilities -------------------------

def _to_1d_numpy(x: ArrayLike) -> np.ndarray:
    """Accept numpy array, list, or torch tensor and return a 1D float numpy array."""
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=float).squeeze()
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {x.shape}")
    return x


def spectral_entropy_welch(x: np.ndarray, fs: float = 1.0, nperseg: Optional[int] = None) -> float:
    """
    Normalized spectral entropy in [0,1] using Welch's method.
    
    Returns:
        0.0 for constant/zero-variance series (perfectly predictable)
        1.0 for maximum entropy (white noise)
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(np.isnan(x)):
        return 1.0  # maximal entropy
    
    # Check for constant series (zero variance)
    if np.std(x) < _EPS:
        return 0.0  # zero entropy = perfectly predictable
    
    x = x - np.nanmean(x)
    x = np.nan_to_num(x, nan=0.0)
    if nperseg is None:
        nperseg = max(64, min(len(x), 1024))
    _, Pxx = welch(x, fs=fs, nperseg=nperseg, detrend='constant', scaling='density')
    Pxx = Pxx + _EPS
    Pxx /= Pxx.sum()
    H = -(Pxx * np.log(Pxx)).sum()
    Hmax = np.log(len(Pxx))
    return float(H / Hmax)


def forecastability_welch(
    x: np.ndarray,
    fs: float = 1.0,
    nperseg: Optional[int] = None,
    window: Optional[int] = None
) -> float:
    """
    Forecastability in [0,1] = 1 - spectral_entropy_welch.
    If 'window' is provided, compute mean forecastability over non-overlapping chunks of length 'window'
    (last chunk can be shorter).
    """
    x = np.asarray(x, dtype=float)
    if window is None or window <= 0 or window >= len(x):
        se = spectral_entropy_welch(x, fs=fs, nperseg=nperseg)
        return float(max(0.0, min(1.0, 1.0 - se)))

    vals = []
    n = len(x)
    k = max(n - window, 0) // window + 1
    for i in range(k):
        s, e = i * window, min((i + 1) * window, n)
        se = spectral_entropy_welch(x[s:e], fs=fs, nperseg=nperseg)
        vals.append(max(0.0, min(1.0, 1.0 - se)))
    return float(np.nanmean(vals)) if vals else np.nan


def stl_decomposition_strengths(x: np.ndarray, period: Optional[int]) -> tuple:
    """
    Compute both seasonality and trend strength from a single STL decomposition.
    
    Seasonality strength = 1 - Var(resid) / Var(seasonal + resid) in [0,1]
    Trend strength = 1 - Var(resid) / Var(trend + resid) in [0,1]
    
    Returns:
        (seasonality_strength, trend_strength) tuple, or (NaN, NaN) if computation fails
    """
    if not STL_AVAILABLE or not period or period < 2 or period >= len(x):
        return np.nan, np.nan
    if np.all(np.isnan(x)):
        return np.nan, np.nan
    x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
    try:
        res = STL(x, period=period, robust=True).fit()
        resid = res.resid
        seas = res.seasonal
        trend = res.trend
        
        # Seasonality strength
        seas_den = np.var(seas + resid) + _EPS
        seas_strength = float(max(0.0, 1.0 - np.var(resid) / seas_den))
        
        # Trend strength
        trend_den = np.var(trend + resid) + _EPS
        trend_strength = float(max(0.0, 1.0 - np.var(resid) / trend_den))
        
        return seas_strength, trend_strength
    except Exception:
        return np.nan, np.nan


def seasonality_strength_stl(x: np.ndarray, period: Optional[int]) -> float:
    """
    Seasonality strength = 1 - Var(resid) / Var(seasonal + resid) in [0,1].
    Returns NaN if period invalid or statsmodels unavailable.
    
    Note: For computing both seasonality and trend, use stl_decomposition_strengths() instead.
    """
    seas, _ = stl_decomposition_strengths(x, period)
    return seas


def trend_strength_stl(x: np.ndarray, period: Optional[int]) -> float:
    """
    Trend strength = 1 - Var(resid) / Var(trend + resid) in [0,1].
    Measures how strong the trend component is relative to residual noise.
    - Close to 1 → strong trend
    - Close to 0 → little or no trend
    
    Note: For computing both seasonality and trend, use stl_decomposition_strengths() instead.
    """
    _, trend = stl_decomposition_strengths(x, period)
    return trend


def missing_ratio(x: np.ndarray) -> float:
    return float(np.mean(np.isnan(x))) if x.size else 1.0


def effective_length(x: np.ndarray) -> int:
    return int(np.sum(~np.isnan(x)))


def coefficient_of_variation(x: np.ndarray) -> float:
    """
    CV = std / |mean| on non-NaN values; returns inf if mean=0 and std>0; 0 if all constant.
    """
    y = x[~np.isnan(x)]
    if y.size == 0:
        return np.nan
    m = float(np.mean(y))
    s = float(np.std(y))
    if m == 0.0:
        return float('inf') if s > 0 else 0.0
    return float(s / abs(m))


def fill_missing(x: np.ndarray, how: str = 'none') -> np.ndarray:
    """
    Fill missing values: 'none' | 'zero' | 'mean' | 'forward' | 'backward'
    """
    x = np.asarray(x, dtype=float).copy()
    if how == 'none':
        return x
    if how == 'zero':
        x[np.isnan(x)] = 0.0
        return x
    if how == 'mean':
        m = np.nanmean(x)
        x[np.isnan(x)] = 0.0 if np.isnan(m) else m
        return x
    if how == 'forward':
        mask = np.isnan(x)
        idx = np.where(~mask, np.arange(len(x)), 0)
        np.maximum.accumulate(idx, out=idx)
        x[mask] = x[idx[mask]]
        x[np.isnan(x)] = 0.0
        return x
    if how == 'backward':
        mask = np.isnan(x)
        idx = np.where(~mask, np.arange(len(x)), len(x) - 1)
        # Backward fill: work from end to beginning
        for i in range(len(idx) - 2, -1, -1):
            if mask[i]:
                idx[i] = idx[i + 1]
        x[mask] = x[idx[mask]]
        x[np.isnan(x)] = 0.0
        return x
    raise ValueError(f"Unknown fill method: {how}")


# ------------------------- Per-series & per-dataset -------------------------

@dataclass
class SeriesQuality:
    forecastability: float
    season_strength: float
    trend_strength: float  # Trend strength from STL decomposition
    missing_ratio: float
    eff_length: int
    cv: float
    adf_stat: float  # may be np.nan
    hurst: float  # Hurst exponent, may be np.nan

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def evaluate_series(
    x: ArrayLike,
    period: Optional[int] = None,
    fs: float = 1.0,
    compute_adf: bool = False,
    compute_hurst: bool = True,
    adf_fill: str = 'mean',
    forecast_window: Optional[int] = None,
    fill_for_metrics: str = 'none'
) -> SeriesQuality:
    """
    Unified per-series evaluation (QUITO).
    - Missing ratio & effective length computed on the raw series.
    - Forecastability computed on series with NaNs set to 0 after centering (via Welch subroutine).
      Optional windowing averages forecastability across non-overlapping chunks.
    - Seasonality strength via STL on series with NaNs filled by median (internal).
    - CV computed on series with chosen fill (fill_for_metrics).
    - ADF (optional) uses 'arch' and is applied to filled series (adf_fill).
    - Hurst exponent (optional) measures long-range dependence.
    """
    x = _to_1d_numpy(x)

    miss = missing_ratio(x)
    eff = effective_length(x)

    x_cv = fill_missing(x, how=fill_for_metrics)
    cv = coefficient_of_variation(x_cv)

    # Handle constant series (cv=0) gracefully
    if cv == 0.0:
        # Constant series: no forecastability (predictable=1? or entropy=0?), no seasonality, stationary
        # Entropy of constant series is 0 -> forecastability = 1.0
        # Hurst = 0.5 for constant (no memory)
        return SeriesQuality(
            forecastability=1.0,  # Perfectly predictable
            season_strength=0.0,  # No seasonality
            trend_strength=0.0,   # No trend
            missing_ratio=miss,
            eff_length=eff,
            cv=0.0,
            adf_stat=-99.9,  # Indicates strong stationarity (constant)
            hurst=0.5  # No memory for constant series
        )

    fcast = forecastability_welch(x, fs=fs, window=forecast_window)
    seas, trend = stl_decomposition_strengths(x, period)  # Single STL computation

    adf_val = np.nan
    if compute_adf and ARCH_AVAILABLE:
        try:
            x_adf = fill_missing(x, how=adf_fill)
            adf_val = float(ADF(x_adf).stat)
        except Exception as e:
            # Suppress common ADF errors for short/constant series to reduce log noise
            if "singular regressor matrix" not in str(e):
                logger.warning(f"ADF test failed for series: {e}")
            adf_val = np.nan

    # Compute Hurst exponent
    hurst_val = np.nan
    if compute_hurst:
        x_filled = fill_missing(x, how='mean')
        hurst_val = hurst_exponent(x_filled)

    return SeriesQuality(
        forecastability=fcast,
        season_strength=seas,
        trend_strength=trend,
        missing_ratio=miss,
        eff_length=eff,
        cv=cv,
        adf_stat=adf_val,
        hurst=hurst_val
    )


def summarize_dataset_medians(
    series_list: List[ArrayLike],
    period: Optional[int] = None,
    fs: float = 1.0,
    compute_adf: bool = False,
    compute_hurst: bool = True,
    forecast_window: Optional[int] = None
) -> Dict[str, float]:
    """
    QUITO medians (robust, simple).
    """
    rows = [evaluate_series(s, period=period, fs=fs, compute_adf=compute_adf,
                            compute_hurst=compute_hurst,
                            forecast_window=forecast_window) for s in series_list]
    F = np.array([r.forecastability for r in rows], float)
    S = np.array([r.season_strength for r in rows], float)
    T = np.array([r.trend_strength for r in rows], float)
    M = np.array([r.missing_ratio for r in rows], float)
    L = np.array([r.eff_length for r in rows], float)
    A = np.array([r.adf_stat for r in rows], float)
    H = np.array([r.hurst for r in rows], float)

    med = lambda a: float(np.nanmedian(a)) if a.size else float('nan')
    out = {
        "forecastability_med": med(F),
        "season_strength_med": med(S),
        "trend_strength_med": med(T),
        "missing_med": med(M),
        "length_med": med(L),
        "hurst_med": med(H),
        "n_series": len(series_list),
    }
    if compute_adf:
        out["adf_stat_med"] = med(A)
    return out


def evaluate_dataset(
    series_list: List[ArrayLike],
    period: Optional[int] = None,
    fs: float = 1.0,
    compute_adf: bool = False,
    compute_hurst: bool = True,
    forecast_window: Optional[int] = None,
    fill_for_metrics: str = 'none',
    verbose: bool = True
) -> Dict[str, Union[float, Dict]]:
    """
    Weighted & unweighted dataset summaries + totals (QUITO).
    """
    lengths, fcasts, cvs, missings, adfs, hursts, trends = [], [], [], [], [], [], []
    num_failed = 0

    for i, s in enumerate(series_list):
        if verbose and (i + 1) % 100 == 0:
            logger.info(f"Evaluated {i + 1}/{len(series_list)} series...")
        try:
            r = evaluate_series(
                s, period=period, fs=fs, compute_adf=compute_adf,
                compute_hurst=compute_hurst,
                forecast_window=forecast_window, fill_for_metrics=fill_for_metrics
            )
            lengths.append(r.eff_length)
            fcasts.append(r.forecastability)
            cvs.append(r.cv)
            missings.append(r.missing_ratio)
            hursts.append(r.hurst)
            trends.append(r.trend_strength)
            if compute_adf:
                adfs.append(r.adf_stat)
        except Exception as e:
            logger.warning(f"Series {i} failed evaluation: {e}")
            num_failed += 1
            continue

    L = np.asarray(lengths, float)
    F = np.asarray(fcasts, float)
    C = np.asarray(cvs, float)
    M = np.asarray(missings, float)
    H = np.asarray(hursts, float)
    T = np.asarray(trends, float)
    A = np.asarray(adfs, float) if compute_adf else np.array([])

    total_points = float(np.nansum(L)) if L.size else 0.0
    if total_points <= 0:
        total_points = 1.0  # avoid div by zero

    def wavg(x, w):
        x = np.asarray(x, float)
        w = np.asarray(w, float)
        mask = ~np.isnan(x)
        if not mask.any():
            return float('nan')
        return float(np.nansum(x[mask] * w[mask]) / np.nansum(w[mask]))

    weighted = {
        "forecastability": wavg(F, L),
        "cv": wavg(C, L),
        "missing_ratio": wavg(M, L),
        "hurst": wavg(H, L),
        "trend_strength": wavg(T, L),
    }
    if compute_adf and A.size:
        weighted["adf_stat"] = wavg(A, L[:len(A)])

    unweighted = {
        "forecastability": float(np.nanmean(F)) if F.size else float('nan'),
        "cv": float(np.nanmean(C)) if C.size else float('nan'),
        "missing_ratio": float(np.nanmean(M)) if M.size else float('nan'),
        "hurst": float(np.nanmean(H)) if H.size else float('nan'),
        "trend_strength": float(np.nanmean(T)) if T.size else float('nan'),
    }
    if compute_adf and A.size:
        unweighted["adf_stat"] = float(np.nanmean(A))

    return {
        "weighted_metrics": weighted,
        "unweighted_metrics": unweighted,
        "total_time_points": int(np.nansum(L)) if L.size else 0,
        "num_series": len(series_list),
        "avg_series_length": float(np.nanmean(L)) if L.size else float('nan'),
        "num_failed": int(num_failed),
    }


# ------------------------- Cross-dataset comparison (QualityScore) -------------------------

def compare_datasets(
    datasets: Dict[str, List[ArrayLike]],
    period: Optional[int] = None,
    fs: float = 1.0,
    compute_adf: bool = False,
    forecast_window: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple datasets with QUITO medians + composite QualityScore.
      QualityScore = 0.45*Forecast_med + 0.25*Season_med + 0.15*(1-Missing_med) + 0.15*LengthNorm
    LengthNorm uses log(median_length)/log(max_median_length_across_datasets).
    """
    summaries = {
        name: summarize_dataset_medians(lst, period=period, fs=fs,
                                        compute_adf=compute_adf, forecast_window=forecast_window)
        for name, lst in datasets.items()
    }

    med_lengths = np.array([d["length_med"] for d in summaries.values()], float)
    Lmax = float(np.nanmax(med_lengths)) if med_lengths.size else 1.0

    for name, d in summaries.items():
        L = max(1.0, d["length_med"])
        d["length_norm"] = float(min(1.0, np.log(L) / (np.log(Lmax) if Lmax > 1 else 1.0)))
        F = d["forecastability_med"]
        S = d["season_strength_med"]
        I = 1.0 - d["missing_med"]
        S_eff = 0.0 if np.isnan(S) else S
        d["QualityScore"] = 0.45 * F + 0.25 * S_eff + 0.15 * I + 0.15 * d["length_norm"]
    return summaries


def print_comparison(summaries: Dict[str, Dict[str, float]], sort_by: str = "QualityScore") -> None:
    """
    Nicely formatted table for compare_datasets output (QUITO).
    """
    order = sorted(summaries, key=lambda k: summaries[k].get(sort_by, 0.0), reverse=True)
    logger.info("\n" + "="*90)
    logger.info("QUITO: Dataset Quality Comparison")
    logger.info("="*90)
    logger.info(f"\n{'Dataset':<20} {'QualityScore':<12} {'Forecast':<10} {'Season':<10} {'Missing%':<10} {'Length':<10}")
    logger.info("-"*90)
    for name in order:
        s = summaries[name]
        seas = "N/A" if np.isnan(s.get("season_strength_med", np.nan)) else f"{s['season_strength_med']:.4f}"
        logger.info(f"{name:<20} "
                    f"{s.get('QualityScore', float('nan')):<12.4f} "
                    f"{s['forecastability_med']:<10.4f} "
                    f"{seas:<10} "
                    f"{s['missing_med']*100:<10.2f} "
                    f"{s['length_med']:<10.0f}")
    logger.info("="*90)
    logger.info("QualityScore = 0.45*Forecast + 0.25*Season + 0.15*(1-Missing) + 0.15*LengthNorm\n")


# ------------------------- Baseline & plotting -------------------------

def seasonal_naive_smape(x: ArrayLike, period: int, n_test: Optional[int] = None) -> float:
    """
    sMAPE (%) for seasonal naive: y_hat[t] = y[t-period]
    """
    x = _to_1d_numpy(x)
    x = x[~np.isnan(x)]
    if x.size < period * 2:
        return np.nan
    if n_test is None:
        n_test = max(period, int(0.2 * len(x)))

    train = x[:-n_test]
    test = x[-n_test:]
    preds = []
    for i in range(len(test)):
        idx = len(train) + i - period
        preds.append(x[idx] if idx >= 0 else train[0])
    preds = np.asarray(preds)
    denom = (np.abs(test) + np.abs(preds)) / 2.0 + _EPS
    return float(100.0 * np.mean(np.abs(test - preds) / denom))


def plot_forecastability_cdf(
    datasets: Dict[str, List[ArrayLike]],
    period: Optional[int] = None,
    fs: float = 1.0,
    forecast_window: Optional[int] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot CDF of per-series forecastability for multiple datasets.
    """
    if not MPL_AVAILABLE:
        logger.warning("matplotlib not available. pip install matplotlib")
        return

    plt.figure(figsize=(10, 6))
    for name, series in datasets.items():
        scores = []
        for s in series:
            r = evaluate_series(s, period=period, fs=fs, forecast_window=forecast_window)
            scores.append(r.forecastability)
        scores = np.asarray(scores, float)
        scores = scores[~np.isnan(scores)]
        scores.sort()
        if scores.size == 0:
            continue
        cdf = np.arange(1, len(scores) + 1) / len(scores)
        plt.plot(scores, cdf, label=name, linewidth=2)

    plt.xlabel("Forecastability", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.title("Forecastability Distribution Across Datasets", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Forecastability CDF plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


# ------------------------- Reporting & I/O helpers -------------------------

def print_dataset_report(results: Dict) -> None:
    """
    Console report (QUITO).
    """
    logger.info("\n" + "="*70)
    logger.info("QUITO: DATASET QUALITY EVALUATION REPORT")
    logger.info("="*70)
    logger.info("\nDataset Statistics:")
    logger.info(f"  Total Time Points: {results.get('total_time_points', 0):,}")
    logger.info(f"  Number of Series:  {results.get('num_series', 0):,}")
    logger.info(f"  Average Length:    {results.get('avg_series_length', float('nan')):.1f}")
    if "num_failed" in results:
        logger.info(f"  Failed Evaluations:{results.get('num_failed', 0)}")

    logger.info("\nWeighted Metrics:")
    for k, v in results.get("weighted_metrics", {}).items():
        logger.info(f"  {k:>15}: {v:.4f}")

    logger.info("\nUnweighted Metrics:")
    for k, v in results.get("unweighted_metrics", {}).items():
        logger.info(f"  {k:>15}: {v:.4f}")

    # Simple interpretations
    fc = results.get("weighted_metrics", {}).get("forecastability", np.nan)
    if not np.isnan(fc):
        if fc > 0.7:
            logger.info(f"\n  ✓ High forecastability ({fc:.3f})")
        elif fc > 0.5:
            logger.info(f"\n  ~ Moderate forecastability ({fc:.3f})")
        else:
            logger.info(f"\n  ✗ Low forecastability ({fc:.3f})")

    mr = results.get("weighted_metrics", {}).get("missing_ratio", np.nan)
    if not np.isnan(mr):
        if mr < 0.01:
            logger.info(f"  ✓ Excellent data quality ({mr*100:.2f}% missing)")
        elif mr < 0.05:
            logger.info(f"  ~ Good data quality ({mr*100:.2f}% missing)")
        else:
            logger.info(f"  ✗ Poor data quality ({mr*100:.2f}% missing)")

    adf = results.get("weighted_metrics", {}).get("adf_stat", np.nan)
    if not np.isnan(adf):
        if adf < -3.43:
            logger.info(f"  ✓ Strong stationarity (ADF={adf:.2f})")
        elif adf < -2.86:
            logger.info(f"  ~ Moderate stationarity (ADF={adf:.2f})")
        else:
            logger.info(f"  ✗ Weak stationarity (ADF={adf:.2f})")
    logger.info("="*70 + "\n")


def evaluate_dataset_from_file(
    file_path: str,
    period: Optional[int] = None,
    fs: float = 1.0,
    compute_adf: bool = False,
    forecast_window: Optional[int] = None,
    fill_for_metrics: str = 'none',
    verbose: bool = True
) -> Dict:
    """
    Convenience loader:
      - .npy: expects (N,) or (M,N) float arrays
      - .csv/.txt: tries pandas (if available) else numpy genfromtxt
    Each row is treated as a series for 2D inputs; 1D -> single series.
    """
    _, ext = os.path.splitext(file_path.lower())
    data = None

    if ext == ".npy":
        data = np.load(file_path, allow_pickle=False)
    else:
        try:
            import pandas as pd
            df = pd.read_csv(file_path, header=None)
            data = df.values
        except Exception:
            data = np.genfromtxt(file_path, delimiter=",")
    if data.ndim == 1:
        series = [data]
    elif data.ndim == 2:
        series = [data[i] for i in range(data.shape[0])]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

    results = evaluate_dataset(
        series,
        period=period,
        fs=fs,
        compute_adf=compute_adf,
        forecast_window=forecast_window,
        fill_for_metrics=fill_for_metrics,
        verbose=verbose
    )
    if verbose:
        print_dataset_report(results)
    return results


def load_time_series_from_parquet(
    file_path: Union[str, Path],
    value_col: str = 'value',
    max_length: Optional[int] = None,
    max_series: Optional[int] = None,
    sampling_strategy: str = 'random',
    use_all_indices: bool = False
) -> List[np.ndarray]:
    """
    Load parquet file and extract time series with truncation/sampling.
    
    Args:
        file_path: Path to parquet file
        value_col: Name of the value column (default: 'value')
        max_length: Maximum length per series (truncate if longer)
        max_series: Maximum number of series to sample per file
        sampling_strategy: 'random', 'first', 'last', or 'uniform'
        use_all_indices: If True, use all columns starting with 'ind_' as separate series
    
    Returns:
        List of time series arrays
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas is required to load parquet files")

    file_path = Path(file_path)
    logger.info(f"Loading {file_path.name}...")
    df = pd.read_parquet(str(file_path))
    
    # Identify date column
    date_col = None
    for col in ['date_time', 'date', 'datetime', 'timestamp']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        logger.warning(f"No date column found in {file_path.name}, skipping...")
        return []
    
    # Sort by date
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    # Check if multiple series (has item_id)
    if 'item_id' in df.columns:
        series_list = []
        unique_ids = df['item_id'].unique()
        
        # Sample series if needed
        if max_series and len(unique_ids) > max_series:
            if sampling_strategy == 'random':
                np.random.seed(42)  # For reproducibility
                unique_ids = np.random.choice(unique_ids, size=max_series, replace=False)
            elif sampling_strategy == 'first':
                unique_ids = unique_ids[:max_series]
            elif sampling_strategy == 'last':
                unique_ids = unique_ids[-max_series:]
            elif sampling_strategy == 'uniform':
                indices = np.linspace(0, len(unique_ids) - 1, max_series, dtype=int)
                unique_ids = unique_ids[indices]
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
            
            logger.info(f"  Sampling {len(unique_ids)} series from {df['item_id'].nunique()} total")
        
        # Extract series for each item_id
        for item_id in unique_ids:
            item_df = df[df['item_id'] == item_id].sort_values(date_col)
            
            # Determine which columns to use
            target_cols = []
            if use_all_indices:
                target_cols = [c for c in item_df.columns if c.startswith('ind_')]
                if not target_cols:
                    logger.debug(f"  No 'ind_' columns found for item_id={item_id}, falling back to single column")

            if not target_cols:
                # Get value column (try common names)
                if value_col in item_df.columns:
                    target_cols = [value_col]
                elif 'ind_1' in item_df.columns:
                    target_cols = ['ind_1']
                    logger.debug(f"  Using 'ind_1' column for item_id={item_id}")
                else:
                    # Try to find numeric columns
                    numeric_cols = item_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        target_cols = [numeric_cols[0]]
                        logger.debug(f"  Using '{numeric_cols[0]}' column for item_id={item_id}")
                    else:
                        logger.warning(f"  No value column found for item_id={item_id}, skipping...")
                        continue

            for col in target_cols:
                values = item_df[col].values
                
                # Truncate if needed
                if max_length and len(values) > max_length:
                    # Take the most recent max_length points
                    values = values[-max_length:]
                    logger.debug(f"  Truncated series {item_id} (col={col}) from {len(item_df)} to {len(values)} points")
                
                if len(values) > 0:
                    series_list.append(values)
        
        logger.info(f"  Loaded {len(series_list)} series from {file_path.name}")
        return series_list
    
    else:
        # Single series (or wide format with multiple columns)
        series_list = []
        target_cols = []
        
        if use_all_indices:
            target_cols = [c for c in df.columns if c.startswith('ind_')]
        
        if not target_cols:
            if value_col in df.columns:
                target_cols = [value_col]
            elif 'ind_1' in df.columns:
                target_cols = ['ind_1']
                logger.info(f"  Using 'ind_1' column")
            else:
                # Try to find numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_cols = [numeric_cols[0]]
                    logger.info(f"  Using '{numeric_cols[0]}' column")
                else:
                    logger.warning(f"No value column found in {file_path.name}, skipping...")
                    return []
        
        for col in target_cols:
            values = df[col].values
            
            # Truncate if needed
            if max_length and len(values) > max_length:
                values = values[-max_length:]
                logger.debug(f"  Truncated series (col={col}) from {len(df)} to {len(values)} points")
            
            series_list.append(values)

        return series_list
