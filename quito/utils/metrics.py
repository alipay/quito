import numpy as np

from scipy.signal import find_peaks


def detect_seasonality_fft(y, min_period=2, max_period=None, peak_threshold=0.1):
    """
    Detect dominant seasonality period using FFT.
    
    Args:
        y: 1D array (L_t,)
        min_period: minimum allowed period
        max_period: maximum allowed period (default: L_t // 2)
        peak_threshold: fraction of max amplitude to consider significant
    
    Returns:
        m: int, detected period (>=2), or 1 if none found
    """
    y = np.asarray(y)
    L = len(y)
    if L < 2 * min_period:
        return 1
    
    if max_period is None:
        max_period = L // 2
    if max_period < min_period:
        return 1

    # Remove mean
    y_centered = y - np.mean(y)
    
    # FFT
    fft_vals = np.fft.rfft(y_centered)
    freqs = np.fft.rfftfreq(L, d=1)
    
    # Magnitude (skip DC component at freq=0)
    magnitudes = np.abs(fft_vals)[1:]
    valid_freqs = freqs[1:]
    
    if len(magnitudes) == 0:
        return 1

    # Only consider frequencies corresponding to period in [min_period, max_period]
    valid_periods = 1 / valid_freqs
    valid_mask = (valid_periods >= min_period) & (valid_periods <= max_period)
    
    if not np.any(valid_mask):
        return 1

    # Find peak in valid region
    masked_mags = np.where(valid_mask, magnitudes, 0)
    if masked_mags.max() < peak_threshold * magnitudes.max():
        return 1  # No significant peak

    peak_idx = np.argmax(masked_mags)
    dominant_period = round(valid_periods[peak_idx])
    dominant_period = np.clip(dominant_period, min_period, max_period)
    
    return int(dominant_period)


def compute_naive_mae(y_train, m=1):
    """
    Compute MAE of (seasonal) naive forecast on training data.
    Returns scalar.
    """
    y_train = np.asarray(y_train)
    if m == 1:
        if len(y_train) < 2:
            return 1.0  # fallback
        errors = np.abs(np.diff(y_train))
    else:
        if len(y_train) <= m:
            return 1.0
        errors = np.abs(y_train[m:] - y_train[:-m])
        
    return np.mean(errors) if len(errors) > 0 else 1.0