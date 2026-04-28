import numpy as np

from scipy.signal import find_peaks


def detect_seasonality_fft(y, min_period=2, max_period=None, peak_threshold=0.1):
    """
    Detect dominant seasonality period using Fast Fourier Transform (FFT).
    
    Analyzes the frequency spectrum of a time series to identify the dominant
    seasonal period. Uses FFT to compute frequency components and identifies
    significant peaks within the valid period range.
    
    Args:
        y (np.ndarray): 1D time series array of shape (L_t,).
        min_period (int, optional): Minimum allowed seasonal period. Defaults to 2.
        max_period (int, optional): Maximum allowed seasonal period. If None,
            defaults to L_t // 2. Defaults to None.
        peak_threshold (float, optional): Fraction of maximum amplitude to consider
            a peak significant (0.0 to 1.0). Defaults to 0.1.
    
    Returns:
        int: Detected dominant period (>=2), or 1 if no significant seasonality
            is found.
            
    Example:
        >>> hourly_data = np.sin(2 * np.pi * np.arange(168) / 24)  # 24-hour cycle
        >>> period = detect_seasonality_fft(hourly_data, min_period=2, max_period=48)
        >>> print(f"Detected period: {period} hours")  # Should detect 24
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
    Compute Mean Absolute Error of naive or seasonal naive forecast.
    
    Calculates the MAE of a naive baseline forecast on the training data.
    For m=1, uses simple naive forecast (last value). For m>1, uses seasonal
    naive forecast (value from m steps ago). This metric is commonly used
    as a baseline for evaluating time series models.
    
    Args:
        y_train (np.ndarray): Training time series data.
        m (int, optional): Seasonal period. Use 1 for simple naive forecast,
            or >1 for seasonal naive forecast (e.g., 24 for hourly data with
            daily seasonality). Defaults to 1.
    
    Returns:
        float: Mean Absolute Error of the naive forecast. Returns 1.0 as
            fallback if insufficient data.
            
    Example:
        >>> y_train = np.array([10, 12, 11, 13, 12, 14])
        >>> mae_naive = compute_naive_mae(y_train, m=1)
        >>> mae_seasonal = compute_naive_mae(y_train, m=3)  # 3-step seasonal
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