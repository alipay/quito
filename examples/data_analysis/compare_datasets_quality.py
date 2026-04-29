#!/usr/bin/env python3
"""
Compare Multiple Datasets - Quality Analysis

This script demonstrates how to compare the quality of multiple time series datasets
using QUITO's composite QualityScore metric.

QualityScore combines:
- Forecastability (45%): How predictable the series is
- Seasonality (25%): Strength of seasonal patterns
- Completeness (15%): 1 - missing ratio
- Length (15%): Normalized by longest dataset

Usage:
    python examples/data_analysis/compare_datasets_quality.py

Dependencies:
    Core: numpy, scipy
    Optional: statsmodels, matplotlib
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np

from quito.utils.dataset_quality import (
    compare_datasets,
    print_comparison,
    plot_forecastability_cdf
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_datasets():
    """
    Generate synthetic datasets with different quality characteristics.
    """
    np.random.seed(42)
    
    datasets = {}
    
    # Dataset 1: High quality - strong patterns, no missing data
    logger.info("Generating 'HighQuality' dataset...")
    high_quality = []
    for _ in range(30):
        length = 1000
        t = np.arange(length)
        trend = 0.02 * t
        seasonal = 15 * np.sin(2 * np.pi * t / 24)
        noise = np.random.randn(length) * 2
        series = trend + seasonal + noise
        high_quality.append(series)
    datasets["HighQuality"] = high_quality
    
    # Dataset 2: Medium quality - moderate patterns, some missing
    logger.info("Generating 'MediumQuality' dataset...")
    medium_quality = []
    for _ in range(30):
        length = 800
        t = np.arange(length)
        trend = 0.01 * t
        seasonal = 8 * np.sin(2 * np.pi * t / 24)
        noise = np.random.randn(length) * 4
        series = trend + seasonal + noise
        # Add 5% missing
        missing_idx = np.random.choice(length, size=int(0.05 * length), replace=False)
        series[missing_idx] = np.nan
        medium_quality.append(series)
    datasets["MediumQuality"] = medium_quality
    
    # Dataset 3: Low quality - weak patterns, lots of noise
    logger.info("Generating 'LowQuality' dataset...")
    low_quality = []
    for _ in range(30):
        length = 600
        t = np.arange(length)
        trend = 0.005 * t
        seasonal = 3 * np.sin(2 * np.pi * t / 24)  # Weak seasonality
        noise = np.random.randn(length) * 10  # High noise
        series = trend + seasonal + noise
        # Add 15% missing
        missing_idx = np.random.choice(length, size=int(0.15 * length), replace=False)
        series[missing_idx] = np.nan
        low_quality.append(series)
    datasets["LowQuality"] = low_quality
    
    # Dataset 4: Random walk - no predictable patterns
    logger.info("Generating 'RandomWalk' dataset...")
    random_walk = []
    for _ in range(30):
        length = 700
        series = np.cumsum(np.random.randn(length))
        random_walk.append(series)
    datasets["RandomWalk"] = random_walk
    
    # Dataset 5: Strong seasonality, short series
    logger.info("Generating 'ShortSeasonal' dataset...")
    short_seasonal = []
    for _ in range(30):
        length = 300  # Shorter series
        t = np.arange(length)
        seasonal = 20 * np.sin(2 * np.pi * t / 24)  # Very strong seasonality
        noise = np.random.randn(length) * 1
        series = seasonal + noise + 50
        short_seasonal.append(series)
    datasets["ShortSeasonal"] = short_seasonal
    
    return datasets


def example_compare_datasets():
    """Main example: Compare multiple datasets."""
    
    logger.info("="*90)
    logger.info("QUITO: Cross-Dataset Quality Comparison")
    logger.info("="*90)
    
    # Generate synthetic datasets
    datasets = generate_synthetic_datasets()
    
    logger.info(f"\nComparing {len(datasets)} datasets...")
    for name, series_list in datasets.items():
        logger.info(f"  {name}: {len(series_list)} series")
    
    # Compare datasets
    logger.info("\nComputing quality metrics...")
    summaries = compare_datasets(
        datasets,
        period=24,  # Daily seasonality
        compute_adf=False
    )
    
    # Print comparison table
    print_comparison(summaries, sort_by="QualityScore")
    
    # Detailed analysis
    logger.info("\nDetailed Analysis:")
    logger.info("-"*90)
    
    for name in sorted(summaries, key=lambda k: summaries[k]["QualityScore"], reverse=True):
        s = summaries[name]
        logger.info(f"\n{name}:")
        logger.info(f"  QualityScore:     {s['QualityScore']:.4f}")
        logger.info(f"  Forecastability:  {s['forecastability_med']:.4f}")
        season = s.get('season_strength_med', float('nan'))
        if not np.isnan(season):
            logger.info(f"  Seasonality:      {season:.4f}")
        else:
            logger.info(f"  Seasonality:      N/A")
        logger.info(f"  Missing:          {s['missing_med']*100:.2f}%")
        logger.info(f"  Median Length:    {s['length_med']:.0f}")
        logger.info(f"  Number of Series: {s['n_series']}")
    
    return datasets, summaries


def example_plot_comparison(datasets, summaries):
    """Generate visualizations comparing datasets."""
    
    logger.info("\n" + "="*90)
    logger.info("Generating Forecastability CDF Plot")
    logger.info("="*90)
    
    output_path = Path(__file__).parent / "dataset_quality_comparison.png"
    
    try:
        plot_forecastability_cdf(
            datasets,
            period=24,
            save_path=str(output_path)
        )
        logger.info(f"✓ Plot saved to: {output_path}")
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")
        logger.info("Install matplotlib to enable plotting: pip install matplotlib")


def example_ranking_interpretation():
    """Interpret quality scores for decision making."""
    
    logger.info("\n" + "="*90)
    logger.info("Quality Score Interpretation Guide")
    logger.info("="*90)
    
    logger.info("""
QualityScore Ranges:
  
  0.80 - 1.00: Excellent
    - Highly forecastable with strong patterns
    - Minimal missing data
    - Good for training large models
    - Expected forecast accuracy: High
  
  0.60 - 0.80: Good
    - Moderate to good forecastability
    - Some missing data tolerable
    - Suitable for most forecasting tasks
    - Expected forecast accuracy: Moderate to High
  
  0.40 - 0.60: Fair
    - Limited predictability
    - May have data quality issues
    - Consider data preprocessing or augmentation
    - Expected forecast accuracy: Moderate
  
  0.00 - 0.40: Poor
    - Weak patterns or high noise
    - Significant data quality issues
    - May not be suitable for standard forecasting
    - Expected forecast accuracy: Low
    
Recommended Actions by Score:
  
  > 0.7:  Use as primary training data
  0.5-0.7: Use with data augmentation or preprocessing
  < 0.5:  Investigate data quality issues, consider alternative approaches
""")


def main():
    """Run all comparison examples."""
    
    # Example 1: Compare datasets
    datasets, summaries = example_compare_datasets()
    
    # Example 2: Plot comparison
    example_plot_comparison(datasets, summaries)
    
    # Example 3: Interpretation guide
    example_ranking_interpretation()
    
    logger.info("\n" + "="*90)
    logger.info("Dataset Comparison Examples Completed!")
    logger.info("="*90)
    logger.info("\nKey Takeaways:")
    logger.info("  1. QualityScore helps rank datasets for forecasting suitability")
    logger.info("  2. Higher forecastability = more predictable patterns")
    logger.info("  3. Missing data significantly impacts quality")
    logger.info("  4. Use quality metrics to guide data selection and preprocessing")
    logger.info("\nNext steps:")
    logger.info("  1. Analyze your own datasets with compare_datasets()")
    logger.info("  2. Use quality scores to prioritize data for model training")
    logger.info("  3. Investigate low-quality datasets for improvement opportunities")


if __name__ == "__main__":
    main()

