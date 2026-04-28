#!/usr/bin/env python3
"""
Analyze Dataset Quality for open_hour_train Parquet Files

This script analyzes the quality of time series data in the open_hour_train directory.
By default, it processes ALL items with FULL length across ALL indices (ind_1 to ind_5).

Metrics computed:
    - Forecastability: 1 - spectral_entropy (higher = more predictable)
    - Seasonality strength: STL decomposition (higher = stronger periodic patterns)
    - Trend strength: STL decomposition (higher = stronger trend)
    - Hurst exponent: Long-range dependence (H<0.5=mean-reverting, H=0.5=random, H>0.5=trending)
    - CV: Coefficient of variation (variability relative to mean)
    - Missing ratio: Fraction of NaN values
    - Effective length: Number of non-NaN values
    - ADF stat (optional): Augmented Dickey-Fuller test for stationarity

Usage:
    # Analyze all files with default settings (full length, all items, all indices)
    python examples/analyze_open_hour_train_quality.py

    # Custom truncation and sampling for faster analysis
    python examples/analyze_open_hour_train_quality.py \
        --max_length 5000 \
        --max_series_per_file 100 \
        --sampling_strategy random

    # Analyze specific files
    python examples/analyze_open_hour_train_quality.py \
        --files hour_train_hour_p1.parquet hour_train_hour_p2.parquet

    # Disable Hurst exponent computation for speed
    python examples/analyze_open_hour_train_quality.py --no-compute-hurst

Dependencies:
    Core: numpy, scipy, pandas
    Optional: statsmodels (for seasonality), arch (for ADF test)
"""

import sys
import json
import multiprocessing as mp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from glob import glob
from functools import partial

from quito.utils.dataset_quality import (
    evaluate_dataset,
    evaluate_series,
    print_dataset_report,
    compare_datasets,
    print_comparison,
    load_time_series_from_parquet
)
from quito.utils.common import (
    save_json,
    load_json
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def _evaluate_series_worker(args):
    """
    Worker function for multiprocessing.
    Takes a tuple of (series, metadata, period, compute_adf, compute_hurst).
    Returns (metadata, metrics_dict).
    """
    series, meta, period, compute_adf, compute_hurst = args
    metrics = evaluate_series(
        series,
        period=period,
        compute_adf=compute_adf,
        compute_hurst=compute_hurst
    )
    return {
        'item_id': meta['item_id'],
        'index': meta['index'],
        'metrics': metrics.to_dict()
    }


def load_series_with_metadata(
    file_path: Path,
    value_col: str = 'value',
    max_length: Optional[int] = None,
    max_series: Optional[int] = None,
    sampling_strategy: str = 'random',
    use_all_indices: bool = False
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Load series with metadata (item_id, index_name).
    
    Returns:
        Tuple of (series_list, metadata_list) where metadata_list contains
        dicts with 'item_id' and 'index' keys for each series.
    """
    import pandas as pd
    
    file_path = Path(file_path)
    df = pd.read_parquet(str(file_path))
    
    # Identify date column
    date_col = None
    for col in ['date_time', 'date', 'datetime', 'timestamp']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        logger.warning(f"No date column found in {file_path.name}")
        return [], []
    
    # Sort by date
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    
    series_list = []
    metadata_list = []
    
    # Check if multiple series (has item_id)
    if 'item_id' in df.columns:
        unique_ids = df['item_id'].unique()
        
        # Sample series if needed
        max_series_param = None if max_series == 0 else max_series
        if max_series_param and len(unique_ids) > max_series_param:
            if sampling_strategy == 'random':
                np.random.seed(42)
                unique_ids = np.random.choice(unique_ids, size=max_series_param, replace=False)
            elif sampling_strategy == 'first':
                unique_ids = unique_ids[:max_series_param]
            elif sampling_strategy == 'last':
                unique_ids = unique_ids[-max_series_param:]
            elif sampling_strategy == 'uniform':
                indices = np.linspace(0, len(unique_ids) - 1, max_series_param, dtype=int)
                unique_ids = unique_ids[indices]
        
        # Extract series for each item_id
        for item_id in unique_ids:
            item_df = df[df['item_id'] == item_id].sort_values(date_col)
            
            # Determine which columns to use
            target_cols = []
            if use_all_indices:
                target_cols = [c for c in item_df.columns if c.startswith('ind_')]
            
            if not target_cols:
                if value_col in item_df.columns:
                    target_cols = [value_col]
                elif 'ind_1' in item_df.columns:
                    target_cols = ['ind_1']
                else:
                    numeric_cols = item_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        target_cols = [numeric_cols[0]]
                    else:
                        continue
            
            for col in target_cols:
                values = item_df[col].values
                
                # Truncate if needed
                if max_length and len(values) > max_length:
                    values = values[-max_length:]
                
                if len(values) > 0:
                    series_list.append(values)
                    metadata_list.append({
                        'item_id': str(item_id),
                        'index': col
                    })
    else:
        # Single series
        target_cols = []
        if use_all_indices:
            target_cols = [c for c in df.columns if c.startswith('ind_')]
        
        if not target_cols:
            if value_col in df.columns:
                target_cols = [value_col]
            elif 'ind_1' in df.columns:
                target_cols = ['ind_1']
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    target_cols = [numeric_cols[0]]
        
        for col in target_cols:
            values = df[col].values
            if max_length and len(values) > max_length:
                values = values[-max_length:]
            
            if len(values) > 0:
                series_list.append(values)
                metadata_list.append({
                    'item_id': 'single',
                    'index': col
                })
    
    return series_list, metadata_list


def _compute_summary_from_metrics(all_metrics_list: List[Dict], compute_adf: bool = False, compute_hurst: bool = True) -> Dict:
    """
    Compute dataset summary statistics from already-computed per-series metrics.
    This avoids re-evaluating all series when building the summary.
    
    Returns a dict compatible with print_dataset_report(), containing:
    - total_time_points, num_series, avg_series_length
    - weighted_metrics: length-weighted averages
    - unweighted_metrics: simple means
    """
    if not all_metrics_list:
        return {}
    
    # Extract arrays from metrics
    lengths = np.array([m.get('eff_length', 0) for m in all_metrics_list], dtype=float)
    fcasts = np.array([m.get('forecastability', np.nan) for m in all_metrics_list], dtype=float)
    cvs = np.array([m.get('cv', np.nan) for m in all_metrics_list], dtype=float)
    missings = np.array([m.get('missing_ratio', 0) for m in all_metrics_list], dtype=float)
    seasons = np.array([m.get('season_strength', np.nan) for m in all_metrics_list], dtype=float)
    trends = np.array([m.get('trend_strength', np.nan) for m in all_metrics_list], dtype=float)
    hursts = np.array([m.get('hurst', np.nan) for m in all_metrics_list], dtype=float)
    adfs = np.array([m.get('adf_stat', np.nan) for m in all_metrics_list], dtype=float) if compute_adf else np.array([])
    
    total_points = int(np.nansum(lengths)) if lengths.size else 0
    
    def wavg(x, w):
        """Weighted average ignoring NaNs."""
        mask = ~np.isnan(x) & ~np.isnan(w)
        if not mask.any():
            return np.nan
        return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))
    
    def safe_median(x):
        valid = x[~np.isnan(x)]
        return float(np.median(valid)) if valid.size else np.nan
    
    def safe_mean(x):
        valid = x[~np.isnan(x)]
        return float(np.mean(valid)) if valid.size else np.nan
    
    # Build weighted_metrics (length-weighted averages) - format expected by print_dataset_report
    weighted_metrics = {
        'forecastability': wavg(fcasts, lengths),
        'cv': wavg(cvs, lengths),
        'missing_ratio': wavg(missings, lengths),
        'season_strength': wavg(seasons, lengths),
        'trend_strength': wavg(trends, lengths),
    }
    
    # Build unweighted_metrics (simple means)
    unweighted_metrics = {
        'forecastability': safe_mean(fcasts),
        'cv': safe_mean(cvs),
        'missing_ratio': safe_mean(missings),
        'season_strength': safe_mean(seasons),
        'trend_strength': safe_mean(trends),
    }
    
    if compute_hurst:
        weighted_metrics['hurst'] = wavg(hursts, lengths)
        unweighted_metrics['hurst'] = safe_mean(hursts)
    
    if compute_adf and adfs.size > 0:
        weighted_metrics['adf_stat'] = wavg(adfs, lengths)
        unweighted_metrics['adf_stat'] = safe_mean(adfs)
    
    # Build summary in format expected by print_dataset_report
    summary = {
        'num_series': len(all_metrics_list),
        'total_time_points': total_points,
        'avg_series_length': float(np.nanmean(lengths)) if lengths.size else float('nan'),
        'weighted_metrics': weighted_metrics,
        'unweighted_metrics': unweighted_metrics,
        # Also keep detailed stats for JSON output
        'forecastability_median': safe_median(fcasts),
        'forecastability_mean': safe_mean(fcasts),
        'cv_median': safe_median(cvs),
        'cv_mean': safe_mean(cvs),
        'missing_ratio_median': safe_median(missings),
        'missing_ratio_mean': safe_mean(missings),
        'eff_length_median': safe_median(lengths),
        'eff_length_mean': safe_mean(lengths),
        'season_strength_median': safe_median(seasons),
        'season_strength_mean': safe_mean(seasons),
        'trend_strength_median': safe_median(trends),
        'trend_strength_mean': safe_mean(trends),
    }
    
    if compute_hurst:
        summary['hurst_median'] = safe_median(hursts)
        summary['hurst_mean'] = safe_mean(hursts)
    
    if compute_adf and adfs.size > 0:
        summary['adf_stat_median'] = safe_median(adfs)
        summary['adf_stat_mean'] = safe_mean(adfs)
    
    return summary


def analyze_single_file(
    file_path: Path,
    value_col: str = 'value',
    max_length: Optional[int] = None,
    max_series: Optional[int] = None,
    sampling_strategy: str = 'random',
    period: int = 24,
    compute_adf: bool = False,
    compute_hurst: bool = True,
    use_all_indices: bool = False,
    output_path: Optional[Path] = None,
    num_workers: int = 8
):
    """Analyze quality of a single parquet file with optional multiprocessing."""
    logger.info("="*80)
    logger.info(f"Analyzing: {file_path.name}")
    logger.info("="*80)
    
    # Check if results already exist
    if output_path and output_path.exists():
        logger.info(f"Loading existing results from {output_path}")
        # For single file, we usually just run it, but we could check content.
        pass

    # Load series with metadata
    series_list, metadata_list = load_series_with_metadata(
        file_path,
        value_col=value_col,
        max_length=max_length,
        max_series=max_series,
        sampling_strategy=sampling_strategy,
        use_all_indices=use_all_indices
    )
    
    if not series_list:
        logger.warning(f"No series found in {file_path.name}")
        return None
    
    # Evaluate all series
    total_series = len(series_list)
    actual_workers = num_workers if num_workers > 0 else mp.cpu_count()
    
    if actual_workers > 1 and total_series > 10:
        # Multiprocessing mode
        logger.info(f"Evaluating {total_series} series using {actual_workers} workers...")
        
        worker_args = [
            (series, meta, period, compute_adf, compute_hurst)
            for series, meta in zip(series_list, metadata_list)
        ]
        
        all_series_metrics = []
        chunk_size = max(1, total_series // (actual_workers * 4))
        
        with mp.Pool(processes=actual_workers) as pool:
            for i, result in enumerate(pool.imap(_evaluate_series_worker, worker_args, chunksize=chunk_size), 1):
                all_series_metrics.append(result)
                if i % 500 == 0 or i == total_series:
                    logger.info(f"  Evaluated {i}/{total_series} series ({100*i/total_series:.1f}%)")
        
        logger.info(f"Completed parallel evaluation of {len(all_series_metrics)} series")
    else:
        # Sequential mode
        logger.info(f"Evaluating {total_series} series sequentially...")
        all_series_metrics = []
        for idx, (series, meta) in enumerate(zip(series_list, metadata_list), 1):
            if idx % 100 == 0 or idx == total_series:
                logger.info(f"  Evaluating series {idx}/{total_series}")
            metrics = evaluate_series(
                series,
                period=period,
                compute_adf=compute_adf,
                compute_hurst=compute_hurst
            )
            all_series_metrics.append({
                'item_id': meta['item_id'],
                'index': meta['index'],
                'metrics': metrics.to_dict()
            })
    
    # Aggregate by item_id and index
    # Key format: "{item_id}_{index}" e.g. "0_ind_1", "123_ind_3"
    logger.info("Aggregating metrics by item_id and index...")
    items_dict = {}
    indices_dict = {}
    
    for entry in all_series_metrics:
        item_id = entry['item_id']
        index = entry['index']
        metrics = entry['metrics']
        
        # Create flat key: "{item_id}_{index}"
        key = f"{item_id}_{index}"
        items_dict[key] = metrics
        
        # Group by index for summary statistics
        if index not in indices_dict:
            indices_dict[index] = []
        indices_dict[index].append(metrics)
    
    # Compute totals across all indicators
    all_metrics_list = [entry['metrics'] for entry in all_series_metrics]
    totals = {}
    if all_metrics_list:
        for key in all_metrics_list[0].keys():
            values = [m[key] for m in all_metrics_list if not np.isnan(m[key])]
            if values:
                totals[f'{key}_mean'] = float(np.mean(values))
                totals[f'{key}_median'] = float(np.median(values))
                totals[f'{key}_min'] = float(np.min(values))
                totals[f'{key}_max'] = float(np.max(values))
    
    # Also compute summary per index
    index_summaries = {}
    for index, metrics_list in indices_dict.items():
        index_summaries[index] = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if not np.isnan(m[key])]
            if values:
                index_summaries[index][f'{key}_mean'] = float(np.mean(values))
                index_summaries[index][f'{key}_median'] = float(np.median(values))
    
    # Build summary from already-computed metrics (avoid re-evaluating all series)
    logger.info("Computing dataset summary from pre-computed metrics...")
    summary = _compute_summary_from_metrics(all_metrics_list, compute_adf=compute_adf, compute_hurst=compute_hurst)
    
    # Build comprehensive results
    # Key format: "{item_id}_{index}" e.g. "0_ind_1", "123_ind_3"
    unique_item_ids = set(entry['item_id'] for entry in all_series_metrics)
    results = {
        'summary': summary,
        'totals': totals,
        'by_index': index_summaries,
        'by_item': items_dict,  # Flat structure: key = "{item_id}_{index}", value = metrics dict
        'num_items': len(unique_item_ids),  # Count of unique item_ids
        'num_series': len(series_list),
        'indices': list(indices_dict.keys())
    }
    
    # Log structure confirmation
    logger.info(f"Stored metrics for {len(items_dict)} item-index combinations across {len(indices_dict)} indices")
    if items_dict:
        sample_key = list(items_dict.keys())[0]
        logger.info(f"Sample key format: '{sample_key}' (format: item_id_index)")
    
    # Print report
    print_dataset_report(results['summary'])
    
    # Save results
    if output_path:
        save_json(results, output_path, logger=logger)
    
    return results


def analyze_multiple_files(
    file_paths: List[Path],
    value_col: str = 'value',
    max_length: Optional[int] = None,
    max_series: Optional[int] = None,
    sampling_strategy: str = 'random',
    period: int = 24,
    compute_adf: bool = False,
    compute_hurst: bool = True,
    compare: bool = True,
    use_all_indices: bool = False,
    output_path: Optional[Path] = None,
    num_workers: int = 8
):
    """Analyze quality of multiple parquet files and optionally compare them.
    
    Args:
        num_workers: Number of parallel workers for series evaluation. 
                     Use 0 for all CPUs, 1 for sequential. Default: 8.
    """
    logger.info("="*80)
    logger.info("Analyzing Multiple Files")
    logger.info("="*80)
    
    datasets = {}
    all_results = {}
    newly_processed = 0  # Track newly processed files for accurate progress
    
    # Resume from existing results if available
    if output_path and output_path.exists():
        logger.info(f"Found existing results file: {output_path}")
        all_results = load_json(output_path, logger=logger)
        logger.info(f"Loaded {len(all_results)} previously processed files.")
        
        # Convert old nested structure to new flat structure if needed
        for file_key, file_data in all_results.items():
            if 'by_item' in file_data and file_data['by_item']:
                # Check if it's old nested structure (has nested dicts)
                first_key = list(file_data['by_item'].keys())[0]
                if isinstance(file_data['by_item'][first_key], dict) and not any('_' in k for k in file_data['by_item'].keys()):
                    logger.info(f"Converting old nested structure to flat structure for {file_key}...")
                    # Convert nested to flat
                    flat_by_item = {}
                    for item_id, indices_dict in file_data['by_item'].items():
                        if isinstance(indices_dict, dict):
                            for index, metrics in indices_dict.items():
                                flat_key = f"{item_id}_{index}"
                                flat_by_item[flat_key] = metrics
                    file_data['by_item'] = flat_by_item
                    logger.info(f"  Converted {len(flat_by_item)} item-index combinations")
    
    total_files = len(file_paths)
    already_processed = len([f for f in file_paths if f.stem in all_results])
    files_to_process = total_files - already_processed
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {total_files} parquet files sequentially")
    logger.info(f"Files already processed: {already_processed}")
    logger.info(f"Files to process: {files_to_process}")
    logger.info(f"{'='*80}\n")
    
    for file_idx, file_path in enumerate(file_paths, 1):
        if file_path.stem in all_results:
            logger.info(f"[{file_idx}/{total_files}] Skipping {file_path.name} (already in results)")
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"[{file_idx}/{total_files}] Processing: {file_path.name}")
        logger.info(f"{'='*80}")
        logger.info(f"Loading data from {file_path.name}...")
        
        # Load series with metadata
        series_list, metadata_list = load_series_with_metadata(
            file_path,
            value_col=value_col,
            max_length=max_length,
            max_series=max_series,
            sampling_strategy=sampling_strategy,
            use_all_indices=use_all_indices
        )
        
        if not series_list:
            logger.warning(f"Skipping {file_path.name} (no series found)")
            continue
        
        logger.info(f"Loaded {len(series_list)} series from {file_path.name}")
        
        # Store for comparison (only for newly processed files)
        if compare:
            datasets[file_path.stem] = series_list
        
        # Evaluate all series with metadata
        total_series = len(series_list)
        
        # Determine number of workers
        actual_workers = num_workers if num_workers > 0 else mp.cpu_count()
        
        if actual_workers > 1 and total_series > 10:
            # Multiprocessing mode
            logger.info(f"Evaluating {total_series} series using {actual_workers} workers...")
            
            # Prepare arguments for workers
            worker_args = [
                (series, meta, period, compute_adf, compute_hurst)
                for series, meta in zip(series_list, metadata_list)
            ]
            
            # Process in parallel with progress updates
            all_series_metrics = []
            chunk_size = max(1, total_series // (actual_workers * 4))  # Balance between overhead and progress updates
            
            with mp.Pool(processes=actual_workers) as pool:
                for i, result in enumerate(pool.imap(_evaluate_series_worker, worker_args, chunksize=chunk_size), 1):
                    all_series_metrics.append(result)
                    if i % 500 == 0 or i == total_series:
                        logger.info(f"  Evaluated {i}/{total_series} series ({100*i/total_series:.1f}%)")
            
            logger.info(f"Completed parallel evaluation of {len(all_series_metrics)} series")
        else:
            # Sequential mode (single worker or small dataset)
            logger.info(f"Evaluating {total_series} series sequentially...")
            all_series_metrics = []
            
            for idx, (series, meta) in enumerate(zip(series_list, metadata_list), 1):
                if idx % 100 == 0 or idx == total_series:
                    logger.info(f"  Evaluating series {idx}/{total_series} (item_id={meta['item_id']}, index={meta['index']})")
                
                metrics = evaluate_series(
                    series,
                    period=period,
                    compute_adf=compute_adf,
                    compute_hurst=compute_hurst
                )
                
                all_series_metrics.append({
                    'item_id': meta['item_id'],
                    'index': meta['index'],
                    'metrics': metrics.to_dict()
                })
            
            logger.info(f"Completed sequential evaluation of {len(all_series_metrics)} series")
        
        logger.info(f"Aggregating metrics by item_id and index...")
        
        # Aggregate by item_id and index
        # Key format: "{item_id}_{index}" e.g. "0_ind_1", "123_ind_3"
        items_dict = {}
        indices_dict = {}
        
        for entry in all_series_metrics:
            item_id = entry['item_id']
            index = entry['index']
            metrics = entry['metrics']
            
            # Create flat key: "{item_id}_{index}"
            key = f"{item_id}_{index}"
            items_dict[key] = metrics
            
            # Group by index for summary statistics
            if index not in indices_dict:
                indices_dict[index] = []
            indices_dict[index].append(metrics)
        
        # Compute totals across all indicators
        all_metrics_list = [entry['metrics'] for entry in all_series_metrics]
        totals = {}
        if all_metrics_list:
            for key in all_metrics_list[0].keys():
                values = [m[key] for m in all_metrics_list if not np.isnan(m[key])]
                if values:
                    totals[f'{key}_mean'] = float(np.mean(values))
                    totals[f'{key}_median'] = float(np.median(values))
                    totals[f'{key}_min'] = float(np.min(values))
                    totals[f'{key}_max'] = float(np.max(values))
        
        # Also compute summary per index
        index_summaries = {}
        for index, metrics_list in indices_dict.items():
            index_summaries[index] = {}
            for key in metrics_list[0].keys():
                values = [m[key] for m in metrics_list if not np.isnan(m[key])]
                if values:
                    index_summaries[index][f'{key}_mean'] = float(np.mean(values))
                    index_summaries[index][f'{key}_median'] = float(np.median(values))
        
        # Build summary from already-computed metrics (avoid re-evaluating all series)
        logger.info("Computing dataset summary from pre-computed metrics...")
        summary = _compute_summary_from_metrics(all_metrics_list, compute_adf=compute_adf, compute_hurst=compute_hurst)
        
        # Build comprehensive results
        # Key format: "{item_id}_{index}" e.g. "0_ind_1", "123_ind_3"
        unique_item_ids = set(entry['item_id'] for entry in all_series_metrics)
        results = {
            'summary': summary,
            'totals': totals,
            'by_index': index_summaries,
            'by_item': items_dict,  # Flat structure: key = "{item_id}_{index}", value = metrics dict
            'num_items': len(unique_item_ids),  # Count of unique item_ids
            'num_series': len(series_list),
            'indices': list(indices_dict.keys())
        }
        
        # Log structure confirmation
        logger.info(f"Stored metrics for {len(items_dict)} item-index combinations across {len(indices_dict)} indices")
        logger.info(f"  Unique item_ids: {len(unique_item_ids)}")
        logger.info(f"  Indices: {list(indices_dict.keys())}")
        if items_dict:
            sample_key = list(items_dict.keys())[0]
            logger.info(f"  Sample key format: '{sample_key}' (format: item_id_index)")
        
        all_results[file_path.stem] = results
        newly_processed += 1
        
        # Save incrementally
        if output_path:
            save_json(all_results, output_path, logger=logger)
            logger.info(f"Saved results for {file_path.name} (progress: {newly_processed}/{files_to_process} new files, {len(all_results)}/{total_files} total)")
        
        logger.info(f"✓ Completed processing {file_path.name}\n")
    
    # Print individual results
    # We can print results for all files (loaded + new)
    for name, results in all_results.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Results for {name}")
        logger.info(f"{'='*80}")
        if isinstance(results, dict) and 'summary' in results:
            print_dataset_report(results['summary'])
            logger.info(f"\nTotal items: {results.get('num_items', 'N/A')}")
            logger.info(f"Total series: {results.get('num_series', 'N/A')}")
            logger.info(f"Indices: {', '.join(results.get('indices', []))}")
        else:
            # Legacy format
            print_dataset_report(results)
    
    # Compare datasets if requested
    # Note: comparison will only include newly processed files because we didn't load raw data for resumed ones
    if compare:
        if len(datasets) > 1:
            logger.info(f"\n{'='*80}")
            logger.info("Cross-Dataset Comparison (Newly processed files only)")
            logger.info(f"{'='*80}")
            
            summaries = compare_datasets(
                datasets,
                period=period,
                compute_adf=compute_adf
            )
            
            print_comparison(summaries, sort_by="QualityScore")
        elif len(datasets) > 0 and len(all_results) > len(datasets):
             logger.warning("\nSkipping comparison for resumed files (raw data not loaded). Only newly processed files would be compared.")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset quality for open_hour_train parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all files with defaults (full length, all items, all indices)
  python examples/analyze_open_hour_train_quality.py

  # Faster analysis with sampling
  python examples/analyze_open_hour_train_quality.py \\
      --max_length 5000 \\
      --max_series_per_file 100

  # Analyze specific files
  python examples/analyze_open_hour_train_quality.py \\
      --files hour_train_hour_p1.parquet hour_train_hour_p2.parquet

  # Skip Hurst exponent for faster analysis
  python examples/analyze_open_hour_train_quality.py --no-compute-hurst

  # Include ADF stationarity test (slow)
  python examples/analyze_open_hour_train_quality.py --compute_adf
        """
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='examples/datasets/parquet_data/open_hour_train',
        help='Directory containing parquet files (default: examples/datasets/parquet_data/open_hour_train)'
    )
    
    parser.add_argument(
        '--files',
        type=str,
        nargs='+',
        default=None,
        help='Specific files to analyze (default: all *.parquet files in data_dir)'
    )
    
    parser.add_argument(
        '--value_col',
        type=str,
        default='value',
        help='Name of the value column (default: value, will try ind_1 if not found)'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=0,
        help='Maximum length per series (truncate if longer, default: 0 for full length)'
    )
    
    parser.add_argument(
        '--max_series_per_file',
        type=int,
        default=0,
        help='Maximum number of series to sample per file (default: 0 for all items)'
    )
    
    parser.add_argument(
        '--sampling_strategy',
        type=str,
        choices=['random', 'first', 'last', 'uniform'],
        default='random',
        help='Strategy for sampling series (default: random)'
    )
    
    parser.add_argument(
        '--period',
        type=int,
        default=24,
        help='Seasonal period for hourly data (default: 24)'
    )
    
    parser.add_argument(
        '--compute_adf',
        action='store_true',
        default=False,
        help='Compute ADF stationarity test (default: False, slow)'
    )
    
    parser.add_argument(
        '--compute_hurst',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Compute Hurst exponent (default: True)'
    )
    
    parser.add_argument(
        '--no_compare',
        action='store_true',
        help='Skip cross-dataset comparison'
    )

    parser.add_argument(
        '--no_all_indices',
        action='store_true',
        help='Only use first value column instead of all ind_* columns'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default='examples/quality_results.json',
        help='Path to save results JSON (default: examples/quality_results.json)'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of parallel workers (0=all CPUs, 1=sequential). Default: 8'
    )
    
    args = parser.parse_args()
    
    # Resolve data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please ensure the data directory exists or specify --data_dir")
        return
    
    # Find parquet files
    if args.files:
        file_paths = [data_dir / f for f in args.files]
        file_paths = [f for f in file_paths if f.exists()]
        if not file_paths:
            logger.error(f"No files found: {args.files}")
            return
    else:
        file_paths = sorted(data_dir.glob("*.parquet"))
        if not file_paths:
            logger.error(f"No parquet files found in {data_dir}")
            return
    
    # Derive boolean flags
    use_all_indices = not args.no_all_indices
    compute_hurst = args.compute_hurst
    max_length = args.max_length if args.max_length > 0 else None
    max_series = args.max_series_per_file if args.max_series_per_file > 0 else None
    
    logger.info(f"Found {len(file_paths)} parquet file(s) to analyze")
    logger.info(f"Settings:")
    logger.info(f"  Max length per series: {max_length or 'Full length'}")
    logger.info(f"  Max series per file: {max_series or 'All items'}")
    logger.info(f"  Sampling strategy: {args.sampling_strategy}")
    logger.info(f"  Seasonal period: {args.period}")
    logger.info(f"  Compute ADF: {args.compute_adf}")
    logger.info(f"  Compute Hurst: {compute_hurst}")
    logger.info(f"  Use all indices: {use_all_indices}")
    logger.info(f"  Output file: {args.output_file}")
    logger.info(f"  Num workers: {args.num_workers}" + (" (all CPUs)" if args.num_workers == 0 else ""))

    output_path = Path(args.output_file)
    
    # Analyze
    if len(file_paths) == 1:
        # Single file analysis
        analyze_single_file(
            file_paths[0],
            value_col=args.value_col,
            max_length=max_length,
            max_series=max_series,
            sampling_strategy=args.sampling_strategy,
            period=args.period,
            compute_adf=args.compute_adf,
            compute_hurst=compute_hurst,
            use_all_indices=use_all_indices,
            output_path=output_path,
            num_workers=args.num_workers
        )
    else:
        # Multiple files analysis
        analyze_multiple_files(
            file_paths,
            value_col=args.value_col,
            max_length=max_length,
            max_series=max_series,
            sampling_strategy=args.sampling_strategy,
            period=args.period,
            compute_adf=args.compute_adf,
            compute_hurst=compute_hurst,
            compare=not args.no_compare,
            use_all_indices=use_all_indices,
            output_path=output_path,
            num_workers=args.num_workers
        )
    
    logger.info("\n" + "="*80)
    logger.info("Analysis Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
