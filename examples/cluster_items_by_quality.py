#!/usr/bin/env python3
"""
Classify items into meaningful time series behavior groups.

Groups are based on domain-meaningful criteria:
    1. Stable Seasonal      - High seasonality, low trend, low CV, easy to forecast
    2. Trending + Seasonal  - High seasonality, high trend, persistent (H > 0.55)
    3. Persistent Trend     - High Hurst (>0.6), low seasonality, high trend
    4. Mean-Reverting       - Hurst < 0.45, low trend, possibly high CV
    5. Noisy & Unpredictable- Low forecastability, high CV, low seasonality/trend
    6. Random-Like / IID    - Hurst ≈ 0.5, low structure everywhere

Usage:
    python examples/cluster_items_by_quality.py --input examples/quality_results.json
    python examples/cluster_items_by_quality.py --method rules  # Rule-based (default)
    python examples/cluster_items_by_quality.py --method kmeans --n_clusters 6
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# MEANINGFUL TIME SERIES BEHAVIOR GROUPS
# ============================================================================
# These groups represent distinct time series behaviors with clear interpretations

BEHAVIOR_GROUPS = {
    1: {
        'name': 'Stable Seasonal',
        'description': 'Regular patterns, easy to forecast',
        'criteria': 'High seasonality, low trend, low CV, moderate/high forecastability',
        'color': '#2ecc71',  # Green
    },
    2: {
        'name': 'Trending + Seasonal',
        'description': 'Classic multiplicative/trending seasonal (e.g., sales growth)',
        'criteria': 'High seasonality, high trend, H > 0.55',
        'color': '#3498db',  # Blue
    },
    3: {
        'name': 'Persistent Trend',
        'description': 'Long-memory trending behavior',
        'criteria': 'High Hurst (>0.6), low seasonality, high trend',
        'color': '#9b59b6',  # Purple
    },
    4: {
        'name': 'Mean-Reverting',
        'description': 'Reverts to mean over time (e.g., spreads, interest rates)',
        'criteria': 'Hurst < 0.45, low trend, possibly high CV',
        'color': '#e74c3c',  # Red
    },
    5: {
        'name': 'Noisy & Unpredictable',
        'description': 'Hard to model; may need differencing or ignore',
        'criteria': 'Low forecastability, high CV, low seasonality/trend',
        'color': '#f39c12',  # Orange
    },
    6: {
        'name': 'Random-Like / IID Noise',
        'description': 'Behaves like white noise',
        'criteria': 'Hurst ≈ 0.5, low structure everywhere',
        'color': '#95a5a6',  # Gray
    },
}


def classify_series_behavior(
    forecastability: float,
    season_strength: float,
    trend_strength: float,
    hurst: float,
    cv: float
) -> int:
    """
    Classify a time series into one of 6 meaningful behavior groups.
    
    Returns:
        Group number (1-6)
    """
    # Handle NaN values with defaults
    forecastability = forecastability if not np.isnan(forecastability) else 0.3
    season_strength = season_strength if not np.isnan(season_strength) else 0.3
    trend_strength = trend_strength if not np.isnan(trend_strength) else 0.3
    hurst = hurst if not np.isnan(hurst) else 0.5
    cv = cv if not np.isnan(cv) else 1.0
    
    # Thresholds
    HIGH_SEASON = 0.5
    LOW_SEASON = 0.3
    HIGH_TREND = 0.5
    LOW_TREND = 0.3
    HIGH_HURST = 0.6
    LOW_HURST = 0.45
    RANDOM_HURST_LO = 0.45
    RANDOM_HURST_HI = 0.55
    HIGH_CV = 1.0
    LOW_CV = 0.5
    HIGH_FORECAST = 0.5
    LOW_FORECAST = 0.3
    
    # 1. Stable Seasonal: High seasonality, low trend, low CV, moderate/high forecastability
    if (season_strength >= HIGH_SEASON and 
        trend_strength < HIGH_TREND and 
        cv < HIGH_CV and 
        forecastability >= LOW_FORECAST):
        return 1
    
    # 2. Trending + Seasonal: High seasonality, high trend, H > 0.55
    if (season_strength >= HIGH_SEASON and 
        trend_strength >= HIGH_TREND and 
        hurst > 0.55):
        return 2
    
    # 3. Persistent Trend: High Hurst (>0.6), low seasonality, high trend
    if (hurst > HIGH_HURST and 
        season_strength < HIGH_SEASON and 
        trend_strength >= HIGH_TREND):
        return 3
    
    # 4. Mean-Reverting: Hurst < 0.45, low trend
    if (hurst < LOW_HURST and 
        trend_strength < HIGH_TREND):
        return 4
    
    # 5. Noisy & Unpredictable: Low forecastability, high CV, low seasonality/trend
    if (forecastability < LOW_FORECAST and 
        cv >= HIGH_CV and 
        season_strength < HIGH_SEASON and 
        trend_strength < HIGH_TREND):
        return 5
    
    # 6. Random-Like / IID Noise: Hurst ≈ 0.5, low structure everywhere
    if (RANDOM_HURST_LO <= hurst <= RANDOM_HURST_HI and 
        season_strength < HIGH_SEASON and 
        trend_strength < HIGH_TREND):
        return 6
    
    # Fallback: Check for dominant characteristics
    
    # Strong seasonality wins
    if season_strength >= HIGH_SEASON:
        if trend_strength >= HIGH_TREND:
            return 2  # Trending + Seasonal
        else:
            return 1  # Stable Seasonal
    
    # Strong trend without seasonality
    if trend_strength >= HIGH_TREND:
        if hurst > 0.55:
            return 3  # Persistent Trend
        else:
            return 5  # Noisy (trending but not persistent)
    
    # Low forecastability
    if forecastability < LOW_FORECAST:
        return 5  # Noisy & Unpredictable
    
    # Default to Random-Like
    return 6


# Domain-meaningful bins for each metric (used for discretization display)
METRIC_BINS = {
    'forecastability': {
        'bins': [0, 0.25, 0.5, 0.75, 1.0],
        'labels': ['very_low', 'low', 'medium', 'high'],
        'description': 'Predictability (1 - spectral entropy)'
    },
    'season_strength': {
        'bins': [0, 0.3, 0.6, 1.0],
        'labels': ['weak', 'moderate', 'strong'],
        'description': 'Strength of seasonal patterns'
    },
    'trend_strength': {
        'bins': [0, 0.3, 0.6, 1.0],
        'labels': ['weak', 'moderate', 'strong'],
        'description': 'Strength of trend component'
    },
    'hurst': {
        'bins': [0, 0.4, 0.6, 1.0],
        'labels': ['mean_reverting', 'random', 'trending'],
        'description': 'Long-range dependence (H<0.5=reverting, H=0.5=random, H>0.5=persistent)'
    },
    'cv': {
        'bins': [0, 0.5, 1.0, 2.0, float('inf')],
        'labels': ['stable', 'moderate', 'variable', 'highly_variable'],
        'description': 'Coefficient of variation (std/mean)'
    },
    'missing_ratio': {
        'bins': [0, 0.01, 0.1, 0.5, 1.0],
        'labels': ['complete', 'minor_gaps', 'partial', 'sparse'],
        'description': 'Fraction of missing values'
    }
}


def discretize_value(value: float, metric: str) -> str:
    """Convert continuous value to category based on domain-meaningful bins."""
    if metric not in METRIC_BINS:
        return 'unknown'
    
    bins = METRIC_BINS[metric]['bins']
    labels = METRIC_BINS[metric]['labels']
    
    if np.isnan(value):
        return 'missing'
    
    for i, threshold in enumerate(bins[1:]):
        if value <= threshold:
            return labels[i]
    return labels[-1]


def load_and_discretize(json_path: Path) -> pd.DataFrame:
    """
    Load quality metrics and create discretized profiles per item.
    
    Returns DataFrame with:
    - item_id as index
    - {metric}_val: continuous value (mean across indices)
    - {metric}_cat: discretized category
    """
    with open(json_path) as f:
        data = json.load(f)
    
    # Collect metrics per item (across all files and indices)
    items = defaultdict(lambda: defaultdict(list))
    
    def process_by_item(by_item_dict):
        """Process a by_item dictionary."""
        for key, metrics in by_item_dict.items():
            # Key format: "{item_id}_{index}" e.g., "123_ind_1"
            parts = key.rsplit('_', 2)
            item_id = '_'.join(parts[:-2]) if len(parts) > 2 else parts[0]
            
            for metric_name, value in metrics.items():
                if metric_name in METRIC_BINS and isinstance(value, (int, float)):
                    items[item_id][metric_name].append(value)
    
    # Handle both single-file and multi-file result structures
    if 'by_item' in data and isinstance(data['by_item'], dict):
        # Single file result: by_item at top level
        process_by_item(data['by_item'])
    else:
        # Multi-file result: by_item nested under file keys
        for file_key, file_data in data.items():
            if isinstance(file_data, dict) and 'by_item' in file_data:
                process_by_item(file_data['by_item'])
    
    # Build profiles: aggregate across indices, then discretize
    rows = []
    for item_id, metrics in items.items():
        row = {'item_id': item_id}
        
        for metric_name in METRIC_BINS.keys():
            values = metrics.get(metric_name, [])
            if values:
                # Use mean across indices
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values) if len(values) > 1 else 0.0
                row[f'{metric_name}_val'] = mean_val
                row[f'{metric_name}_std'] = std_val
                row[f'{metric_name}_cat'] = discretize_value(mean_val, metric_name)
            else:
                row[f'{metric_name}_val'] = np.nan
                row[f'{metric_name}_std'] = np.nan
                row[f'{metric_name}_cat'] = 'missing'
        
        rows.append(row)
    
    return pd.DataFrame(rows).set_index('item_id')


def find_optimal_k(X: np.ndarray, k_range: range = range(2, 11)) -> int:
    """Find optimal number of clusters using silhouette score."""
    scores = []
    for k in k_range:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append((k, score))
        logger.info(f"  K={k}: silhouette={score:.3f}")
    
    best_k = max(scores, key=lambda x: x[1])[0]
    return best_k


def classify_items_by_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify items into 6 meaningful behavior groups using rule-based criteria.
    
    Returns DataFrame with 'group' column (1-6) and 'group_name' column.
    """
    groups = []
    
    for idx, row in df.iterrows():
        group = classify_series_behavior(
            forecastability=row.get('forecastability_val', np.nan),
            season_strength=row.get('season_strength_val', np.nan),
            trend_strength=row.get('trend_strength_val', np.nan),
            hurst=row.get('hurst_val', np.nan),
            cv=row.get('cv_val', np.nan)
        )
        groups.append(group)
    
    df['cluster'] = groups
    df['group_name'] = df['cluster'].map(lambda g: BEHAVIOR_GROUPS[g]['name'])
    
    return df


def cluster_discretized(
    df: pd.DataFrame, 
    n_clusters: int = None,
    method: str = 'rules'
) -> tuple:
    """
    Classify/cluster items into groups.
    
    Args:
        df: DataFrame with metrics (*_val columns)
        n_clusters: Number of clusters (only for kmeans/hierarchical)
        method: 'rules' (default), 'kmeans', or 'hierarchical'
    
    Returns:
        (df_with_clusters, encoded_features_df)
    """
    cat_cols = [c for c in df.columns if c.endswith('_cat')]
    
    # One-hot encode for visualization (used by all methods)
    df_encoded = pd.get_dummies(df[cat_cols], prefix_sep='=')
    X = df_encoded.values.astype(float)
    
    if method == 'rules':
        # Rule-based classification into 6 meaningful groups
        logger.info("Classifying items using rule-based behavior groups...")
        df = classify_items_by_rules(df)
        logger.info(f"Classified {len(df)} items into {df['cluster'].nunique()} groups")
        return df, df_encoded
    
    # Unsupervised clustering methods
    logger.info(f"Feature matrix: {X.shape[0]} items × {X.shape[1]} binary features")
    
    # Find optimal K if not specified
    if n_clusters is None:
        logger.info("Finding optimal number of clusters...")
        max_k = min(11, len(df) // 5)
        if max_k > 2:
            n_clusters = find_optimal_k(X, k_range=range(2, max_k))
            logger.info(f"→ Optimal K: {n_clusters}")
        else:
            n_clusters = 2
            logger.info(f"→ Using K={n_clusters} (dataset too small for optimization)")
    
    # Cluster
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    
    df['cluster'] = model.fit_predict(X)
    
    return df, df_encoded


def analyze_clusters(df: pd.DataFrame, use_rules: bool = True) -> dict:
    """
    Analyze and describe each cluster's characteristics.
    
    Returns dict mapping cluster_id -> profile description
    """
    cat_cols = [c for c in df.columns if c.endswith('_cat')]
    val_cols = [c for c in df.columns if c.endswith('_val')]
    
    print(f"\n{'='*70}")
    if use_rules:
        print("BEHAVIOR GROUP ANALYSIS")
    else:
        print("CLUSTER ANALYSIS")
    print(f"{'='*70}")
    
    cluster_profiles = {}
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        n = len(cluster_df)
        pct = 100 * n / len(df)
        
        print(f"\n{'─'*70}")
        
        # Get group info if using rules
        if use_rules and cluster_id in BEHAVIOR_GROUPS:
            group_info = BEHAVIOR_GROUPS[cluster_id]
            print(f"GROUP {cluster_id}: {group_info['name']}")
            print(f"  {n} items ({pct:.1f}%)")
            print(f"  Description: {group_info['description']}")
            print(f"  Criteria: {group_info['criteria']}")
            cluster_profiles[cluster_id] = group_info['name']
        else:
            print(f"CLUSTER {cluster_id}: {n} items ({pct:.1f}%)")
        
        print(f"{'─'*70}")
        
        # Show metric statistics
        print("  Metric Statistics:")
        for col in val_cols:
            metric = col.replace('_val', '')
            mean_val = cluster_df[col].mean()
            std_val = cluster_df[col].std()
            cat_col = f'{metric}_cat'
            
            if cat_col in cluster_df.columns:
                counts = cluster_df[cat_col].value_counts()
                dominant = counts.index[0]
                dominant_pct = 100 * counts.iloc[0] / n
                print(f"    {metric:18s}: mean={mean_val:6.3f} (±{std_val:.3f}) | {dominant} ({dominant_pct:.0f}%)")
            else:
                print(f"    {metric:18s}: mean={mean_val:6.3f} (±{std_val:.3f})")
        
        # Build profile for non-rule-based clustering
        if not use_rules or cluster_id not in BEHAVIOR_GROUPS:
            profile_parts = []
            for col in cat_cols:
                metric = col.replace('_cat', '')
                counts = cluster_df[col].value_counts()
                dominant = counts.index[0]
                dominant_pct = 100 * counts.iloc[0] / n
                if dominant_pct > 60:
                    profile_parts.append(f"{metric}={dominant}")
            cluster_profiles[cluster_id] = ' + '.join(profile_parts) if profile_parts else 'Mixed'
    
    return cluster_profiles


def visualize_clusters(
    df: pd.DataFrame, 
    df_encoded: pd.DataFrame = None,
    output_path: Path = None,
    method: str = 'tsne',
    use_rules: bool = True
):
    """
    Create visualization of cluster characteristics.
    
    Args:
        df: DataFrame with cluster assignments and metrics
        df_encoded: One-hot encoded features used for clustering
        output_path: Path to save the plot
        method: 'tsne', 'pca', or 'both' for dimensionality reduction
        use_rules: Whether using rule-based groups (affects colors and labels)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
    except ImportError:
        logger.warning("matplotlib not available, skipping visualization")
        return
    
    val_cols = [c for c in df.columns if c.endswith('_val')]
    n_clusters = df['cluster'].nunique()
    
    # Use group colors if rule-based, otherwise use tab10
    if use_rules:
        cluster_ids = sorted(df['cluster'].unique())
        colors = {cid: to_rgba(BEHAVIOR_GROUPS.get(cid, {}).get('color', '#888888')) 
                  for cid in cluster_ids}
        labels = {cid: BEHAVIOR_GROUPS.get(cid, {}).get('name', f'Cluster {cid}') 
                  for cid in cluster_ids}
    else:
        cluster_ids = sorted(df['cluster'].unique())
        tab_colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
        colors = {cid: tab_colors[i] for i, cid in enumerate(cluster_ids)}
        labels = {cid: f'Cluster {cid}' for cid in cluster_ids}
    
    # Determine layout based on visualization method
    if method == 'both':
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    plot_idx = 0
    
    # Get feature matrix for dimensionality reduction
    if df_encoded is not None:
        X = df_encoded.values.astype(float)
    else:
        # Fall back to value columns
        X = df[val_cols].fillna(0).values
    
    # t-SNE visualization
    if method in ['tsne', 'both']:
        ax = axes[plot_idx]
        plot_idx += 1
        
        logger.info("Computing t-SNE embedding...")
        perplexity = min(30, len(df) - 1)  # perplexity must be less than n_samples
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X)
        
        for cluster_id in cluster_ids:
            mask = df['cluster'] == cluster_id
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                      c=[colors[cluster_id]], label=labels[cluster_id], 
                      alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        title = 'Behavior Groups (t-SNE)' if use_rules else 't-SNE Visualization'
        ax.set_title(title)
        ax.legend(loc='best', fontsize=7)
    
    # PCA visualization
    if method in ['pca', 'both']:
        ax = axes[plot_idx]
        plot_idx += 1
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        for cluster_id in cluster_ids:
            mask = df['cluster'] == cluster_id
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      c=[colors[cluster_id]], label=labels[cluster_id], 
                      alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        
        var_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}%)')
        title = 'Behavior Groups (PCA)' if use_rules else 'PCA Visualization'
        ax.set_title(title)
        ax.legend(loc='best', fontsize=7)
    
    # Cluster sizes bar chart
    ax = axes[plot_idx]
    plot_idx += 1
    
    cluster_counts = df['cluster'].value_counts().sort_index()
    bar_colors = [colors[cid] for cid in cluster_counts.index]
    bars = ax.bar(range(len(cluster_counts)), cluster_counts.values, color=bar_colors)
    
    if use_rules:
        ax.set_xticks(range(len(cluster_counts)))
        ax.set_xticklabels([labels[cid] for cid in cluster_counts.index], rotation=45, ha='right', fontsize=8)
        ax.set_title('Items per Behavior Group')
    else:
        ax.set_xticks(range(len(cluster_counts)))
        ax.set_xticklabels([f'C{cid}' for cid in cluster_counts.index])
        ax.set_title('Items per Cluster')
    
    ax.set_ylabel('Number of Items')
    for bar, count in zip(bars, cluster_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                str(count), ha='center', va='bottom', fontsize=9)
    
    # Cluster metric profiles (normalized bar chart)
    ax = axes[plot_idx]
    
    metrics = [c.replace('_val', '') for c in val_cols]
    n_metrics = len(metrics)
    
    if n_metrics > 0:
        cluster_means = df.groupby('cluster')[val_cols].mean()
        
        # Normalize each metric to 0-1 range
        cluster_means_norm = cluster_means.copy()
        for col in val_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                cluster_means_norm[col] = (cluster_means[col] - min_val) / (max_val - min_val)
            else:
                cluster_means_norm[col] = 0.5
        
        x = np.arange(n_metrics)
        width = 0.8 / len(cluster_ids)
        
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id in cluster_means_norm.index:
                values = cluster_means_norm.loc[cluster_id].values
                ax.bar(x + i * width, values, width, label=labels[cluster_id], 
                       color=colors[cluster_id], alpha=0.8)
        
        ax.set_xticks(x + width * (len(cluster_ids) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_ylabel('Normalized Value (0-1)')
        ax.set_title('Metric Profiles by Group' if use_rules else 'Cluster Metric Profiles')
        ax.legend(loc='upper right', fontsize=6)
        ax.set_ylim(0, 1.15)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {output_path}")
    
    plt.show()


def generate_cluster_report(df: pd.DataFrame, profiles: dict, output_path: Path, use_rules: bool = True):
    """Generate markdown report summarizing clusters/groups."""
    
    if use_rules:
        lines = [
            "# Time Series Behavior Classification",
            "",
            "Items are classified into 6 meaningful behavior groups based on their quality metrics.",
            "",
            "## Behavior Groups",
            "",
            "| Group | Name | Description | Criteria |",
            "|-------|------|-------------|----------|",
        ]
        
        for gid, info in BEHAVIOR_GROUPS.items():
            lines.append(f"| {gid} | {info['name']} | {info['description']} | {info['criteria']} |")
        
        lines.extend([
            "",
            "## Overview",
            "",
            f"- **Total items**: {len(df)}",
            f"- **Groups with items**: {df['cluster'].nunique()}",
            f"- **Classification method**: Rule-based",
            "",
        ])
    else:
        lines = [
            "# Item Clustering by Quality Metrics",
            "",
            "## Overview",
            "",
            f"- **Total items**: {len(df)}",
            f"- **Number of clusters**: {df['cluster'].nunique()}",
            f"- **Clustering method**: Discretized metrics + K-Means",
            "",
            "## Discretization Bins",
            "",
            "| Metric | Bins | Categories |",
            "|--------|------|------------|",
        ]
        
        for metric, config in METRIC_BINS.items():
            bins_str = str(config['bins'])
            labels_str = ', '.join(config['labels'])
            lines.append(f"| {metric} | {bins_str} | {labels_str} |")
        
        lines.extend(["", ""])
    
    lines.extend([
        "## Group Summary" if use_rules else "## Cluster Summary",
        "",
    ])
    
    val_cols = [c for c in df.columns if c.endswith('_val')]
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        n = len(cluster_df)
        pct = 100 * n / len(df)
        
        if use_rules and cluster_id in BEHAVIOR_GROUPS:
            group_info = BEHAVIOR_GROUPS[cluster_id]
            lines.extend([
                f"### Group {cluster_id}: {group_info['name']} ({n} items, {pct:.1f}%)",
                "",
                f"**Description**: {group_info['description']}",
                "",
                "| Metric | Mean | Dominant Category |",
                "|--------|------|-------------------|",
            ])
        else:
            lines.extend([
                f"### Cluster {cluster_id}: {n} items ({pct:.1f}%)",
                "",
                f"**Profile**: {profiles.get(cluster_id, 'Mixed')}",
                "",
                "| Metric | Mean | Dominant Category |",
                "|--------|------|-------------------|",
            ])
        
        for col in val_cols:
            metric = col.replace('_val', '')
            cat_col = f'{metric}_cat'
            mean_val = cluster_df[col].mean()
            dominant = cluster_df[cat_col].value_counts().index[0]
            lines.append(f"| {metric} | {mean_val:.3f} | {dominant} |")
        
        lines.append("")
    
    # Category distributions per cluster
    lines.extend([
        "## Category Distributions by Cluster",
        "",
    ])
    
    cat_cols = [c for c in df.columns if c.endswith('_cat')]
    for col in cat_cols:
        metric = col.replace('_cat', '')
        lines.extend([
            f"### {metric}",
            "",
            "| Cluster | " + " | ".join(METRIC_BINS[metric]['labels']) + " |",
            "|---------|" + "|".join(["------"] * len(METRIC_BINS[metric]['labels'])) + "|",
        ])
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id]
            counts = cluster_df[col].value_counts()
            row = [f"Cluster {cluster_id}"]
            for label in METRIC_BINS[metric]['labels']:
                count = counts.get(label, 0)
                pct = 100 * count / len(cluster_df)
                row.append(f"{count} ({pct:.0f}%)")
            lines.append("| " + " | ".join(row) + " |")
        
        lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Saved report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster items by discretized quality metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect optimal number of clusters
  python examples/cluster_items_by_quality.py --input examples/quality_results.json

  # Specify number of clusters
  python examples/cluster_items_by_quality.py --n_clusters 5

  # Use hierarchical clustering
  python examples/cluster_items_by_quality.py --method hierarchical
        """
    )
    
    parser.add_argument('--input', type=str, default='examples/quality_results.json',
                        help='Input JSON file with quality results')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of clusters (only for kmeans/hierarchical methods)')
    parser.add_argument('--method', choices=['rules', 'kmeans', 'hierarchical'], default='rules',
                        help='Classification method: rules (default), kmeans, or hierarchical')
    parser.add_argument('--output', type=str, default='examples/item_clusters.csv',
                        help='Output CSV with cluster assignments')
    parser.add_argument('--plot', type=str, default='examples/cluster_visualization.png',
                        help='Output visualization path')
    parser.add_argument('--report', type=str, default='examples/CLUSTER_REPORT.md',
                        help='Output markdown report path')
    parser.add_argument('--viz_method', choices=['tsne', 'pca', 'both'], default='tsne',
                        help='Dimensionality reduction method for visualization (default: tsne)')
    
    args = parser.parse_args()
    
    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    # Load and discretize
    logger.info(f"Loading metrics from {args.input}...")
    df = load_and_discretize(input_path)
    logger.info(f"Loaded {len(df)} items")
    
    if len(df) < 5:
        logger.error("Not enough items for clustering (need at least 5)")
        return
    
    # Show discretization info
    print("\n" + "="*70)
    print("DISCRETIZATION BINS")
    print("="*70)
    for metric, config in METRIC_BINS.items():
        print(f"\n{metric}: {config['description']}")
        for i, label in enumerate(config['labels']):
            lo = config['bins'][i]
            hi = config['bins'][i+1]
            hi_str = f"{hi}" if hi != float('inf') else "∞"
            print(f"  {label:20s}: [{lo}, {hi_str})")
    
    # Show category distributions
    print("\n" + "="*70)
    print("CATEGORY DISTRIBUTIONS")
    print("="*70)
    for col in [c for c in df.columns if c.endswith('_cat')]:
        metric = col.replace('_cat', '')
        print(f"\n{metric}:")
        for cat, count in df[col].value_counts().items():
            print(f"  {cat:20s}: {count:5d} ({100*count/len(df):5.1f}%)")
    
    # Classify/Cluster
    use_rules = (args.method == 'rules')
    
    print("\n" + "="*70)
    if use_rules:
        print("RULE-BASED BEHAVIOR CLASSIFICATION")
        print("="*70)
        print("\nBehavior Groups:")
        for gid, info in BEHAVIOR_GROUPS.items():
            print(f"  {gid}. {info['name']}: {info['description']}")
    else:
        print("UNSUPERVISED CLUSTERING")
    print("="*70)
    
    df, df_encoded = cluster_discretized(df, n_clusters=args.n_clusters, method=args.method)
    
    # Analyze
    profiles = analyze_clusters(df, use_rules=use_rules)
    
    # Save CSV
    output_path = Path(args.output)
    df.to_csv(output_path)
    logger.info(f"Saved classification results to {output_path}")
    
    # Generate report
    generate_cluster_report(df, profiles, Path(args.report), use_rules=use_rules)
    
    # Visualize
    visualize_clusters(df, df_encoded, Path(args.plot), method=args.viz_method, use_rules=use_rules)
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"  Clusters CSV: {args.output}")
    print(f"  Report: {args.report}")
    print(f"  Visualization: {args.plot}")


if __name__ == "__main__":
    main()

