# aggregating results, the input directory must have structure:
# ------ input_dir
# ---------------- model_1
# ------------------------ {lw}_{fh}_{features}
# --------------------------------- EVALUATE
# ------------------------------------------ ver_{trail_number}
# ------------------------------------------------------------- eval_results_{model_name}.json
# ---------------- model_2
# ...

import sys
import os
import json
import pandas as pd
import numpy as np
import argparse

from glob import glob


def parse_single_results(results_path, item_csv):
    with open(results_path, 'r') as f:
        results_json = json.load(f)
    # flatten results json
    flatten_lst = []
    for i in results_json['final_results']:
        new_dict = {'user_id': i['user_id']}
        new_dict.update(i['metrics'])
        flatten_lst.append(new_dict)

    result_df = pd.DataFrame(flatten_lst)
    metrics_cols = result_df.columns[~result_df.columns.isin(['user_id'])]
    mapping = item_csv[['group_name']].to_dict()
    result_df['group_name'] = result_df['user_id'].map(mapping['group_name'])
    # compute per df agg stats
    mean_df = result_df.groupby(['group_name'])[metrics_cols].mean().reset_index()
    std_df = result_df.groupby(['group_name'])[metrics_cols].std().reset_index()
    median_df = result_df.groupby(['group_name'])[metrics_cols].median().reset_index()

    full_results_dict = {
        'original': result_df,
        'mean': mean_df,
        'std': std_df,
        'median': median_df
    }

    return full_results_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default='./outputs'
    )
    parser.add_argument(
        "--models",
        nargs='+',
        type=str,
        default=['chronos', 'tirex', 'timesfm', 'patchtst', 'dlinear', 'crossformer', 'pyraformer', 'tsmixer',
                 'itransformer', 'snaive', 'es']
    )
    parser.add_argument(
        "--fh",
        type=int,
        default=288
    )
    parser.add_argument(
        "--lw",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--features",
        type=str,
        default='M'
    )
    parser.add_argument(
        "--item_csv_path",
        type=str,
        default='examples/item_clusters.csv'
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./outputs/aggregated_results'
    )

    args = parser.parse_args()

    print(f'Aggregating results for {args.lw}_{args.fh}_{args.features}')
    print('==========================================================')
    # check if the input directory exists
    if not os.path.exists(args.results_dir):
        raise ValueError(f"Input directory {args.results_dir} does not exist")

    # get item_csv, this contains the meta information for each item id
    item_csv = pd.read_csv(args.item_csv_path)
    item_csv.index = item_csv['item_id']

    models = []
    for model in args.models:
        model_dir = os.path.join(args.results_dir, model)
        if os.path.exists(model_dir):
            models.append(model)
        else:
            print(f"Model {model} does not exist in the input directory, SKIPPING!!!!")

    results_std_dfs = {}
    results_mean_dfs = {}
    results_median_dfs = {}
    for model in models:
        setting = f'{args.lw}_{args.fh}_{args.features}'
        pattern = os.path.join(args.results_dir, model, setting, 'EVALUATE', 'ver_*')
        trials = sorted(glob(pattern))
        if trials:
            # get the latest trial results
            trial = trials[-1]
            results_pattern = os.path.join(trial, f'eval_results_*.json')
            results_paths = glob(results_pattern)
            if not results_paths:
                print(f"Model {model} does not have any results in the results directory {trial}, SKIPPING!!!!")
                continue
            else:
                results_path = results_paths[0]
        else:
            results_path = None
            print(
                f"Model {model} does not have any trials starts with ver_ in the input directory {pattern}, SKIPPING!!!!")
            continue

        results_df_dict = parse_single_results(results_path, item_csv)
        results_mean_dfs[model] = results_df_dict['mean']
        results_median_dfs[model] = results_df_dict['median']
        results_std_dfs[model] = results_df_dict['std']

    result_std_df = pd.concat(results_std_dfs, names=['model']).reset_index(level=0)
    result_mean_df = pd.concat(results_mean_dfs, names=['model']).reset_index(level=0)
    result_median_df = pd.concat(results_median_dfs, names=['model']).reset_index(level=0)

    # calculate rank for each metric
    metric_names = result_mean_df.columns[~result_mean_df.columns.isin(['group_name', 'model'])]
    for metric in metric_names:
        result_mean_df[f'{metric}_rank'] = result_mean_df.groupby(['group_name'])[metric].rank(ascending=True)
        result_median_df[f'{metric}_rank'] = result_median_df.groupby(['group_name'])[metric].rank(ascending=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    result_std_df.to_csv(os.path.join(args.output_dir, f'results_std_df_{args.lw}_{args.fh}_{args.features}.csv'),
                         index=False)
    result_mean_df.to_csv(os.path.join(args.output_dir, f'results_mean_df_{args.lw}_{args.fh}_{args.features}.csv'),
                          index=False)
    result_median_df.to_csv(os.path.join(args.output_dir, f'results_median_df_{args.lw}_{args.fh}_{args.features}.csv'),
                            index=False)

    print(f'Aggregating results for {args.lw}_{args.fh}_{args.features} done. Results saved to {args.output_dir}')


if __name__ == '__main__':
    main()




