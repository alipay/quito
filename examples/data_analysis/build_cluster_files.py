#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import json
import argparse
import numpy as np
import pandas as pd
from glob import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def build_cluster_files(item_csv, data_dir, output_dir):
    # get all files from data_dir
    data = pd.read_parquet(data_dir)
    n_clusters = item_csv['cluster'].unique()
    for c in n_clusters:
        c_id = item_csv[item_csv['cluster'] == c]['item_id']
        logging.info(f'There are {len(c_id)} items in cluster {c}')
        c_data = data[data['item_id'].isin(c_id)].reset_index(drop=True)
        logging.info(f'The data contains {c_data.shape} data points')
        # write c_data to file
        output_path = os.path.join(output_dir, f'open_data_{int(c)}.parquet')
        c_data.to_parquet(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--item_csv', type=str, default='item_clusters.csv',
                        help='Input JSON file with quality results')
    parser.add_argument('--data_dir', type=str, default='datasets/full_data',
                        help='data directory for full data')
    parser.add_argument('--output_dir', type=str, default='datasets/cluster_data',
                        help='output data directory for cluster data')
    args = parser.parse_args()

    item_csv = pd.read_csv(args.item_csv)
    build_cluster_files(item_csv, args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()