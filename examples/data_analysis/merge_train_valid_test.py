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

def merge_train_valid_test(data_dir, output_dir):
    # merge train valid test for min 
    for f_p in glob(os.path.join(data_dir, 'train_min_p*.parquet')):
        n = f_p.split('/')[-1]
        part = n.split('.')[0].split('_')[-1]
        test_n = f'test_label_min_{part}.parquet'
        valid_n = f'valid_min_{part}.parquet'
        print(f'processing {f_p}, part {part}, test name {test_n}, valid name {valid_n}')

        train_df = pd.read_parquet(f_p)
        valid_df = pd.read_parquet(os.path.join(data_dir, valid_n))
        test_df = pd.read_parquet(os.path.join(data_dir, test_n))

        full_df = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
        full_df.to_parquet(os.path.join(output_dir, f'open_min_{part}.parquet'))

    # merge train valid test for hour
    for f_p in glob(os.path.join(data_dir, 'hour_train_hour_p*.parquet')):
        n = f_p.split('/')[-1]
        part = n.split('.')[0].split('_')[-1]
        test_n = f'hour_test_label_hour_{part}.parquet'
        valid_n = f'hour_valid_hour_{part}.parquet'
        print(f'processing {f_p}, part {part}, test name {test_n}, valid name {valid_n}')

        train_df = pd.read_parquet(f_p)
        valid_df = pd.read_parquet(os.path.join(data_dir, valid_n))
        test_df = pd.read_parquet(os.path.join(data_dir, test_n))

        full_df = pd.concat([train_df, valid_df, test_df], axis=0).reset_index(drop=True)
        full_df.to_parquet(os.path.join(output_dir, f'open_hour_{part}.parquet'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/parquet_data',
                        help='data directory for full data')
    parser.add_argument('--output_dir', type=str, default='datasets/full_data',
                        help='output data directory for cluster data')
    args = parser.parse_args()

    merge_train_valid_test(item_csv, args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()

