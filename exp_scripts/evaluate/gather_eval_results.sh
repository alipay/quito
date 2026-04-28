#!/bin/bash

# Define arrays
lws=(96 576 1024)
fhs=(48 288 512)
features=(S M)

# Nested loop with console logging only
for lw in "${lws[@]}"; do
    for fh in "${fhs[@]}"; do
        for feature in "${features[@]}"; do
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running: lw=$lw, fh=$fh, features=$feature"
            python examples/aggregate_results.py --lw $lw --fh $fh --features $feature --item_csv_path 'examples/item_csv.csv'
            
            # Check exit status and report
            if [ $? -eq 0 ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ SUCCESS: lw=$lw, fh=$fh, features=$feature"
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ FAILED: lw=$lw, fh=$fh, features=$feature"
            fi
            echo "---"
        done
    done
done

echo "All jobs completed."