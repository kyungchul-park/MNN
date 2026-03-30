#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def numeric_cols(df):
    cols = []
    for c in df.columns:
        if c == 'monotonic_ns':
            continue
        try:
            pd.to_numeric(df[c], errors='raise')
            cols.append(c)
        except Exception:
            pass
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--metrics', required=True)
    ap.add_argument('--out', default='mnn_sysfs_correlation.csv')
    args = ap.parse_args()

    run = pd.read_csv(args.run)
    metrics = pd.read_csv(args.metrics)

    for c in ['execute_begin_ns', 'execute_end_ns']:
        if c not in run.columns:
            raise SystemExit(f'missing {c} in run csv')

    metrics['monotonic_ns'] = pd.to_numeric(metrics['monotonic_ns'], errors='coerce')
    ncols = numeric_cols(metrics)
    for c in ncols:
        metrics[c] = pd.to_numeric(metrics[c], errors='coerce')

    rows = []
    for _, r in run.iterrows():
        beg = r['execute_begin_ns']
        end = r['execute_end_ns']
        win = metrics[(metrics['monotonic_ns'] >= beg) & (metrics['monotonic_ns'] <= end)]
        out = {
            'frame_id': r.get('frame_id', np.nan),
            'execute_total_us': r.get('execute_total_us', np.nan),
            'kernel_gpu_us_sum': r.get('kernel_gpu_us_sum', np.nan),
            'kernel_queue_to_submit_us_sum': r.get('kernel_queue_to_submit_us_sum', np.nan),
            'kernel_submit_to_start_us_sum': r.get('kernel_submit_to_start_us_sum', np.nan),
            'wait_us': r.get('wait_us', np.nan),
            'readback_us': r.get('readback_us', np.nan),
            'write_us': r.get('write_us', np.nan),
            'samples': len(win),
        }
        for c in ncols:
            if len(win) > 0:
                out[c + '_avg'] = win[c].mean()
                out[c + '_max'] = win[c].max()
                out[c + '_min'] = win[c].min()
            else:
                out[c + '_avg'] = np.nan
                out[c + '_max'] = np.nan
                out[c + '_min'] = np.nan
        rows.append(out)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
