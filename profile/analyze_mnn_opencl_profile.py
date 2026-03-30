#!/usr/bin/env python3
"""Analyze MNN OpenCL profiling CSVs.

Expected CSVs
-------------
run.csv columns:
  frame_id,run_total_us,enqueue_us,wait_us,readback_us,write_us,map_us,unmap_us,
  kernel_gpu_us_sum,kernel_submit_delay_us_sum,kernel_start_delay_us_sum,
  num_kernels,fallback_count,resize_happened

kernel.csv columns:
  frame_id,seq,op_name,kernel_name,gws,lws,
  queued_ns,submit_ns,start_ns,end_ns,
  queue_to_submit_us,submit_to_start_us,gpu_exec_us,total_event_us

phase.csv columns (optional):
  frame_id,phase,start_ns,end_ns,duration_us

Examples
--------
python analyze_mnn_opencl_profile.py --run run.csv --kernel kernel.csv
python analyze_mnn_opencl_profile.py --run run.csv --kernel kernel.csv --phase phase.csv --top 20
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import pandas as pd


def percentile(series: pd.Series, q: float) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.quantile(q))


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    if a.nunique() <= 1 or b.nunique() <= 1:
        return float("nan")
    return float(a.corr(b))


def classify_frame(row: pd.Series) -> str:
    wait_us = float(row.get("wait_us", 0) or 0)
    gpu_us = float(row.get("kernel_gpu_us_sum", 0) or 0)
    submit_us = float(row.get("kernel_submit_delay_us_sum", 0) or 0)
    start_us = float(row.get("kernel_start_delay_us_sum", 0) or 0)
    readback_us = float(row.get("readback_us", 0) or 0)
    write_us = float(row.get("write_us", 0) or 0)
    fallback_count = int(row.get("fallback_count", 0) or 0)
    resize_happened = int(row.get("resize_happened", 0) or 0)

    if fallback_count > 0:
        return "CPU fallback likely"
    if resize_happened > 0:
        return "Resize / realloc / dynamic shape likely"
    if readback_us + write_us > max(gpu_us, 1) * 0.5 and (readback_us + write_us) > 500:
        return "Transfer / blocking I/O likely"
    if wait_us > gpu_us * 0.8 and submit_us + start_us > gpu_us * 0.5:
        return "Queue submit/start delay or hidden sync"
    if gpu_us > max(submit_us + start_us, 1) * 1.5:
        return "GPU execution variability"
    if wait_us > 1000 and gpu_us < wait_us * 0.5:
        return "CPU scheduling / finish wait / external contention"
    return "Mixed / unclear"


def summarize_run(run_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "run_total_us",
        "enqueue_us",
        "wait_us",
        "readback_us",
        "write_us",
        "map_us",
        "unmap_us",
        "kernel_gpu_us_sum",
        "kernel_submit_delay_us_sum",
        "kernel_start_delay_us_sum",
    ]
    present = [c for c in cols if c in run_df.columns]
    rows = []
    for c in present:
        s = pd.to_numeric(run_df[c], errors="coerce").fillna(0)
        rows.append(
            {
                "metric": c,
                "mean_us": round(float(s.mean()), 2),
                "p50_us": round(percentile(s, 0.50), 2),
                "p90_us": round(percentile(s, 0.90), 2),
                "p95_us": round(percentile(s, 0.95), 2),
                "p99_us": round(percentile(s, 0.99), 2),
                "max_us": round(float(s.max()), 2),
            }
        )
    return pd.DataFrame(rows)


def top_spikes(run_df: pd.DataFrame, top: int) -> pd.DataFrame:
    df = run_df.copy()
    df = df.sort_values("run_total_us", ascending=False).head(top)
    df["diagnosis"] = df.apply(classify_frame, axis=1)
    return df


def summarize_kernels(kernel_df: pd.DataFrame, top: int) -> pd.DataFrame:
    grouped = (
        kernel_df.groupby(["kernel_name"], dropna=False)
        .agg(
            calls=("kernel_name", "count"),
            gpu_mean_us=("gpu_exec_us", "mean"),
            gpu_p95_us=("gpu_exec_us", lambda s: s.quantile(0.95)),
            gpu_max_us=("gpu_exec_us", "max"),
            submit_mean_us=("submit_to_start_us", "mean"),
            queue_submit_mean_us=("queue_to_submit_us", "mean"),
            total_event_mean_us=("total_event_us", "mean"),
        )
        .reset_index()
        .sort_values(["gpu_p95_us", "gpu_max_us"], ascending=False)
        .head(top)
    )
    for c in grouped.columns[1:]:
        grouped[c] = grouped[c].map(lambda x: round(float(x), 2))
    return grouped


def per_frame_kernel_rollup(kernel_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        kernel_df.groupby("frame_id", dropna=False)
        .agg(
            kernel_gpu_us_sum=("gpu_exec_us", "sum"),
            kernel_submit_delay_us_sum=("submit_to_start_us", "sum"),
            kernel_queue_to_submit_us_sum=("queue_to_submit_us", "sum"),
            num_kernels=("kernel_name", "count"),
            max_kernel_gpu_us=("gpu_exec_us", "max"),
        )
        .reset_index()
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Path to run.csv")
    parser.add_argument("--kernel", required=True, help="Path to kernel.csv")
    parser.add_argument("--phase", required=False, help="Path to phase.csv")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--outdir", default="analysis_out")
    args = parser.parse_args()

    run_df = pd.read_csv(args.run)
    kernel_df = pd.read_csv(args.kernel)
    phase_df: Optional[pd.DataFrame] = pd.read_csv(args.phase) if args.phase else None

    kernel_rollup = per_frame_kernel_rollup(kernel_df)
    merged = run_df.merge(kernel_rollup, on="frame_id", how="left", suffixes=("", "_k"))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_df = summarize_run(merged)
    spikes_df = top_spikes(merged, args.top)
    kernels_df = summarize_kernels(kernel_df, args.top)

    summary_df.to_csv(outdir / "summary.csv", index=False)
    spikes_df.to_csv(outdir / "top_spikes.csv", index=False)
    kernels_df.to_csv(outdir / "kernel_summary.csv", index=False)

    corr_rows = []
    if "run_total_us" in merged.columns:
        for c in [
            "kernel_gpu_us_sum",
            "kernel_submit_delay_us_sum",
            "kernel_queue_to_submit_us_sum",
            "wait_us",
            "readback_us",
            "write_us",
            "map_us",
            "unmap_us",
            "fallback_count",
            "resize_happened",
        ]:
            if c in merged.columns:
                corr_rows.append(
                    {"metric": c, "corr_with_run_total": round(safe_corr(merged["run_total_us"], merged[c]), 4)}
                )
    corr_df = pd.DataFrame(corr_rows).sort_values("corr_with_run_total", ascending=False)
    corr_df.to_csv(outdir / "correlations.csv", index=False)

    if phase_df is not None and set(["frame_id", "phase", "duration_us"]).issubset(phase_df.columns):
        phase_summary = (
            phase_df.groupby("phase", dropna=False)["duration_us"]
            .agg(["count", "mean", lambda s: s.quantile(0.95), "max"])
            .reset_index()
        )
        phase_summary.columns = ["phase", "count", "mean_us", "p95_us", "max_us"]
        phase_summary.to_csv(outdir / "phase_summary.csv", index=False)

    print("=== Summary ===")
    print(summary_df.to_string(index=False))
    print()

    print("=== Correlation with run_total_us ===")
    if len(corr_df):
        print(corr_df.to_string(index=False))
    else:
        print("No correlation rows available")
    print()

    print("=== Top spike frames ===")
    cols = [
        c for c in [
            "frame_id", "run_total_us", "wait_us", "kernel_gpu_us_sum",
            "kernel_submit_delay_us_sum", "kernel_queue_to_submit_us_sum",
            "readback_us", "write_us", "fallback_count", "resize_happened", "diagnosis"
        ] if c in spikes_df.columns
    ]
    print(spikes_df[cols].to_string(index=False))
    print()

    print("=== Top kernels by p95 GPU time ===")
    print(kernels_df.to_string(index=False))
    print()

    print(f"Wrote analysis CSVs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
