"""Performance and memory benchmarking suite for Strataforge AI."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import psutil

from strataforge.demo.generator import generate_baseline_dataset, generate_drifted_dataset
from strataforge.drift.detector import detect_drift
from strataforge.ml.anomaly import StrataforgeAnomalyModel
from strataforge.profiling.profiler import profile_dataset
from strataforge.reporting.html_generator import generate_html_report
from strataforge.reporting.summary import compute_quality_score


def get_process_memory_mb() -> float:
    """Return current process resident memory in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def run_benchmark(n_rows: int = 100_000) -> dict[str, float | int]:
    """Execute standardized benchmark on synthetic dataset."""
    results: dict[str, float | int] = {"n_rows": n_rows}

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / f"bench_{n_rows}.csv"
        parquet_path = Path(tmp_dir) / f"bench_{n_rows}.parquet"

        # 1. Dataset Generation Benchmark
        t0 = time.perf_counter()
        df = generate_baseline_dataset(n_rows=n_rows, seed=42)
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        results["file_size_csv_mb"] = round(csv_path.stat().st_size / (1024 * 1024), 2)
        results["file_size_parquet_mb"] = round(parquet_path.stat().st_size / (1024 * 1024), 2)

        # 2. In-Memory Profiling Benchmark
        mem_before = get_process_memory_mb()
        t0 = time.perf_counter()
        prof_in_mem = profile_dataset(df)
        t_prof_in_mem = time.perf_counter() - t0
        mem_after = get_process_memory_mb()
        results["time_profile_in_mem_sec"] = round(t_prof_in_mem, 3)
        results["mem_delta_in_mem_mb"] = round(max(0.0, mem_after - mem_before), 2)

        # 3. Chunked Streaming Profiling Benchmark
        mem_before_chunk = get_process_memory_mb()
        t0 = time.perf_counter()
        _ = profile_dataset(csv_path, chunk_size=25_000)
        t_prof_chunked = time.perf_counter() - t0
        mem_after_chunk = get_process_memory_mb()
        results["time_profile_chunked_sec"] = round(t_prof_chunked, 3)
        results["mem_delta_chunked_mb"] = round(max(0.0, mem_after_chunk - mem_before_chunk), 2)

        # 4. Statistical Drift Calculation Benchmark (PSI + KS test on all columns)
        drift_df = generate_drifted_dataset(df.iloc[: min(len(df), 25_000)], seed=1337)
        t0 = time.perf_counter()
        drift_res = detect_drift(df.iloc[: min(len(df), 25_000)], drift_df)
        t_drift = time.perf_counter() - t0
        results["time_drift_analysis_sec"] = round(t_drift, 3)

        # 5. Isolation Forest Anomaly Detection (Fit & Score on continuous numerical features)
        t0 = time.perf_counter()
        model = StrataforgeAnomalyModel(contamination=0.05, random_state=42)
        model.fit(df.iloc[: min(len(df), 50_000)])
        is_anom, scores = model.score(df.iloc[: min(len(df), 50_000)])
        t_ml = time.perf_counter() - t0
        results["time_ml_fit_score_sec"] = round(t_ml, 3)

        # 6. Standalone HTML Report Generation
        t0 = time.perf_counter()
        rep_path = Path(tmp_dir) / "bench_report.html"
        score_sum = compute_quality_score(drift_metrics=drift_res.metrics, total_rows=n_rows)
        generate_html_report(
            profile=prof_in_mem,
            score_summary=score_sum,
            output_path=rep_path,
            drift_metrics=drift_res.metrics,
        )
        t_html = time.perf_counter() - t0
        results["time_html_report_sec"] = round(t_html, 3)
        results["html_report_size_kb"] = round(rep_path.stat().st_size / 1024, 2)

    return results


def main() -> None:
    print("=" * 60)
    print("STRATAFORGE AI — Performance & Memory Benchmark Suite")
    print("=" * 60)

    for n in [10_000, 50_000, 100_000]:
        print(f"\nRunning benchmark for N = {n:,} rows...")
        res = run_benchmark(n)
        for k, v in res.items():
            print(f"  • {k:<30}: {v}")


if __name__ == "__main__":
    main()
