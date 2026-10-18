"""Generate realistic traffic against the API (feeds the monitoring dashboard).

Patients are sampled from the cleaned dataset with small jitter; a share of
requests is deliberately invalid so validation-error metrics are exercised.

    python scripts/load_test.py --url http://localhost:8000 --requests 300 --concurrency 8
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402


def make_payloads(n: int, invalid_share: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    df = pd.read_csv(config.CLEAN_DATA_FILE).drop(columns=[config.TARGET])
    payloads = []
    for _ in range(n):
        row = df.sample(1, random_state=rng.randint(0, 10**6)).iloc[0].to_dict()
        record = {}
        for key, value in row.items():
            if pd.isna(value):
                continue  # optional fields (ca/thal) left out, like a real client would
            record[key] = int(value) if key in config.CATEGORY_LEVELS or key == "age" else float(value)
        record["chol"] = round(min(600.0, max(110.0, record["chol"] * rng.uniform(0.95, 1.05))), 1)
        if rng.random() < invalid_share:
            record["age"] = rng.choice([150, -1, 5])  # out-of-range -> HTTP 422
        payloads.append(record)
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--invalid-share", type=float, default=0.05)
    parser.add_argument("--host-header", default=None, help="Host header, e.g. for an Ingress")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    headers = {"Host": args.host_header} if args.host_header else {}
    session = requests.Session()
    payloads = make_payloads(args.requests, args.invalid_share, args.seed)

    def call(payload: dict) -> tuple[int, float, int | None]:
        start = time.perf_counter()
        response = session.post(f"{args.url}/predict", json=payload, headers=headers, timeout=10)
        elapsed = time.perf_counter() - start
        prediction = response.json().get("prediction") if response.status_code == 200 else None
        return response.status_code, elapsed, prediction

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(call, payloads))
    wall = time.perf_counter() - started

    latencies = sorted(r[1] * 1000 for r in results)
    statuses = Counter(r[0] for r in results)
    predictions = Counter(r[2] for r in results if r[2] is not None)
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    print(f"requests={len(results)} wall={wall:.1f}s throughput={len(results) / wall:.1f} req/s")
    print(f"status codes={dict(statuses)} predictions={dict(predictions)}")
    print(f"latency ms: p50={statistics.median(latencies):.1f} p95={p95:.1f} max={latencies[-1]:.1f}")
    if statuses.get(500) or statuses.get(503):
        sys.exit(1)


if __name__ == "__main__":
    main()
