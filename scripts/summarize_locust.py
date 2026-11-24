"""
Turn the Locust CSV files into a README table and a comparison chart.

Usage (run_load_tests.sh calls this for you):
    python scripts/summarize_locust.py --replicas 1 2 4 --users 100 --duration 2m

Reads:   results/locust_<N>_containers_stats.csv (+ _instances.json if present)
Writes:  results/load_test_summary.md     -> paste into README.md
         results/load_test_comparison.png -> add to README.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RESULTS = Path("results")


def load_row(n: int, endpoint: str = "/predict") -> dict | None:
    """Read one container count's Locust stats and return the numbers we report."""
    path = RESULTS / f"locust_{n}_containers_stats.csv"
    if not path.exists():
        print(f"missing {path}, skipped")
        return None
    df = pd.read_csv(path)
    row = df[df["Name"] == endpoint]
    if row.empty:
        row = df[df["Name"] == "Aggregated"]
    r = row.iloc[0]
    total = int(r["Request Count"])
    failures = int(r["Failure Count"])

    # How evenly nginx spread the load (from the X-Instance header counted by locustfile.py)
    inst_path = RESULTS / f"locust_{n}_containers_instances.json"
    spread = ""
    if inst_path.exists():
        counts = {k: v for k, v in json.loads(inst_path.read_text()).items() if k != "unknown"}
        if counts:
            shares = sorted((100 * v / sum(counts.values()) for v in counts.values()), reverse=True)
            spread = " / ".join(f"{s:.0f}%" for s in shares)

    return {
        "Containers": n,
        "Requests": total,
        "Failures": failures,
        "Failure %": round(100 * failures / total, 2) if total else 0.0,
        "Requests/s": round(float(r["Requests/s"]), 1),
        "Avg (ms)": round(float(r["Average Response Time"])),
        "Median (ms)": round(float(r["Median Response Time"])),
        "p95 (ms)": round(float(r["95%"])),
        "p99 (ms)": round(float(r["99%"])),
        "Max (ms)": round(float(r["Max Response Time"])),
        "Load split per replica": spread,
    }


def make_chart(df: pd.DataFrame, out: Path, users: str) -> None:
    """Two panels side by side (one measure each, never two y-axes on one chart):
    latency (median vs p95) and throughput."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange, ink, grid = "#2a78d6", "#eb6834", "#52514e", "#e6e5e0"
    x = range(len(df))
    labels = [f"{n} container{'s' if n > 1 else ''}" for n in df["Containers"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: median and p95 latency, grouped bars (2 px white gap between the bars)
    w = 0.38
    b1 = ax1.bar([i - w / 2 for i in x], df["Median (ms)"], w, color=blue, label="Median", edgecolor="white", linewidth=2)
    b2 = ax1.bar([i + w / 2 for i in x], df["p95 (ms)"], w, color=orange, label="95th percentile", edgecolor="white", linewidth=2)
    ax1.bar_label(b1, fmt="%d", color=ink, fontsize=9, padding=2)
    ax1.bar_label(b2, fmt="%d", color=ink, fontsize=9, padding=2)
    ax1.set_title("/predict response time (lower is better)", loc="left", fontsize=12, weight="bold")
    ax1.set_ylabel("milliseconds", color=ink)
    ax1.legend(frameon=False, loc="upper right")

    # Panel 2: throughput
    b3 = ax2.bar(list(x), df["Requests/s"], 0.55, color=blue, edgecolor="white", linewidth=2)
    ax2.bar_label(b3, fmt="%.1f", color=ink, fontsize=9, padding=2)
    ax2.set_title("/predict throughput (higher is better)", loc="left", fontsize=12, weight="bold")
    ax2.set_ylabel("requests per second", color=ink)

    for ax in (ax1, ax2):
        ax.set_xticks(list(x), labels)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(grid)
        ax.tick_params(colors=ink)
        ax.yaxis.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    fig.suptitle(f"Locust flood test: {users} concurrent users", x=0.01, ha="left", fontsize=13, color=ink)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"chart  -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--users", default="100")
    parser.add_argument("--duration", default="2m")
    args = parser.parse_args()

    rows = [r for n in args.replicas if (r := load_row(n))]
    if not rows:
        raise SystemExit("No Locust results found in results/. Run scripts/run_load_tests.sh first.")
    df = pd.DataFrame(rows)

    # Speed-up relative to the first (smallest) setup
    base = df.iloc[0]
    df["Throughput vs 1st"] = (df["Requests/s"] / base["Requests/s"]).round(2).astype(str) + "×"

    # Markdown table written by hand (avoids needing the extra 'tabulate' package)
    cols = list(df.columns)
    table = "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                      + ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)])
    summary = (f"**Setup:** {args.users} simulated users, {args.duration} per run, each user sends a leaf "
               f"image to `/predict` every 1–2 s (plus occasional `/health`). Each API container is limited "
               f"to 1 CPU / 2 GB RAM behind an nginx `least_conn` load balancer.\n\n{table}\n\n"
               f"![Load test comparison](results/load_test_comparison.png)\n")
    out_md = RESULTS / "load_test_summary.md"
    out_md.write_text(summary)
    print(table)
    print(f"\ntable  -> {out_md}")
    make_chart(df, RESULTS / "load_test_comparison.png", args.users)


if __name__ == "__main__":
    main()
