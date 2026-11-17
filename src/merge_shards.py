"""
Merge sharded LOSO outputs into a single result. Each shard wrote
<results>/<shard>/loso_fate.csv (its held-out folds) and gate_weights_fate.csv.
This concatenates all per-patient rows, recomputes mean/std across ALL folds, and
writes <results>/loso_fate.csv + <results>/gate_weights_fate.csv identical in shape
to a single-GPU run.

Usage: python merge_shards.py --results /data/results --tag fate
"""
import os
import csv
import glob
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/results")
    ap.add_argument("--tag", default="fate")
    args = ap.parse_args()

    metric_keys = ["patient", "auc", "sensitivity", "specificity", "f1", "threshold"]
    rows = []
    for f in sorted(glob.glob(os.path.join(args.results, "*", f"loso_{args.tag}.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["patient"] in ("mean", "std"):
                    continue                       # per-shard aggregates: drop, recompute
                rows.append(r)
    if not rows:
        raise SystemExit(f"no shard CSVs under {args.results}/*/loso_{args.tag}.csv")

    rows.sort(key=lambda r: r["patient"])
    arr = {k: np.array([float(r[k]) for r in rows], float) for k in metric_keys[1:5]}
    out = os.path.join(args.results, f"loso_{args.tag}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=metric_keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in metric_keys})
        w.writerow({"patient": "mean", **{k: f"{arr[k].mean():.4f}" for k in arr}})
        w.writerow({"patient": "std", **{k: f"{arr[k].std():.4f}" for k in arr}})
    print(f"merged {len(rows)} folds -> {out}")
    print(f"  mean AUC={arr['auc'].mean():.4f}+/-{arr['auc'].std():.4f}  "
          f"sens={arr['sensitivity'].mean():.3f}  spec={arr['specificity'].mean():.3f}  "
          f"f1={arr['f1'].mean():.3f}")

    # gate weights (interpretability), if present
    grows, header = [], None
    for f in sorted(glob.glob(os.path.join(args.results, "*", f"gate_weights_{args.tag}.csv"))):
        with open(f) as fh:
            rd = list(csv.reader(fh))
            if not rd:
                continue
            header = rd[0]
            grows += rd[1:]
    if grows and header:
        grows.sort(key=lambda r: r[0])
        gout = os.path.join(args.results, f"gate_weights_{args.tag}.csv")
        with open(gout, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(grows)
        print(f"merged gate weights -> {gout}")


if __name__ == "__main__":
    main()
