"""
Merge 24 per-shard CSVs from the freq-init run into a single result file.
Usage: python merge_freqinit.py --results /data/results-freqinit --out /data/results
"""
import argparse, csv, glob, os
import numpy as np


def merge(results_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(results_dir, "chb*/loso_fate.csv")))
    if not shards:
        print(f"No shard CSVs found under {results_dir}/chb*/loso_fate.csv")
        return

    rows = []
    for path in shards:
        with open(path) as f:
            for r in csv.DictReader(f):
                if r["patient"] not in ("mean", "std"):
                    rows.append(r)

    rows.sort(key=lambda r: r["patient"])
    keys = list(rows[0].keys())
    metric_keys = [k for k in keys if k != "patient"]
    arr = {k: np.array([float(r[k]) for r in rows if r.get(k)], float)
           for k in metric_keys if rows[0].get(k)}

    out_path = os.path.join(out_dir, "loso_fate_freqinit.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
        w.writerow({"patient": "mean",
                    **{k: f"{arr[k].mean():.4f}" for k in arr}})
        w.writerow({"patient": "std",
                    **{k: f"{arr[k].std():.4f}" for k in arr}})

    print(f"Merged {len(rows)} patients -> {out_path}")
    print(f"Mean AUC: {arr['auc'].mean():.4f} ± {arr['auc'].std():.4f}")
    print(f"Sens: {arr['sensitivity'].mean():.4f}  Spec: {arr['specificity'].mean():.4f}")

    # Gate weights
    gate_shards = sorted(glob.glob(os.path.join(results_dir, "chb*/gate_weights_fate.csv")))
    if gate_shards:
        grows = []
        for path in gate_shards:
            with open(path) as f:
                grows.extend(r for r in csv.DictReader(f))
        grows.sort(key=lambda r: r["patient"])
        gpath = os.path.join(out_dir, "gate_weights_fate_freqinit.csv")
        with open(gpath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=grows[0].keys())
            w.writeheader()
            w.writerows(grows)
        print(f"Gate weights -> {gpath}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/results-freqinit")
    ap.add_argument("--out", default="/data/results")
    args = ap.parse_args()
    merge(args.results, args.out)
