"""
Efficiency table: parameters, FLOPs (thop if available), and inference latency
on CPU and GPU for FATE, its ablations, and the baselines.

  python efficiency.py --out ../../data      # writes efficiency.csv

Latency is measured at batch=1 (single-window, edge scenario) and batch=64
(throughput), reported in ms per window.
"""
import os
import csv
import time
import argparse
import torch

from model import FATE, ABLATIONS
from baselines import build_model


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def flops_per_window(m):
    try:
        from thop import profile
        macs, _ = profile(m, inputs=(torch.randn(1, 18, 1024),), verbose=False)
        return float(macs)            # multiply-accumulates for one window
    except Exception:
        return None


@torch.no_grad()
def latency_ms(m, device, batch, iters=40, warmup=8):
    m = m.to(device).eval()
    x = torch.randn(batch, 18, 1024, device=device)
    for _ in range(warmup):
        m(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        m(x)
    if device == "cuda":
        torch.cuda.synchronize()
    total = (time.time() - t) / iters      # s per batch
    return total / batch * 1e3             # ms per window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    has_cuda = torch.cuda.is_available()

    variants = [("fate", "none"), ("eegnet", "none"), ("cnnlstm", "none"),
                ("runet", "none")] + [("fate", a) for a in ABLATIONS if a != "none"]

    rows = []
    for name, abl in variants:
        m = build_model(name, abl)
        tag = name + ("" if abl == "none" else f"_{abl}")
        row = {"model": tag, "params": count_params(m),
               "mflops": (lambda f: f"{f/1e6:.1f}" if f else "n/a")(flops_per_window(m)),
               "cpu_ms_b1": f"{latency_ms(m, 'cpu', 1):.2f}",
               "cpu_ms_b64": f"{latency_ms(m, 'cpu', 64):.3f}"}
        if has_cuda:
            row["gpu_ms_b1"] = f"{latency_ms(m, 'cuda', 1):.3f}"
            row["gpu_ms_b64"] = f"{latency_ms(m, 'cuda', 64):.4f}"
        rows.append(row)
        print(row)

    keys = ["model", "params", "mflops", "cpu_ms_b1", "cpu_ms_b64"] + \
           (["gpu_ms_b1", "gpu_ms_b64"] if has_cuda else [])
    path = os.path.join(args.out, "efficiency.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
