"""
Leave-one-subject-out (LOSO) training for FATE on CHB-MIT.

Usage:
  python train.py --config ../configs/fate_chbmit.yaml
  python train.py --data-root /data/chbmit --cache /data/cache --out ../../data

Writes per-fold and mean metrics to <out>/loso_results.csv (traceable numbers
for the paper). Prints a summary and exits (NRP-friendly: no idle GPU).
"""
import os
import csv
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import data as datamod
from model import FATE
from baselines import build_model
from data import load_patient, loso_splits
from metrics import compute_metrics, best_f1_threshold


def _mlflow(cfg):
    """Lazy-import MLflow if --mlflow is set; return the module or None.

    Import is deferred and wrapped so a missing mlflow package (or an unreachable
    tracking store) degrades to a no-op rather than failing the training run.
    """
    if not cfg.get("mlflow"):
        return None
    try:
        import mlflow
        uri = os.environ.get("MLFLOW_TRACKING_URI", "file:/data/mlruns")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(cfg.get("mlflow_experiment", "seizure-loso"))
        return mlflow
    except Exception as e:                       # pragma: no cover - env dependent
        print(f"  [mlflow disabled: {e}]")
        return None


def focal_loss(logits, target, alpha=0.75, gamma=2.0):
    ce = nn.functional.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    a = torch.where(target == 1, alpha, 1 - alpha)
    return (a * (1 - pt) ** gamma * ce).mean()


def smote_balance(X, y, ratio=0.2, seed=42):
    """Oversample the ictal class to <ratio> of interictal (default 1:5)."""
    try:
        from imblearn.over_sampling import SMOTE
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        target = min(n0, max(n1, int(ratio * n0)))
        if n1 == 0 or target <= n1:
            return X, y
        flat = X.reshape(len(X), -1)
        k = min(5, n1 - 1)
        if k < 1:
            return X, y
        Xr, yr = SMOTE(sampling_strategy={1: target}, k_neighbors=k,
                       random_state=seed).fit_resample(flat, y)
        return Xr.reshape(-1, X.shape[1], X.shape[2]).astype(np.float32), yr.astype(np.int64)
    except Exception as e:
        print(f"  [smote skipped: {e}]")
        return X, y


def run_fold(test_pid, train_pids, cfg, device):
    per_pt = [load_patient(cfg["dirs"][p], cfg["cache"]) for p in train_pids]
    Xtr = np.concatenate([Xp for Xp, _ in per_pt])
    ytr = np.concatenate([yp for _, yp in per_pt])
    # per-window domain (patient) label for domain-adversarial training. Leak-free:
    # only the training patients appear here, so the held-out test patient is never
    # a discriminator class and never contributes a domain gradient.
    dtr = np.concatenate([np.full(len(yp), i, np.int64) for i, (_, yp) in enumerate(per_pt)])
    n_domains = len(train_pids)
    Xte, yte = load_patient(cfg["dirs"][test_pid], cfg["cache"])
    if len(np.unique(yte)) < 2:
        print(f"  {test_pid}: no positive test windows, skipping")
        return None

    # 90/10 train/val split, SMOTE on train only
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(len(Xtr))
    cut = int(0.9 * len(idx))
    tr, va = idx[:cut], idx[cut:]
    if cfg.get("dann"):
        # keep domain labels aligned with windows -> skip SMOTE (which fabricates
        # windows with no patient identity) and carry (X, y, domain) through the loader.
        Xt, yt, dt = Xtr[tr], ytr[tr], dtr[tr]
    else:
        Xt, yt = smote_balance(Xtr[tr], ytr[tr], cfg["smote_ratio"], cfg["seed"])
        dt = None

    def loader(X, y, bs, shuffle, d=None):
        tensors = [torch.from_numpy(X), torch.from_numpy(y)]
        if d is not None:
            tensors.append(torch.from_numpy(d))
        ds = TensorDataset(*tensors)
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=cfg["workers"], pin_memory=True)

    dl_tr = loader(Xt, yt, cfg["batch_size"], True, dt)
    dl_va = loader(Xtr[va], ytr[va], cfg["batch_size"], False)
    dl_te = loader(Xte, yte, cfg["batch_size"], False)

    if cfg.get("dann"):
        from model_dann import build_dann
        model = build_dann(cfg["model"], cfg["ablation"], n_domains=n_domains,
                           drop=cfg["dropout"], n_steps=cfg["n_steps"]).to(device)
    else:
        model = build_model(cfg["model"], cfg["ablation"], drop=cfg["dropout"],
                            n_steps=cfg["n_steps"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2)

    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_auc, best_state, patience = -1.0, None, 0
    history = []   # per-epoch (train_loss, val_auc) for training-curve figures
    # freq_reg_loss lives on model.decomp (bare backbone) or model.backbone.decomp (DANN)
    decomp_owner = getattr(model, "backbone", model)
    for epoch in range(cfg["max_epochs"]):
        model.train()
        # DANN gradient-reversal strength ramp 0 -> alpha over training (Ganin schedule)
        p = epoch / max(1, cfg["max_epochs"] - 1)
        grl_alpha = cfg.get("dann_alpha", 1.0) * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
        ep_loss, ep_n = 0.0, 0
        for batch in dl_tr:
            if cfg.get("dann"):
                xb, yb, db = batch
                db = db.to(device, non_blocking=True)
            else:
                xb, yb = batch
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                if cfg.get("dann"):
                    logits, dom_logits = model.forward_da(xb, grl_alpha)
                    loss = focal_loss(logits, yb, cfg["focal_alpha"], cfg["focal_gamma"])
                    loss = loss + cfg["dann_lambda"] * nn.functional.cross_entropy(dom_logits, db)
                else:
                    loss = focal_loss(model(xb), yb, cfg["focal_alpha"], cfg["focal_gamma"])
                # hasattr, not isinstance(model, FATE): FATEv3 (model_v3.py) has the same
                # decomp.freq_reg_loss() interface but is a separate class (avoids a
                # model.py <-> model_v3.py circular import).
                if cfg["freq_lambda"] > 0 and hasattr(decomp_owner, "decomp"):
                    loss = loss + cfg["freq_lambda"] * decomp_owner.decomp.freq_reg_loss()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ep_loss += float(loss.detach()); ep_n += 1
        sched.step()
        va_auc = evaluate(model, dl_va, device)["auc"]
        history.append({"epoch": epoch, "train_loss": ep_loss / max(1, ep_n),
                        "val_auc": float(va_auc)})
        if va_auc > best_auc:
            best_auc, best_state, patience = va_auc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break

    if best_state:
        model.load_state_dict(best_state)
    # Operating threshold chosen on the validation split (NOT the test set) to
    # avoid leakage; AUC is threshold-free. Sens/Spec/F1 use this val threshold.
    yv, pv = _scores(model, dl_va, device)
    thr = best_f1_threshold(yv, pv) if len(np.unique(yv)) > 1 else 0.5
    yte_, pte_ = _scores(model, dl_te, device)
    m = compute_metrics(yte_, pte_, threshold=thr)
    m["val_auc"] = float(best_auc)
    if cfg.get("save_preds"):
        # raw arrays + training history for the standard model figures
        # (ROC / PR / confusion / training curves). Small; one file per patient.
        import json as _json
        np.savez_compressed(os.path.join(cfg["out"], f"preds_{test_pid}.npz"),
                            y_true=yte_.astype(np.int8), y_score=pte_.astype(np.float32),
                            threshold=np.float32(thr))
        with open(os.path.join(cfg["out"], f"history_{test_pid}.json"), "w") as _fh:
            _json.dump(history, _fh)
    if cfg.get("save_ckpt"):
        # Real model weights, needed for any figure that inspects actual learned
        # internals (per-band attention maps, filter-bank frequency response) rather
        # than summary scalars. Off by default: ~2.5MB x 24 folds adds up on NRP PVCs.
        torch.save(model.state_dict(), os.path.join(cfg["out"], f"ckpt_{test_pid}.pt"))
    if cfg.get("event_eval"):
        from eventeval import event_level_eval
        ev = event_level_eval(model, cfg["dirs"][test_pid], device, thr, datamod,
                              datamod.WIN / datamod.FS, datamod.STRIDE / datamod.FS)
        m.update(ev)
        print(f"    event: AUPRC={ev['auprc']:.4f} eventSens={ev['event_sensitivity']:.3f} "
              f"FA/h={ev['fa_per_hour']:.2f} ({ev['n_seizures']} sz, {ev['hours']}h)")
    if cfg.get("postproc"):
        # Phase B: tune false-alarm post-processing on calibration training patient(s)
        # (sequential, leak-free w.r.t. the test patient), freeze, score test with SzCORE.
        from eventeval import sequential_windows
        from iterate import sweep_val_files, score_patient_files
        wsec, ssec = datamod.WIN / datamod.FS, datamod.STRIDE / datamod.FS
        calib_pids = [p for p in sorted(train_pids) if p != test_pid][:max(1, cfg.get("calib_n", 2))]
        calib_files = []
        for cp in calib_pids:
            calib_files += sequential_windows(model, cfg["dirs"][cp], device, datamod, wsec, ssec)
        params, valm = sweep_val_files(calib_files, wsec, ssec,
                                       fp_per_day_budget=cfg.get("fp_budget", 10.0))
        test_files = sequential_windows(model, cfg["dirs"][test_pid], device, datamod, wsec, ssec)
        pp = score_patient_files(test_files, wsec, ssec, params)
        m["pp_event_sens"] = pp["sensitivity"]
        m["pp_fp_per_day"] = pp["fp_per_day"]
        m["pp_f1"] = pp["f1"]
        m["pp_params"] = str(params)
        print(f"    postproc (calib={calib_pids}, budget={cfg.get('fp_budget')}/day): "
              f"valFP/day={valm['fp_per_day']:.1f} -> testSens={pp['sensitivity']:.3f} "
              f"testFP/day={pp['fp_per_day']:.1f} F1={pp['f1']:.3f} {params}")
    if cfg.get("gate_analysis") and isinstance(model, FATE) and model.faa is not None:
        import csv as _csv
        from gateanalysis import gate_and_occlusion
        ga = gate_and_occlusion(model, Xte, yte, device)
        with open(os.path.join(cfg["out"], "gate_analysis.csv"), "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["patient", "base_auc",
                        "gI_delta", "gI_theta", "gI_alpha", "gI_beta",
                        "gN_delta", "gN_theta", "gN_alpha", "gN_beta",
                        "occ_delta", "occ_theta", "occ_alpha", "occ_beta"])
            w.writerow([test_pid, f"{ga['base_auc']:.4f}"]
                       + [f"{v:.4f}" for v in ga["gate_ictal"]]
                       + [f"{v:.4f}" for v in ga["gate_inter"]]
                       + [f"{v:.4f}" for v in ga["occ_auc"]])
        print(f"    gate ictal={[round(v,2) for v in ga['gate_ictal']]} "
              f"inter={[round(v,2) for v in ga['gate_inter']]} "
              f"occAUC={[round(v,3) for v in ga['occ_auc']]}")
    if isinstance(model, FATE) and model.faa is not None:
        m["gate"] = mean_gate(model, dl_te, device)     # per-band importance
    print(f"  {test_pid}: AUC={m['auc']:.4f} Sens={m['sensitivity']:.3f} "
          f"Spec={m['specificity']:.3f} F1={m['f1']:.3f} valAUC={m['val_auc']:.4f} thr={thr:.3f}")
    return m


@torch.no_grad()
def _scores(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        ps.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def mean_gate(model, loader, device):
    model.eval()
    gs = []
    for xb, _ in loader:
        _, g = model(xb.to(device), return_gate=True)
        gs.append(g.cpu().numpy())
    return np.concatenate(gs).mean(0)   # (n_bands,) mean gate over test windows


@torch.no_grad()
def evaluate(model, loader, device, full=False):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        ps.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        ys.append(yb.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return compute_metrics(y, p) if full else {"auc": compute_metrics(y, p)["auc"]}


def main():
    # datamod/load_patient are module globals rebound below for --dataset siena;
    # the declaration must precede any use of datamod (e.g. NORM_SCHEME) in main().
    global datamod, load_patient
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/data/chbmit"))
    ap.add_argument("--cache", default=os.environ.get("CACHE_DIR", "/data/cache"))
    ap.add_argument("--out", default="../../data")
    ap.add_argument("--patients", default="", help="training pool, comma list (default: all found)")
    ap.add_argument("--folds", default="", help="held-out test patients to run THIS shard "
                    "(default: all). Training always uses the full pool minus the held-out patient, "
                    "so sharding by --folds is exact LOSO, not a smaller-pool approximation.")
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42,
                    help="global RNG seed (torch, numpy, random) and the SMOTE/split seed. "
                         "Set per shard for multi-seed robustness runs; write each seed to a "
                         "separate --out dir so results do not collide.")
    ap.add_argument("--model", default="fate", choices=["fate", "eegnet", "cnnlstm", "runet", "graphst", "fate_v3", "tfcanet", "tempctx", "tempxf"])
    ap.add_argument("--temporal-steps", type=int, default=8,
                    help="Phase C: number of consecutive intra-window sub-windows the "
                         "temporal-context models (tempctx/tempxf) split each window into")
    ap.add_argument("--ablation", default="none",
                    choices=["none", "no_faa", "no_gate", "fixed_bands", "no_transformer",
                             "nograph", "noscale",
                             "no_cross_band_attn", "no_channel_se", "no_phase", "no_5th_band", "lean"])
    ap.add_argument("--freq-lambda", type=float, default=0.01,
                    help="weight for out-of-band frequency regularization loss (0=off)")
    ap.add_argument("--window-sec", type=float, default=None,
                    help="window length in seconds (overrides loader default; e.g. 2 or 4)")
    ap.add_argument("--stride-sec", type=float, default=None,
                    help="window stride in seconds (default = window/2)")
    ap.add_argument("--dataset", default="chbmit", choices=["chbmit", "siena"],
                    help="which dataset loader to use")
    ap.add_argument("--norm", default="zscore",
                    choices=["zscore", "robust", "mad", "log", "instance"],
                    help="per-channel normalization scheme (normalize.py). Ablation "
                         "of amplitude-invariant schemes vs the z-score baseline; "
                         "targets the chb05-style amplitude-shift hard-tail failure.")
    ap.add_argument("--event-eval", action="store_true",
                    help="also compute AUPRC + event-level sensitivity + false-alarms/hour "
                         "on the held-out patient (uncapped, time-ordered)")
    ap.add_argument("--gate-analysis", action="store_true",
                    help="write gate weights (ictal vs interictal) + band-occlusion AUC "
                         "to <out>/gate_analysis.csv (faithfulness test)")
    ap.add_argument("--postproc", action="store_true",
                    help="Phase B: tune false-alarm post-processing (firing-power+hysteresis+"
                         "refractory) on calibration patient(s), freeze, score test with SzCORE "
                         "-> pp_event_sens / pp_fp_per_day / pp_f1 (implies event-level scoring)")
    ap.add_argument("--fp-budget", type=float, default=10.0,
                    help="target false-alarms/day budget for the post-proc sweep")
    ap.add_argument("--calib-patients", type=int, default=2,
                    help="how many training patients to calibrate post-proc on (sequential)")
    # Artifact dumping is ON BY DEFAULT — we always store the raw data needed for the
    # standard ML figures (ROC / PR / confusion / training curves). Disable only if space-bound.
    ap.add_argument("--save-preds", dest="save_preds", action="store_true", default=True,
                    help="dump per-patient preds_<pid>.npz (y_true,y_score,threshold) + "
                         "history_<pid>.json (per-epoch train_loss,val_auc). ON by default.")
    ap.add_argument("--no-save-preds", dest="save_preds", action="store_false",
                    help="disable prediction/history dumping (not recommended)")
    ap.add_argument("--save-ckpt", action="store_true", default=False,
                    help="also save the trained model state_dict to <out>/ckpt_<pid>.pt "
                         "(off by default; needed for real per-band attention figures, "
                         "see src/spectral_figures.py)")
    ap.add_argument("--dann", action="store_true",
                    help="Phase D2: domain-adversarial training (gradient reversal on a "
                         "patient discriminator over the training patients) on top of the "
                         "chosen --model backbone, for patient-invariant features. Leak-free: "
                         "the held-out test patient is never a discriminator class.")
    ap.add_argument("--dann-alpha", type=float, default=1.0,
                    help="max gradient-reversal strength for --dann (ramped 0->alpha over "
                         "training, DANN schedule)")
    ap.add_argument("--dann-lambda", type=float, default=0.1,
                    help="weight on the domain-adversarial loss added to the focal loss")
    ap.add_argument("--mlflow", action="store_true",
                    help="Phase E: log params, per-fold metrics, and artifacts to MLflow "
                         "(MLFLOW_TRACKING_URI, default file:/data/mlruns). Lazily imported; "
                         "a missing mlflow package is a no-op, never a run failure.")
    ap.add_argument("--mlflow-experiment", default=os.environ.get("MLFLOW_EXPERIMENT", "seizure-loso"),
                    help="MLflow experiment name (default seizure-loso)")
    args = ap.parse_args()
    # apply the chosen normalization scheme to the data loader (cache is keyed per
    # patient, so use a scheme-specific cache dir to avoid mixing normalizations)
    datamod.NORM_SCHEME = args.norm
    if args.norm != "zscore" and args.cache:
        args.cache = os.path.join(args.cache, f"norm_{args.norm}")

    # Select the dataset loader (same LOSO harness, different I/O).
    if args.dataset == "siena":
        import data_siena as datamod
        load_patient = datamod.load_patient
        subj_glob = "PN*"
    else:
        subj_glob = "chb*"

    # Window size is a first-class knob for the adaptive-window analysis. Set it
    # BEFORE any load_patient call so cached npz are keyed to the right cache dir.
    if args.window_sec is not None:
        datamod.WIN = int(args.window_sec * datamod.FS)
        datamod.STRIDE = int((args.stride_sec if args.stride_sec is not None
                              else args.window_sec / 2) * datamod.FS)
    print(f"dataset={args.dataset} window={datamod.WIN/datamod.FS:.1f}s stride={datamod.STRIDE/datamod.FS:.1f}s")

    pdirs = sorted(glob.glob(os.path.join(args.data_root, subj_glob)))
    dirs = {os.path.basename(p): p for p in pdirs if os.path.isdir(p)}
    # training pool = all patients (or --patients override); folds run here = --folds subset
    pool = args.patients.split(",") if args.patients else sorted(dirs)
    pool = [p for p in pool if p in dirs]
    folds = args.folds.split(",") if args.folds else pool
    folds = [p for p in folds if p in dirs]
    assert pool and folds, f"no patient dirs under {args.data_root}"

    # Global seeding for reproducible multi-seed shards (torch, numpy, random).
    import random as _random
    _random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    on_gpu = torch.cuda.is_available()
    if on_gpu:
        torch.backends.cudnn.benchmark = True
    cfg = dict(dirs=dirs, cache=args.cache, seed=args.seed, model=args.model, ablation=args.ablation,
               batch_size=128 if on_gpu else 64, workers=0,
               lr=3e-4, weight_decay=1e-4, dropout=0.25, focal_alpha=0.75,
               focal_gamma=2.0, smote_ratio=0.2, patience=10, max_epochs=args.max_epochs,
               freq_lambda=args.freq_lambda, event_eval=args.event_eval,
               gate_analysis=args.gate_analysis, out=args.out,
               postproc=args.postproc, fp_budget=args.fp_budget, calib_n=args.calib_patients,
               save_preds=args.save_preds, save_ckpt=args.save_ckpt,
               n_steps=args.temporal_steps,
               dann=args.dann, dann_alpha=args.dann_alpha, dann_lambda=args.dann_lambda,
               mlflow=args.mlflow, mlflow_experiment=args.mlflow_experiment)
    device = "cuda" if on_gpu else "cpu"
    tag = args.model + ("" if args.ablation == "none" else f"_{args.ablation}")
    print(f"device={device} {torch.cuda.get_device_name(0) if on_gpu else ''}  "
          f"model={tag}  pool={len(pool)} patients  folds_this_run={folds}")

    os.makedirs(args.out, exist_ok=True)

    # Phase E: optional MLflow run wrapping the whole shard. Lazily imported; a
    # missing package or unreachable store degrades to a no-op (mlf is None).
    mlf = _mlflow(cfg)
    if mlf is not None:
        mlf.start_run(run_name=f"{tag}-{'-'.join(folds)}")
        mlf.log_params({
            "model": args.model, "ablation": args.ablation, "dataset": args.dataset,
            "norm": args.norm, "window_sec": datamod.WIN / datamod.FS,
            "stride_sec": datamod.STRIDE / datamod.FS, "max_epochs": args.max_epochs,
            "lr": cfg["lr"], "weight_decay": cfg["weight_decay"], "dropout": cfg["dropout"],
            "focal_alpha": cfg["focal_alpha"], "focal_gamma": cfg["focal_gamma"],
            "smote_ratio": cfg["smote_ratio"], "freq_lambda": cfg["freq_lambda"],
            "temporal_steps": args.temporal_steps, "dann": args.dann,
            "dann_alpha": args.dann_alpha, "dann_lambda": args.dann_lambda,
            "pool_size": len(pool),
        })

    rows, t0 = [], time.time()
    for test_pid in folds:
        train_pids = [p for p in pool if p != test_pid]
        r = run_fold(test_pid, train_pids, cfg, device)
        if r:
            r["patient"] = test_pid
            rows.append(r)
            if mlf is not None:
                for k in ("auc", "auprc", "sensitivity", "specificity", "f1", "val_auc",
                          "event_sensitivity", "fa_per_hour", "pp_event_sens",
                          "pp_fp_per_day", "pp_f1"):
                    if isinstance(r.get(k), (int, float)):
                        mlf.log_metric(f"{test_pid}_{k}", float(r[k]))
                for art in (f"preds_{test_pid}.npz", f"history_{test_pid}.json"):
                    ap_ = os.path.join(args.out, art)
                    if os.path.exists(ap_):
                        mlf.log_artifact(ap_)

    if rows:
        keys = ["patient", "auc", "auprc", "sensitivity", "specificity", "f1",
                "threshold", "val_auc", "event_sensitivity", "fa_per_hour",
                "pp_event_sens", "pp_fp_per_day", "pp_f1", "pp_params"]
        arr = {k: np.array([r[k] for r in rows], float) for k in ["auc", "sensitivity", "specificity", "f1"]}
        res_path = os.path.join(args.out, f"loso_{tag}.csv")
        with open(res_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
            w.writerow({"patient": "mean", **{k: f"{arr[k].mean():.4f}" for k in arr}})
            w.writerow({"patient": "std", **{k: f"{arr[k].std():.4f}" for k in arr}})
        # per-band learned gate weights (interpretability), FATE only
        gated = [r for r in rows if "gate" in r]
        if gated:
            gpath = os.path.join(args.out, f"gate_weights_{tag}.csv")
            with open(gpath, "w", newline="") as fh:
                bands = [f"band{i+1}" for i in range(len(gated[0]["gate"]))]
                w = csv.writer(fh); w.writerow(["patient"] + bands)
                for r in gated:
                    w.writerow([r["patient"]] + [f"{v:.4f}" for v in r["gate"]])
            print(f"gate weights -> {gpath}")
        print(f"\nMean AUC={arr['auc'].mean():.4f}±{arr['auc'].std():.4f}  "
              f"({len(rows)} folds, {time.time()-t0:.0f}s)  -> {res_path}")
        if mlf is not None:
            for k in arr:
                mlf.log_metric(f"mean_{k}", float(arr[k].mean()))
                mlf.log_metric(f"std_{k}", float(arr[k].std()))
            mlf.log_artifact(res_path)

    if mlf is not None:
        mlf.end_run()


if __name__ == "__main__":
    main()
