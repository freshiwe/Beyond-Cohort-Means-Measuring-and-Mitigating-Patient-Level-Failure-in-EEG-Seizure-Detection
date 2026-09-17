# ==================== INSTALL.md ====================

# Installation Guide

## Requirements

| Component | Version |
|-----------|---------|
| Python    | ≥ 3.10  |
| PyTorch   | ≥ 2.0.0 |
| CUDA      | 11.8 (recommended) |
| RAM       | ≥ 16 GB |
| GPU VRAM  | ≥ 8 GB  |

---

## Option 1 — pip (recommended for NRP.ai / Colab)

```bash
git clone https://github.com/YOUR_USERNAME/EEG-Seizure-Detection.git
cd EEG-Seizure-Detection
pip install -r requirements.txt
pip install -e .
```

Verify the installation:

```bash
python environment_setup.py
```

---

## Option 2 — Conda

```bash
git clone https://github.com/YOUR_USERNAME/EEG-Seizure-Detection.git
cd EEG-Seizure-Detection
conda env create -f environment.yml
conda activate eeg-seizure
pip install -e .
```

---

## Option 3 — Docker

```bash
git clone https://github.com/YOUR_USERNAME/EEG-Seizure-Detection.git
cd EEG-Seizure-Detection
docker build -t eeg-seizure .
docker run --gpus all -p 8888:8888 -v $(pwd)/data:/workspace/data eeg-seizure
```

Open `http://localhost:8888` in your browser.

---

## Option 4 — NRP.ai (Kubernetes Job)

The full 23-patient LOSO experiment was run on NRP.ai using
distributed Kubernetes jobs. See `scripts/nrp/` for job manifests.

```bash
# Submit a single LOSO fold
kubectl apply -f scripts/nrp/loso_job.yaml

# Submit all 23 folds in parallel
kubectl apply -f scripts/nrp/loso_array_job.yaml
```

---

## Data Setup

See `data/README.md` for CHB-MIT download instructions.

```bash
# After creating a PhysioNet account and accepting the DUA:
python scripts/download_data.py \
    --patients chb01 chb02 chb03 ... chb24 \
    --data_dir data/chbmit

# Preprocess into .npz windows
python scripts/preprocess.py --config configs/default.yaml
```

---



