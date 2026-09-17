import subprocess
import sys
import os
from pathlib import Path

def run(cmd, desc=""):
    print(f"\n{'─'*55}")
    if desc:
        print(f"  {desc}")
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode != 0:
        print(f" Failed (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f" Done")


import platform
py = platform.python_version()
print(f"  Python : {py}")
assert tuple(int(x) for x in py.split(".")[:2]) >= (3, 10), \
    "Python 3.10+ required"

run("pip install --upgrade pip", "Upgrading pip")

PACKAGES = [
    # Deep learning
    "torch>=2.0.0",
    "torchvision>=0.15.0",

    # EEG / signal processing
    "mne>=1.5.0",
    "scipy>=1.11.0",

    # Data science
    "numpy>=1.26.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "imbalanced-learn>=0.11.0",

    # Visualisation
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",

    # Experiment tracking
    "wandb>=0.16.0",

    # Utilities
    "tqdm>=4.66.0",
    "pyyaml>=6.0.0",
    "joblib>=1.3.0",
    "thop>=0.1.1",            # FLOPs counting
    "einops>=0.7.0",          # tensor operations

    # Notebook support
    "jupyter>=1.0.0",
    "ipywidgets>=8.0.0",
]

pkg_str = " ".join(f'"{p}"' for p in PACKAGES)
run(f"pip install {pkg_str}", "Installing core packages")

print("\n" + "─"*55)
print("  Verifying GPU availability")
try:
    import torch
    if torch.cuda.is_available():
        n    = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f" {n} GPU(s) found")
        print(f"     Name   : {name}")
        print(f"     Memory : {mem:.1f} GB")
    else:
        print(" No GPU ")
except ImportError:
    print(" PyTorch import failed")
    sys.exit(1)

#Verify key imports
print("\n" + "─"*55)
print("  Verifying key imports")
checks = [
    ("torch",            "PyTorch"),
    ("mne",              "MNE"),
    ("sklearn",          "scikit-learn"),
    ("imblearn",         "imbalanced-learn"),
    ("scipy",            "SciPy"),
    ("matplotlib",       "Matplotlib"),
    ("seaborn",          "Seaborn"),
    ("numpy",            "NumPy"),
    ("pandas",           "Pandas"),
    ("thop",             "thop (FLOPs)"),
    ("einops",           "einops"),
]

all_ok = True
for module, label in checks:
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "unknown")
        print(f"   {label:<22} {ver}")
    except ImportError:
        print(f" {label:<22} NOT FOUND")
        all_ok = False

if not all_ok:
    print("\n Some imports failed. Re-run this script.")
    sys.exit(1)

