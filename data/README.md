# Beyond the Cohort Mean: Measuring, Predicting, and Partly Mitigating Patient-Level Failure in Cross-Patient EEG Seizure Detection


[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-CHB--MIT-lightblue.svg)](https://physionet.org/content/chbmit/1.0.0/)

---

## Overview

This repository contains the official implementation of:

> **" Beyond the Cohort Mean: Measuring, Predicting, and Partly Mitigating Patient-Level Failure in Cross-Patient EEG Seizure Detection"**

We address the fundamental challenge of cross-patient generalisation in EEG-based seizure detection — the ability to detect seizures in patients whose brain activity was never seen during training. Unlike most published work that evaluates on data from the same patients used for training, we adopt a strict **leave-one-subject-out (LOSO)** protocol across all 23 available CHB-MIT patients, providing a clinically realistic assessment of deployment on new unseen patients.

Our primary architectural contribution is the **Frequency-Aware Attention (FAA)** module, which explicitly decomposes EEG into clinically established frequency bands (Delta, Theta, Alpha, Beta) and applies per-band self-attention followed by cross-band sigmoid gating — enabling the model to dynamically weight which frequency bands are most discriminative for seizure detection across patients.

Beyond the architecture, this work makes a secondary empirical contribution: a systematic characterisation of **why cross-patient seizure detection fails**, revealing a bimodal patient population structure in CHB-MIT and demonstrating that patients with more seizures are paradoxically harder to classify — a finding contrary to the conventional data-scarcity hypothesis.
