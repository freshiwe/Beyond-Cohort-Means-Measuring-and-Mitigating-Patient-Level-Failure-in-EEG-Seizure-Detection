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



---

## Key Findings

### 1. Bimodal patient population in CHB-MIT

CHB-MIT contains two statistically distinct patient subgroups —
an easy cluster (mean AUC ≈ 0.95) and a hard tail (mean AUC ≈ 0.47),
confirmed by Gaussian Mixture Model analysis (2-component BIC
significantly lower than 1-component).

### 2. More seizures → harder to classify

Contrary to the conventional data-scarcity hypothesis,
patients with *more* seizures are systematically harder to classify
(Spearman ρ = −0.XX, p < 0.05). This suggests that
frequent-seizure patients in CHB-MIT are medication-refractory
cases with heterogeneous ictal morphology that defeats
population-level frequency priors.

### 3. BatchNorm adaptation is insufficient

Test-time BatchNorm adaptation using unlabelled interictal EEG
produces negligible improvement (mean ΔAUC = +0.004), indicating
that the inter-patient domain shift in seizure detection operates
at the **weight level** rather than the statistics level —
motivating weight-space adaptation (few-shot fine-tuning)
as the next step.


## Dataset

**CHB-MIT Scalp EEG Database**  
Children's Hospital Boston — MIT  
[https://physionet.org/content/chbmit/1.0.0/](https://physionet.org/content/chbmit/1.0.0/)

- 23 paediatric patients with intractable epilepsy
- 916+ hours of continuous scalp EEG
- 256 Hz sampling rate, 18 common channels
- Annotated seizure onset and offset times

---

## Citation

If you use this code or the findings of this paper in your
research, please cite:

```bibtex
@article{yourname2025faa,
  title   = {Beyond the Cohort Mean: Measuring, Predicting, and Partly Mitigating Patient-Level Failure in Cross-Patient EEG Seizure Detection},
  author  = {Joshua Omorodion, Muadh Olamilekan, and Aalaa Rehman,},
  journal = {IEEE BigData 2026 - 9th Special Session on HealthCare Data.},
  year    = {2026},
  note    = {Under review}
}
```

---

## Contributing

Contributions are welcome. Please open an issue before
submitting a pull request. For major changes, discuss
the proposed change in an issue first.

```bash
# Development setup
git clone https://github.com/freshiwe/Beyond-Cohort-Means-Measuring-and-Mitigating-Patient-Level-Failure-in-EEG-Seizure-Detection.git
cd EEG-Seizure-Detection
pip install -r requirements.txt
pip install -e .

# Run tests before submitting a PR
python scripts/test_synthetic.py
flake8 src/ scripts/ --max-line-length=100
```

---

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

The CHB-MIT dataset is separately licensed under the
[PhysioNet Restricted Health Data License 1.5.0](https://physionet.org/content/chbmit/view-license/1.0.0/).
You must accept that license independently before
downloading the data.

---

## Acknowledgements

- The CHB-MIT dataset was collected at Children's Hospital
  Boston under the supervision of Dr. Ali Shoeb and
  Dr. John Guttag and made available through PhysioNet.
- Compute resources provided by the
  [National Research Platform (NRP.ai)](https://nationalresearchplatform.org/)
  via Kubernetes GPU cluster allocation.
- EEGNet architecture components adapted from
  [Lawhern et al. 2018](https://doi.org/10.1088/1741-2552/aace8c).

---

## Contact

**Abdullateef Muadh Olamilekan**  
KIIT University  
2129135@kiit.ac.in 

*For questions about the code, open a GitHub issue.  
For questions about the paper, contact the corresponding author.*