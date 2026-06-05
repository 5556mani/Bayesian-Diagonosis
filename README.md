# Bayesian Medical Dialogue Diagnosis System

This repository provides an automated, highly interpretable medical dialogue framework implemented in Python. It is based on the methodology proposed in the IEEE research paper: **"Bayesian-Based Symptom Screening for Medical Dialogue Diagnosis"** (ICTS4eHealth 2023).

---

## 📄 Research Paper Summary
Traditional automated medical diagnosis models rely heavily on Deep Reinforcement Learning (DRL), which operates as a "black box" with low medical interpretability and high resource costs. 

This paper introduces a lightweight alternative using **Bayesian Inference** paired with a **Symptom Screening Algorithm**. By analyzing symptom sets using intersection and union logic, the system determines the most distinguishing questions to ask a patient. This dynamically isolates high-probability conditions and eliminates low-probability ones using a binary search approach, minimizing conversation rounds while maintaining diagnostic accuracy.

---

## ⚙️ Implementation Process
The system implements the paper's dual-stage workflow through a modular three-step architecture:

1. **Feature Extraction (`Bayesian_calculations.py`):** Parses raw medical datasets to calculate static disease priors $P(d)$ and conditional symptom likelihoods $P(s|d)$. It replaces structural zeros with an epsilon factor ($\epsilon = 10^{-6}$) to guarantee mathematical stability during multiplication loops.
2. **Inference Engine (`mimic_doctor.py`):** Simulates the reasoning of an attending physician. It scales differential diagnosis probabilities up or down as symptoms are confirmed or denied, continuing the dialogue until top confidence hits a threshold ($\ge 90\%$).
3. **Patient Simulator Suite (`mimic_patient.py`):** Automatically samples records from the verification cohort, acts as a patient to answer doctor inquiries using ground-truth data, and generates benchmark logs reporting system accuracy and average dialogue length.

---

## 📁 Repository Structure
```text
├── 1_Installing_Data/          # Raw patient cohort data and symptom matrices
├── 2_Extracting_features/       # Python preprocessor scripts and generated probability matrix CSVs
├── 3_update_conversation/       # Core dialogue loops (Doctor CLI prompt & automated Patient simulator)
├── main.py                      # Global entrypoint script
└── pyproject.toml / uv.lock     # Environment management configuration and strict package locks
