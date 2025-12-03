# 🚚 3D Box Packing + GA Route Optimization (Dissertation Project)

This repository contains the full implementation of my dissertation work on optimizing vehicle loading and routing using:

- Genetic Algorithms (GA) for route ordering  
- 3D Box/Bin Packing (3D-BPP)  
- Iterative Isolation / VLR Repair methods  
- Automated replicate experiments  
- Timestamped experiment pipelines  
- Full reproducibility of results

The system reads real-world merged & normalized datasets, runs multiple GA replicates, applies spatial repair, summarizes results, and generates publication-ready figures.

---

# 📁 Project Structure

    Dataset Generation/
    │
    ├── input_dataset/
    │     └── XML100_1111_01_merged_with_boxes_norm.json
    │
    ├── experiments/
    │     └── run_replicates.ps1
    │
    ├── scripts/
    │     ├── summarize_replicates_fixed.py
    │     ├── extract_isolations.py
    │     └── make_figures_fixed.py
    │
    ├── dataset_generation/
    ├── results/
    │     └── runs/
    │
    └── run_all_experiments.ps1

---

# 🎯 Project Purpose

This project evaluates a hybrid optimisation pipeline combining:

- GA-driven route/order optimisation  
- Realistic 3D box/bulk packing  
- Automatic infeasibility detection & repair  
- Multi-replicate GA runs  
- Automated summary & figure generation  

---

# ⚙️ Setup Instructions

## 1️⃣ Clone the repository
    git clone <your-repo-url>
    cd "Dataset Generation"

## 2️⃣ Create and activate a virtual environment
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

## 3️⃣ Install dependencies
If using requirements:

    pip install -r requirements.txt

Otherwise:

    pip install numpy pandas matplotlib

## 4️⃣ Add your dataset file
Place your dataset here:

    input_dataset/XML100_1111_01_merged_with_boxes_norm.json

---

# 🚀 Running the Project

## ✅ Option A: Full Automated Pipeline (Recommended)

Runs the full workflow and saves results in timestamped folders:

    .\run_all_experiments.ps1

This script automatically:
- Loads dataset  
- Runs GA replicates  
- Saves everything under `results/runs/<timestamp>/`  
- Generates: summary CSV, isolation CSV, figures, logs  

---

## ✅ Option B: Run GA Replicates Manually

    .\experiments\run_replicates.ps1 `
        -normfile ".\input_dataset\XML100_1111_01_merged_with_boxes_norm.json" `
        -replicates 20 `
        -seed_start 2000

Outputs go to:

    experiments_output/

---

# 📊 Output Files Explained

### ✔ replicates_summary.csv  
Per-seed GA performance (best score, duration, unpacked, infeasible, etc.)

### ✔ figures/  
Automatically generated plots:  
- best_score vs seed  
- duration vs seed  
- unpacked distribution  
- infeasible histogram  

### ✔ isolation_summary.csv  
Shows spatial-repair / isolation behaviour.

### ✔ summary_stats.txt  
Contains timestamp, params, and output paths.

### ✔ experiments_output/  
Raw per-seed JSON outputs.

---

# 📌 Recommended .gitignore

    # Python
    __pycache__/
    *.pyc

    # Virtual environment
    .venv/

    # Output folders
    experiments_output/
    results/
    output/

    # Logs
    *.log

    # Large datasets
    input_dataset/*.json
    input_dataset/*.csv

    # Backup files
    *.bak

---

# 🧠 Key Concepts Implemented

### ✔ Genetic Algorithm (GA)
Optimises route/stop ordering; multiple seeds measure stability.

### ✔ 3D Packing + Repair
Uses rotation rules, feasibility checks, and VLR/Isolation repair.

### ✔ Replicate-Based Robustness
Shows variability, convergence stability, and runtime consistency.

### ✔ Automatic Analysis
Generates figures and summaries for dissertation use.

---

# 🧪 Reproducibility

To reproduce **all** experiments:

    .\run_all_experiments.ps1

Outputs will appear under:

    results/runs/<timestamp>/

---


