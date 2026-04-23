# 3L-SDVRP Comparison Framework with Packing-Aware Genetic Algorithms

## Overview
This repository contains a dissertation comparison framework for the Three-Dimensional Loading Split Delivery Vehicle Routing Problem (3L-SDVRP). The project studies how route construction changes when vehicle routing is evaluated together with 3D loading feasibility inside a fixed container.

The framework compares several genetic-algorithm-based models on synthetic benchmark datasets derived from XML CVRP instances. The implementation is inspired by PEAC-HNF-style packing-aware routing ideas, but is intentionally simplified into a feasibility-first comparison pipeline that is easier to reproduce and analyze in a dissertation or workshop-paper setting.

## Project Structure
```text
Dissertation-3L-SDVRP/
│
├── comparison_models/
├── VRP/
├── legacy_scripts/
├── README.md
├── requirements.txt
└── .gitignore
```

### `comparison_models/`
Contains the active comparison framework:
- baseline implementations
- proposed model implementation
- shared GA utilities
- experiment runners
- reporting scripts
- graph-generation scripts

### `VRP/`
Contains the active dataset-generation and validation pipeline:
- XML parsing
- synthetic dataset generation
- box generation
- dataset validation

### `legacy_scripts/`
Contains archived development code, historical experiment scripts, and older utilities that are not part of the current comparison pipeline. This folder is kept for traceability and reproducibility of the development process, but it is not required for the main workflow.

### Generated Outputs
Large generated outputs such as experiment results, graphs, ablation outputs, and conference-batch datasets are excluded from Git tracking through `.gitignore` to keep the repository lightweight.

## Models

### `baseline_a`
Distance-only genetic algorithm.
- ignores packing feasibility
- optimizes route distance only

### `baseline_b`
Weak packing-aware baseline.
- checks 3D packing feasibility
- applies a weak infeasibility penalty

### `baseline_c`
Strong feasibility-first baseline.
- checks packing feasibility
- applies a strong infeasibility penalty
- serves as the main packing-aware baseline for comparison

### `proposed_model`
Packing-aware GA with structural route improvements.
- adaptive decoding
- local route repair
- route-structure-aware scoring
- dataset-size-aware policies

## Key Features of the Proposed Model
- adaptive decoding with split-candidate evaluation
- tiny-route repair
- customer relocation repair
- route-balance mutation
- size-aware scoring and repair policies for small, medium, large, and very-large datasets

## How to Run

### 1. Generate datasets
```powershell
python VRP/generate_dataset_batch.py --all-sizes --groups 1111 1142
```

### 2. Run experiments
```powershell
python comparison_models/compare_all_models.py --dataset XML100 --model proposed_model
```

### 3. Full pipeline
```powershell
powershell comparison_models/run_full_pipeline.ps1
```

### 4. Ablation study
```powershell
python comparison_models/run_ablation_study.py
```

## Results Summary
The final comparison framework shows that the proposed model improves route structure quality more consistently than it improves pure routing distance.

In the current tuned configuration, the proposed model typically improves:
- minimum route fill
- route fill balance
- route composition quality
- packing-aware structural stability

Distance performance is:
- competitive on smaller datasets
- competitive again on larger datasets after size-aware tuning
- somewhat higher on part of the medium-size band

The strongest representative sizes are:
- XML50
- XML100
- XML250
- XML500
- XML750

The weaker band is:
- XML150 to XML450

This makes the proposed model strongest as a packing-aware structural improvement approach rather than a pure distance-dominance method.

## Key Contributions
- unified comparison framework for routing-only and packing-aware GA models
- proposed packing-aware GA enhancements with adaptive decoding and repair
- size-aware optimization policy for different dataset scales
- extensive automated experiment, reporting, and graph-generation pipeline

## Notes
- generated outputs and graphs are excluded from Git tracking to keep the repository lightweight
- `legacy_scripts/` contains archived development code and older experimental utilities
- the active comparison pipeline is fully reproducible using the commands above
