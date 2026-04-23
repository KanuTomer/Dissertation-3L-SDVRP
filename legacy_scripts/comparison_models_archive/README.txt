comparison_models/

common/
- Shared evaluator
- Shared packer
- Shared selection/crossover
- Shared metrics logger
- Shared plotting scripts
- Shared dataset/experiment utilities

baseline_a/
- No packing
- No penalty

baseline_b/
- Packing enabled
- Weak penalty

baseline_c/
- Packing enabled
- Strong penalty

proposed_model/
- Packing enabled
- Strong penalty
- Enhanced mutation

Datasets:
- Generated datasets are discovered from `VRP/generated_datasets/`
- Supported dataset files include `XML50_1111_01_merged_with_boxes_norm.json`
- Supported dataset files include `XML100_1111_01_merged_with_boxes_norm.json`
- Supported dataset files include `XML500_1111_01_merged_with_boxes_norm.json`

Outputs:
- Per-model results are saved under `comparison_models/outputs/<model>/<dataset_name>/`
- Combined summaries are saved to `comparison_models/model_comparison_summary.csv`
- Final plots are saved to `comparison_models/final_plots/`
