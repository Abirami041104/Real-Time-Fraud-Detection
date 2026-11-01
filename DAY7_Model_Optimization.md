# Day 7: Random Forest Model Optimization

## Objective
Optimize the existing Random Forest fraud detection model by experimenting with different hyperparameter configurations and comparing their performance.

## Dataset and Experimental Setup
- Dataset: PaySim financial transaction dataset
- Original records: 6,362,620
- Sample size: 500,000
- Training records: 400,000
- Testing records: 100,000
- Sampling: Stratified sampling
- Train/test split: 80:20, stratified
- Algorithm: Random Forest Classifier
- Random state: 42

All experiments used the same sampled dataset and train/test split for a fair comparison.

## Hyperparameter Experiments

Four configurations were evaluated:

1. **Baseline:** 100 trees, maximum depth 15, minimum leaf samples 2.
2. **More Trees:** 150 trees, maximum depth 15, minimum leaf samples 2.
3. **Deeper Trees:** 100 trees, maximum depth 20, minimum leaf samples 2.
4. **Larger Minimum Leaf:** 100 trees, maximum depth 15, minimum leaf samples 5.

## Performance Comparison

| Experiment | Trees | Max Depth | Min Leaf | PR-AUC | ROC-AUC | Training Time (sec) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 100 | 15 | 2 | 1.000000 | 1.000000 | 7.81 |
| More Trees | 150 | 15 | 2 | 1.000000 | 1.000000 | 11.67 |
| Deeper Trees | 100 | 20 | 2 | 1.000000 | 1.000000 | 8.18 |
| Larger Min Leaf | 100 | 15 | 5 | 1.000000 | 1.000000 | 7.82 |

## Key Findings

- All four configurations achieved approximately 1.0 PR-AUC and ROC-AUC.
- Increasing the number of trees did not improve the measured scores.
- Increasing maximum tree depth did not provide a measurable improvement.
- Increasing minimum leaf samples also produced no measurable improvement.
- The baseline configuration had the lowest training time of 7.81 seconds.

## Selected Configuration

The baseline configuration was selected as the provisional choice because it achieved the same measured performance with the shortest training time.

- n_estimators = 100
- max_depth = 15
- min_samples_leaf = 2
- class_weight = balanced

The original saved model was not overwritten.

## Important Limitation

The experiment used a stratified random split of the synthetic PaySim dataset. Perfect scores do not guarantee real-world fraud detection performance. Further chronological validation and testing on independent real-world data would be needed before deployment.

## Output File
`reports/day7_model_optimization.csv`

## Conclusion
Day 7 successfully compared four Random Forest configurations. The baseline configuration was retained as the provisional choice based on training efficiency and equivalent measured evaluation scores.

**Status: Completed**