# Stage 5 Baselines

Stage 5 implemented baseline evaluation before transformer fine-tuning.

Implemented baselines:

- majority empty-label baseline
- majority frequent-label baseline
- family-frequency baseline
- visible rule-based baseline

Metrics include:

- all-label micro and macro F1
- per-label precision, recall, and F1
- semantic-label micro and macro F1
- exact-match label-set accuracy
- validity accuracy
- severity accuracy
- critical semantic-error miss rate

The visible rule-based baseline performed strongly on full and request-only inputs, showing that many examples are request-obvious. Contract-dependent hard subsets were created to evaluate cases where the natural-language endpoint contract matters more.

Future model results should be reported on:

- full seen test
- full unseen-family test
- contract-dependent seen hard subset
- contract-dependent unseen hard subset
