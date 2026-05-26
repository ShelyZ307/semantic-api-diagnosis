# Stage 4 Final Dataset

Stage 4 generated the frozen Version 1 dataset splits.

Final local split sizes:

- `final_train.jsonl`: 3000 examples
- `final_validation.jsonl`: 500 examples
- `final_seen_test.jsonl`: 700 examples
- `final_unseen_family_test.jsonl`: 700 examples

The final split generator creates all splits jointly and rejects cross-split near-duplicates. This prevents validation and seen-test examples from being overly similar to training examples.

Quality gates passed:

- no exact duplicate serialized inputs across splits
- no exact duplicate endpoint contract plus request pairs across splits
- no family leakage between seen and unseen files
- balanced family distributions
- all labels represented in each split
- semantic labels represented in each split
