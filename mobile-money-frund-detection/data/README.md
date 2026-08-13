# Data Directory

This directory is intended to hold the dataset inputs used by the fraud-detection training workflow.

The notebook code sequences identify a historical dataset CSV used for model development:

`PS_20174392719_1491204439457_log.csv`

The current repository snapshot does not include that raw file, so the training entry point falls back to a deterministic synthetic sample generated in-memory to keep the serialization contract testable. To complete a full training/evaluation reconstruction, restore the original CSV here and rerun the notebook or `src/train.py` workflow.
