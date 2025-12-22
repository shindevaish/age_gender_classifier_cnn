# Changelog

## [2025-12-22]
### Dataset Experiments & Model Training

- Trained baseline model on the raw (simple) dataset.
- Performed EDA and data cleaning, then retrained the model.
- Normalized ages above 90 by converting them to age 91 and retrained the model.
- Removed all images with age greater than 90 and retrained the model.

**Observations:**
- Removing age >90 samples improved model stability.
- Age normalization ( >90 → 91 ) caused accuracy drop.