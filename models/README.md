# Model bundle

The notebook (section 17) writes `ids_xgb_bundle.joblib` (~5 MB). It is a dictionary containing:

- `pipeline`: cleaning + ColumnTransformer + tuned XGBoost, fitted end to end
- the decision threshold (0.630) and its provenance (policy, subset, alternatives at 1% / 10% FPR)
- hyperparameters, preprocessing parameters, categorical/numeric column lists
- sealed-test metrics with confidence intervals, the ablation table, a version tag and a timestamp

Run the notebook to regenerate it, or download it from the GitHub Releases page, then place it in this folder.
