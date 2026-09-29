# models/

Trained pipelines are written here by `python tasks.py train` (or `make train`):

- `<feature_set>_<model>.joblib` — fitted scikit-learn pipelines (lr, svm) for
  `handcrafted`, `char`, and `combined` feature sets
- `metadata.json` — feature set, hyper-parameters, training date, dataset hash,
  held-out test metrics, and which feature set was selected as best

Model files are gitignored. Re-create them from the data with the commands above.
