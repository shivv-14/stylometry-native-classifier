# Model card: native vs non-native stylometric classifier

## Model details

- Logistic Regression and calibrated Linear SVM (scikit-learn) on handcrafted stylometric
  features, character 3–5-gram TF-IDF, or both combined
- Author: B. Shiva Ganesh (shivv-14). Code license: MIT
- Trained by `python tasks.py train`; parameters, training date and dataset hash are in
  `models/metadata.json`

## Intended use

- Teaching and research on stylometry and on how dataset design (writer leakage, topic,
  length) affects reported accuracy
- Exploring which surface features are associated with the label in a learner corpus

## Out-of-scope use

Any decision about a real person: academic, employment, immigration, disciplinary, legal
or forensic. Identifying authors. Texts outside short English argumentative essays. See
`docs/ethics_and_limitations.md`.

## Data

ICNALE Written Essays (recommended): learner essays plus a native-speaker (ENS) group on
the same two prompts. Balanced at writer level, texts under 120 words removed, texts
truncated to 200 words. Counts per class are generated in
`data/processed/data_summary.md` and stored in `models/metadata.json`.
The data is not distributed with the model.

## Metrics

All metrics are generated, never typed by hand:

- `reports/results/results.md` / `results.csv`: macro-F1 (with 95% bootstrap CI),
  accuracy, per-class precision/recall/F1, ROC-AUC, mean ± std over 5 writer-separated
  splits, compared with majority and length-only baselines
- E4: document-random vs writer-separated gap (writer leakage)
- E5: feature-group ablation; E6: cross-topic
- `reports/results/l1_recall.csv`: recall per L1
- `reports/figures/`: confusion matrices, ROC curves, top features, ablation, distributions

## Caveats

- Scores reflect one corpus and its collection conditions; they will not transfer to other
  genres, topics or populations.
- The label is confounded with proficiency, L1, education and task conditions.
- Explanations show associations in this dataset, not causes.
- Confidence values are model scores, not certainty; short texts are unreliable.
