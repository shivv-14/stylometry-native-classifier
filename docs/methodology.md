# Methodology

## 1. Task

Binary classification of an anonymous English passage as written by a **native** or a
**non-native** writer, using stylometric features and character n-grams. Research questions:

- **RQ1** Can stylometric features distinguish native and non-native English writing?
- **RQ2** Do character n-grams perform better than handcrafted linguistic features?
- **RQ3** Does performance hold when the test set contains only unseen writers?

Out of scope: transformers, multilingual input, L1 identification, open-set prediction,
forensic or legal use.

## 2. Data design

The recommended corpus is ICNALE Written Essays: learners from several Asian countries and
a native-speaker group (ENS) answer the **same two prompts** under the same conditions.
Keeping prompt, genre and task identical across classes removes the most obvious shortcut
(topic). LOCNESS can replace the native group but brings different topics and conditions
(see `data/README.md`). Any corpus can be plugged in through the generic CSV loader.

Preparation (`scripts/prepare_data.py`):

1. **Cleaning**: Unicode NFKC, curly to straight quotes, whitespace normalisation,
   metadata header lines removed. Punctuation and casing are kept because they are
   features.
2. **Minimum length**: texts under `data.min_words` (default 120) words are dropped; the
   number dropped per class is logged and written to `data_summary.md`.
3. **Writer-level balancing**: the native class is usually much smaller. Non-native
   **writers** are downsampled to the number of native writers, stratified by L1 and
   proficiency when available; all texts of a selected writer are kept. Alternatively
   `data.balance: class_weight` keeps all data and uses `class_weight="balanced"`.
4. **Length control**: every text is truncated to its first `data.truncate_words`
   (default 200) words so raw length cannot separate the classes. The original and final
   word counts are stored.

## 3. Splitting

- **Main protocol, writer-separated**: `StratifiedGroupKFold` with `writer_id` as group;
  one fold (≈20% of writers per class) is the test set. The code asserts that no writer is
  in both sets (`assert_no_writer_overlap`), and a unit test checks it for five seeds.
- **Seeds**: the whole experiment is repeated for 5 seeds (5 different writer splits);
  results are reported as mean ± std.
- **Tuning**: `GridSearchCV` with `StratifiedGroupKFold(n_splits=5)` inside the training
  writers only, scoring macro-F1.
- **Contrast (E4)**: a document-level random split, where the same writer's essays can be in
  train and test. The gap to the writer-separated score estimates how much writer leakage
  would inflate results.
- **Cross-topic (E6)**: writers are split in half; the model trains on one prompt from half A
  and is tested on the other prompt from half B (unseen writers, unseen topic).
- **No leakage through preprocessing**: TF-IDF vocabulary/IDF and the `StandardScaler` are
  steps of the sklearn `Pipeline`, so they are fit on training folds only. Handcrafted
  features are deterministic per text (nothing is learned), so caching them is safe.

## 4. Features

Five handcrafted groups (sentence, vocabulary, function words, POS, punctuation; see
`feature_dictionary.md`) and character 3–5-gram TF-IDF. Feature sets:

| set | content |
|---|---|
| `length` | word count + mean sentence length (baseline) |
| `handcrafted` | 5 groups, standardised |
| `char` | char n-gram TF-IDF (`char` or `char_wb`, tuned) |
| `combined` | `FeatureUnion` of standardised handcrafted + char TF-IDF, with an optional weight on the handcrafted block |

## 5. Models

- `LogisticRegression(max_iter=2000, solver="liblinear")`, C ∈ {0.01, 0.1, 1, 10}
- `LinearSVC`, C ∈ {0.01, 0.1, 1, 10}, wrapped in `CalibratedClassifierCV(method="sigmoid", cv=3)`
  so the app can show probabilities
- Baselines: majority class (`DummyClassifier`), length-only logistic regression
- Optional: `RandomForestClassifier` on handcrafted features (`build_pipeline("handcrafted", "rf", cfg)`)

`python tasks.py train` tunes and fits LR and SVM for the handcrafted, char and combined
sets on the seed-`seed` writer split and saves them with `models/metadata.json`
(parameters, training date, dataset SHA-256, held-out metrics). The "best" feature set is
chosen by **cross-validation score on training writers**, never by the test set.

## 6. Experiments (`scripts/run_experiments.py`)

| ID | features | models | split |
|---|---|---|---|
| B0 | none | majority | writer-separated |
| B1 | length only | LR | writer-separated |
| E1 | handcrafted | LR, SVM | writer-separated |
| E2 | char n-grams | LR, SVM | writer-separated |
| E3 | combined | LR, SVM | writer-separated |
| E4 | best of E1–E3 (by CV) | LR, SVM | document-random vs writer-separated |
| E5 | handcrafted minus one group | LR | writer-separated |
| E6 | best (by CV) | best | cross-topic |

## 7. Evaluation protocol

For every run: accuracy; precision, recall and F1 per class and macro; ROC-AUC (positive
class `native`); confusion matrix; 95% percentile bootstrap CI (1000 resamples) of macro-F1
on the test set. Aggregation: mean ± std over the 5 seeds. If the best full model beats the
length-only baseline by less than `evaluation.length_baseline_flag_gap` macro-F1, the
report prints a warning.

Explainability (`src/stylometry/explain/explain.py`): global top-k coefficients per class;
group importance as (a) the E5 ablation drop and (b) the sum of |coef| per group on
standardised handcrafted features; local contributions (coefficient × feature value) for
the app. For the calibrated SVM the coefficients of the calibration folds are averaged.
All of these describe **associations in this dataset**, not causes.

All numbers in the README and reports come from files written by the scripts:
`reports/results/results.csv`, `results.md`, and `reports/figures/`.
