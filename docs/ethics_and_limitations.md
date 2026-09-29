# Ethics and limitations

This is a small research project about **patterns in one dataset**. It does not measure
anyone's identity, nationality, intelligence or English ability.

## Confounders

A native/non-native label in a corpus is tied to many other things. A classifier can pick
up any of them:

- **L1 (first language)**: non-native writers from different L1s write differently from
  each other; "non-native" is not one style.
- **Proficiency**: many differences are really proficiency differences. A highly proficient
  non-native writer may look "native" to the model and vice versa.
- **Education and age**: ICNALE learners are mostly university students; native groups may
  differ in age, degree level and writing training.
- **Genre and task**: short argumentative essays on set prompts. Results say nothing about
  emails, fiction, academic papers, or speech.
- **Topic**: ICNALE uses the same two prompts for both classes; LOCNESS does not, so
  topic words can become shortcuts there. E6 (cross-topic) checks part of this.
- **Editing/proofreading**: spell checkers, grammar tools, teachers or AI assistants
  change exactly the features used here.
- **Imbalance**: native writers are the minority in learner corpora. Writer-level
  downsampling fixes the class ratio but shrinks the data.
- **Task and collection conditions**: time limits, dictionary use, typed vs handwritten,
  instructions given.
- **Length**: longer or shorter texts change feature estimates. Texts are truncated to a
  fixed number of words and a length-only baseline is reported to detect this shortcut.

## Corpus used for the current results (W&I+LOCNESS)

ICNALE requires personal registration, so the published results use the openly
downloadable W&I+LOCNESS corpus. This is a weaker design than ICNALE:

- **Topic/genre/task confound**: learner essays are Write & Improve practice tasks
  (letters, reviews, stories, essays), while LOCNESS essays are argumentative university
  essays. The model can separate the classes by topic and genre words, not only by
  native-like style. Character n-grams are especially able to pick up topic words.
- **Tiny native class**: 50 native essays, so ≈100 writers after balancing and ≈20 test
  writers per split. Expect wide confidence intervals and large seed-to-seed variation.
- **Length**: native essays are much longer before truncation; after truncation to 200
  words some learner texts remain shorter. The length-only baseline (B1) shows how much
  this alone explains.
- **No L1 metadata** for W&I learners, so per-L1 recall cannot be reported; recall per
  CEFR level is the closest available fairness check.
- **No prompt IDs**, so the cross-topic experiment (E6) cannot run.

## Dataset-specific correlations, not universal differences

A coefficient saying "more semicolons → native" means only that in **this corpus, under
these conditions**, semicolons were more frequent in one group. It is not a rule of
English, and it can be reversed in another corpus. The app and figures therefore say
"features associated with the model's prediction on this dataset".

## Privacy and de-anonymisation

Stylometry can be used to link anonymous texts to authors. This project classifies a
coarse group label only, keeps no user input (the app does not store texts), and does not
publish corpus texts. Do not use it or its features to identify individuals.

## Prohibited uses

Do not use this model or its outputs for:

- academic integrity decisions, grading or admissions
- hiring, promotion or any employment decision
- immigration, visa, citizenship or asylum decisions
- disciplinary, legal or forensic purposes
- profiling or screening people by origin or language background

## Fairness

Errors are unlikely to be spread evenly: writers from some L1 backgrounds or proficiency
levels may be misclassified far more often. When L1 metadata exists, `run_experiments.py`
writes `reports/results/l1_recall.csv` (recall per L1 for the best model, pooled over
seeds) and `results.md` includes the table. Read it before drawing any conclusion from the
overall score.

## Technical limitations

- One corpus, one genre, two prompts; small number of native writers after balancing.
- spaCy's tagger and sentence splitter are trained on native, edited text and make more
  errors on learner text, which can itself become a signal.
- Probabilities from the calibrated SVM and LR are calibrated only on this data.
- Texts under ~120 words give noisy estimates; the app refuses texts under 40 words.
