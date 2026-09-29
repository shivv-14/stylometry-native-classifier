# Data

The corpora used here are licensed and require registration, so they are **not** in this
repository and are never downloaded automatically. `data/raw/`, `data/interim/` and
`data/processed/` are gitignored.

If `data/raw/` is empty, `python tasks.py prepare` prints these instructions and exits
without error. No fake data is ever generated to produce results. The only synthetic
texts in the repo are in `tests/fixtures/SYNTHETIC_samples.csv`, used by unit tests only.

## Option A (recommended): ICNALE Written Essays

ICNALE contains essays by learners from Asian countries **and** a native-speaker group
(ENS) written on the same two prompts (PTJ: part-time jobs for college students;
SMK: smoking in restaurants). Using ENS as the native class keeps topic, genre and task
the same for both classes, which is the cleanest design available.

1. Register and download at <https://language.sakura.ne.jp/icnale/> (Written Essays,
   plain-text files).
2. Copy all essay `.txt` files, including the ENS files, into `data/raw/icnale/`
   (sub-folders are fine):

   ```
   data/raw/icnale/W_CHN_PTJ0_004_B1_1.txt
   data/raw/icnale/W_ENS_SMK0_012_XX_1.txt
   ...
   ```

3. Check that `data.icnale_filename_regex` in `configs/config.yaml` matches your file
   names. It must contain the named groups `country`, `topic`, `writer`, `proficiency`.
   The loader stops with an error listing example file names if any file does not match,
   so adjust the regex to your release rather than renaming files.
4. `python tasks.py prepare`

Label rule: `native` if the country code is `ENS` (configurable as
`data.native_country_code`), otherwise `non_native`. Writer ID = country + number, since
numbers repeat across countries.

## Option B: LOCNESS as the native class

1. Request LOCNESS from UCLouvain (CECL).
2. Put LOCNESS `.txt` files in `data/raw/locness/` and ICNALE files in `data/raw/icnale/`.
3. Set `data.source: icnale+locness`.

LOCNESS replaces the ICNALE ENS group. **Warning:** LOCNESS topics and task conditions
(untimed essays by British/American students on many different topics) differ from ICNALE,
so topic and genre confounding is much higher and results are harder to interpret.
LOCNESS has no reliable writer IDs; each essay is treated as its own writer.

## Option B2 (used for the current results): W&I+LOCNESS, open download

The BEA-2019 shared-task release of **Cambridge English Write & Improve (W&I)** learner
essays plus **LOCNESS** native essays can be downloaded without registration:

```bash
python tasks.py download      # -> data/raw/wi_locness/ (json/ + licences)
```

and `data.source: wi_locness` (the current default). By downloading it you accept its
licences (`licence.wi.txt`, `license.locness.txt`): **non-commercial research and
education only, no redistribution**, cite Yannakoudakis et al. (2018) for W&I and credit
CECL (UCLouvain) for LOCNESS. Excerpts under 100 words only.

What the loader does:

- `A/B/C.*.json`: learner essays; `writer_id = WI_<userid>`, proficiency = CEFR level
  (A1–C2). Essays with no `userid` are dropped (their author is unknown, so writer
  separation could not be guaranteed); the count is logged.
- `N.dev.json`: 50 LOCNESS essays; no writer IDs, so each essay is its own writer;
  proficiency `native`, L1 `ENG`.
- Prompts are not given, so `topic` is empty and the cross-topic experiment (E6) is skipped.

Caveats specific to this corpus (see also `docs/ethics_and_limitations.md`):
only **50 native essays** are available, so the balanced dataset is small (≈100 writers)
and scores have wide confidence intervals; W&I and LOCNESS were written on **different
prompts under different conditions** (online practice tasks vs university essays), so
topic and genre are confounded with the label; native essays are much longer before
truncation.

## Option C: any corpus as CSV

Put a CSV at `data/raw/dataset.csv` and set `data.source: csv`.

| column | required | meaning |
|---|---|---|
| `text` | yes | the passage |
| `label` | yes | `native` or `non_native` |
| `writer_id` | yes | unique per person (used for writer-separated splits) |
| `topic` | no | prompt/topic (enables the cross-topic experiment) |
| `l1` | no | first language / country (enables per-L1 recall) |
| `proficiency` | no | proficiency level (used for stratified balancing) |

## What `prepare` produces

- `data/processed/dataset.parquet`: cleaned, filtered, balanced, truncated texts plus
  metadata and `word_count_original` / `word_count`
- `data/processed/data_summary.md`: texts, writers, mean length and topics per class
