# Feature dictionary

All handcrafted features are computed per text by `HandcraftedFeatures`
(`src/stylometry/features/handcrafted.py`), which runs spaCy `en_core_web_sm` once per
text (NER and lemmatizer disabled). Counts are normalised per 100 words or given as
proportions so that text length does not drive them directly.

**Word** = a spaCy token that is not punctuation and not whitespace.
**Alphabetic word** = a token made only of letters, lower-cased (used for vocabulary).

## Sentence (`sent_`), `features/sentence.py`

| feature | definition |
|---|---|
| `sent_mean_len` | mean sentence length in words (spaCy sentence segmentation) |
| `sent_median_len` | median sentence length |
| `sent_std_len` | standard deviation of sentence length |
| `sent_min_len` | shortest sentence |
| `sent_max_len` | longest sentence |
| `sent_per_100w` | number of sentences per 100 words |

Sentences with no words (e.g. a line of only punctuation) are ignored. A text with no
words gets 0 for every feature.

## Vocabulary (`vocab_`), `features/vocabulary.py`

| feature | definition |
|---|---|
| `vocab_ttr100` | type-token ratio of the first 100 alphabetic words (all words if shorter) |
| `vocab_mattr` | moving-average TTR, window 50 (Covington & McFall 2010); plain TTR if the text is shorter than the window. Length-robust version of TTR |
| `vocab_avg_word_len` | mean characters per alphabetic word |
| `vocab_long_word_prop` | proportion of alphabetic words with 7 or more characters |
| `vocab_hapax_ratio` | words occurring exactly once / total alphabetic words |
| `vocab_unique_per_100w` | distinct words per 100 alphabetic words |

Window sizes and the long-word threshold are in `configs/config.yaml`.

## Function words (`fw_`), `features/function_words.py`

`fw_<word>`: occurrences of the word per 100 words (case-insensitive, exact token match)
for: the, a, an, of, to, in, and, for, with, but, on, at, from, as, by, that, which, this,
it, is, be, not, or, so, if, there. `fw_total` is their sum. The list is
`features.function_words` in the config.

## Part of speech (`pos_`), `features/pos.py`

`pos_<TAG>`: percentage of words with spaCy universal POS tag TAG, for NOUN, PROPN, VERB,
ADJ, ADV, PRON, DET, ADP, CCONJ, SCONJ, AUX, PART, NUM. Tags come from a statistical tagger
and will contain errors, especially on learner text.

## Punctuation (`punct_`), `features/punctuation.py`

Characters per 100 words, counted on the cleaned text (curly quotes are normalised to
straight quotes first).

| feature | characters |
|---|---|
| `punct_comma` | `,` |
| `punct_period` | `.` |
| `punct_semicolon` | `;` |
| `punct_colon` | `:` |
| `punct_question` | `?` |
| `punct_exclamation` | `!` |
| `punct_parentheses` | `(` `)` |
| `punct_quotes` | `"` |
| `punct_dash` | `-` en dash, em dash |
| `punct_apostrophe` | `'` (includes single quotes, which cannot be told apart after normalisation) |

## Length (`length_`), baseline only

| feature | definition |
|---|---|
| `length_word_count` | number of words |
| `sent_mean_len` | as above |

Used only by the length-only baseline (B1). With truncation to 200 words, word count is
nearly constant for texts that were long enough.

## Character n-grams, `features/char_ngrams.py`

`TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=5000,
sublinear_tf=True, lowercase=False)`. `analyzer="char_wb"` (n-grams inside word
boundaries) is tried as a tuning option. Casing is kept because capitalisation habits are
part of the signal. In explanations these appear as `char:'<n-gram>'`.
