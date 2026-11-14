# PHEME Text Preprocessing

## Purpose

Text preprocessing makes source posts more consistent for later TF-IDF features while preserving an auditable copy of the original text. It is applied after the leakage-safe event split and does not train a model or learn vocabulary.

Run:

```bash
make prepare-splits
```

The command reads `backend/data/processed/pheme_veracity.csv` and the split manifest, then writes `train.csv`, `validation.csv`, and `test.csv` under `backend/data/processed/splits/`. Each row contains `text_original`, `text_clean`, `label`, `event`, `thread_id`, `source_tweet_id`, and `split`.

## Stateless transformations

The current default options are recorded in `backend/data/reports/preprocessing_report.json`:

- Unicode is normalized with NFKC.
- HTML entities are unescaped.
- Text is case-folded to lowercase.
- URLs are replaced with the token `url`.
- User mentions are replaced with the token `user`.
- Hashtag markers are removed but the hashtag word is preserved, such as `#BreakingNews` → `breakingnews`.
- Leading retweet markers are removed.
- Repeated punctuation runs are reduced to one punctuation mark.
- Repeated whitespace is normalized.

Null values become empty strings. Negation words such as `not`, `no`, and `never` are not removed. Stopwords are not removed blindly because function words and negation can affect meaning. Stemming and lemmatization are not applied; either would be a separately documented experiment.

The raw `text` column is never overwritten. Labels and event/thread metadata are checked for exact preservation.

## Leakage prevention

There are three distinct stages:

1. **Stateless cleaning:** Unicode, casing, URL, mention, hashtag, punctuation, and whitespace operations. These can be applied to all rows because they do not inspect labels or learn from the corpus.
2. **Learned feature extraction:** a future TF-IDF vectorizer must be fitted only on `train.csv` `text_clean`, then used to transform validation and test text.
3. **Model training:** a future classifier must be fitted only on training features and labels. Validation is for model selection; test is held out for final evaluation.

This command performs only stage 1. It does not calculate label-specific vocabulary, fit a vectorizer, select preprocessing settings using validation/test data, or alter the event-level split.

## Observed PHEME text signals

Before cleaning, the real processed PHEME data contains URLs, mentions, hashtags, numbers, Unicode, emojis, punctuation, and retweet-style prefixes. The inspection and before/after statistics are stored in the preprocessing report. These signals are measured before removal or replacement so the academic analysis does not hide their prevalence.

## Model implications

TF-IDF benefits from consistent casing, URL/mention handling, and whitespace normalization. Multinomial Naive Bayes is sensitive to token counts and may benefit from stable replacement tokens. Logistic Regression and Linear SVM can use sparse word and n-gram features and remain explainable through feature weights. Removing hashtags or negation too aggressively could discard useful context, so the current pipeline keeps hashtag words and negation.

## Limitations

Replacing every URL or mention loses domain and account information. Hashtag segmentation is not performed, so `#CharlieHebdo` remains one token. Unicode normalization does not detect language, sarcasm, bots, or factual correctness. Repeated punctuation normalization may discard emphasis. These choices should be reported and, if changed, evaluated as controlled preprocessing experiments rather than silently modified.
