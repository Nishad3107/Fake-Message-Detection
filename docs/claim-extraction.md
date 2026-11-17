# India-First Claim Extraction

Claim extraction is the boundary between a message-pattern classifier and a
future evidence verifier. The current PHEME model classifies learned language
patterns. It does not extract claims, retrieve sources, or establish truth.

## Rule-based baseline

The baseline in `backend/app/claim_extraction.py` is deterministic and uses
only the supplied text. It:

- normalizes Unicode and whitespace while preserving `claim_text`;
- splits text into sentences;
- detects likely factual sentences, questions, opinions, and speculation;
- identifies negation, years, dates, numbers, percentages, currencies, URLs,
  and a small India-first place-name list;
- detects Devanagari, English/Latin, Romanized Hindi, and mixed-script input;
- extracts conservative entity-like sequences and leaves unknown fields as
  `null` rather than guessing;
- creates up to five query candidates from words present in the claim.

The extraction result records sentence position, claim relation, method,
confidence, review requirement, and the exact source claim ID for every query.

## Language and modality behavior

Supported initial language conditions are Hindi Devanagari, Hinglish or
Romanized Hindi, and English content about India. Mixed-language input is
marked explicitly. Unsupported scripts preserve the original text but return
`verification_unavailable`; no translation is performed.

Plain text, social-media text, and article text are the only implemented input
types. URL strings, images, and videos return `not_implemented` in this
baseline. No URL is fetched and no OCR or transcript is inferred.

## Opinions, questions, and missing fields

Opinion markers such as “I think” or `मुझे लगता` are labelled `opinion`, not
treated as factual claims automatically, and require human review. Questions
are retained as claims with `is_question=true`, `unknown` type, and no automatic
verification query. Speculative wording also triggers review. Missing subject,
predicate, object, date, location, or entity information remains `null`.

## Multiple claims

Sentences receive stable IDs and positions. Claims are independent by default.
When a connective such as “because”, “therefore”, “however”, `लेकिन`, or
`इसलिए` is present, the relation is conservatively marked `unknown` instead of
inventing support or contradiction. A later verifier must resolve such
relations using evidence, not extraction heuristics alone.

## Query candidates

Candidate queries are limited to the original text, normalized text, detected
entities, and detected date/location context. They include query type,
language, source claim ID, and generation reason. No external knowledge,
translation, or invented entity is added.

## Limitations

Rules can miss morphology, sarcasm, coreference, code-switching, implied
claims, and multi-sentence context. Entity heuristics are not named-entity
recognition. Extraction confidence is a review signal, not a factual
probability. Errors here affect retrieval recall and any later verdict, so
ambiguous, mixed-language, opinion, multi-claim, unsupported-language, and
missing-context cases require human review.

The fixture report at
`backend/data/reports/verification/claim-extraction-fixtures.json` contains
fictional test examples only and is not a research evaluation.
