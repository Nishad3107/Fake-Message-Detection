# Temporal Verification Policy

Truth claims can depend on time. A future verifier must distinguish the time a
claim was made, the time an event occurred, the time evidence was published,
and the time the system retrieved it.

## Required timestamps

- `claim_time`: when the statement was published or uttered, if known;
- `event_time`: when the described event occurred or is expected to occur;
- `publication_date`: when a source published the evidence;
- `retrieved_at`: when this system obtained the source;
- `verification_time`: when the verdict was produced.

Unknown dates remain unknown; they are not inferred from filenames or page
order.

## Temporal rules

1. A claim is evaluated relative to an explicit verification time.
2. Evidence must be labelled as before or after the claim/event when dates are
   available.
3. Later evidence may explain what eventually happened, but must not be used
   to claim that an earlier prediction had information it could not have had.
4. Evaluation splits must prevent future documents, later fact-check wording,
   or post-event updates from leaking into a past-time task.
5. If the conclusion changes with the requested date, return
   `TEMPORALLY_UNCLEAR` rather than forcing a binary answer.
6. If dates conflict or cannot be validated, preserve the conflict and lower
   confidence.

## Evaluation

Temporal experiments must declare a cutoff date, allowed evidence window,
retrieval timestamp policy, and whether future evidence is permitted. A
time-aware test set should be locked before model selection. Reports must
separate contemporaneous verification from retrospective verification.
