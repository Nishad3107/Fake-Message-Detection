# India-First Language Policy

The future verification layer is designed first for Hindi, Hinglish/Romanized
Hindi, and English content about Indian events. This policy does not change the
current PHEME classifier, which remains its own English-oriented pattern model.

## Language handling

- Detect Unicode Devanagari, Latin-script Hindi indicators, English, and mixed
  rows. Keep the detector output and uncertainty with every claim.
- Preserve original text and apply Unicode normalization without silently
  changing meaning.
- Treat Hinglish and Romanized Hindi as a distinct language condition rather
  than assuming an English model is equivalent.
- Use transliteration as an explicit preprocessing view. Store the method,
  direction, and uncertainty; do not overwrite the original claim.
- Translation may be used only as a documented retrieval aid or comparison
  experiment. A translated passage is not an exact citation of the original.
- Evidence may be cited in its original language. If a translated excerpt is
  displayed, show the original excerpt and label the translation as generated
  or human-reviewed.

## Unsupported language behavior

If the claim or required evidence is outside the approved language policy and
no reliable human or translation workflow is available, return
`VERIFICATION_UNAVAILABLE` with a clear reason. Never silently translate and
present the result as exact evidence.

## India-first source context

Source selection must account for Hindi/English availability, Indian event
context, publication date, publisher identity, and evidence provenance. It
must not treat an Indian domain as automatically credible or a non-Indian
domain as automatically unreliable.
