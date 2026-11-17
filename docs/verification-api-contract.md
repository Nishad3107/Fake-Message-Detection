# Future Verification API Contract

This contract is planned and is not implemented in the current FastAPI
application. Existing PHEME endpoints remain unchanged.

## Response shape

```json
{
  "claim": {
    "text": "...",
    "language": "hi"
  },
  "pattern_analysis": {
    "label": "unverified",
    "confidence": 0.61,
    "warning": "Pattern classification is not factual verification."
  },
  "evidence_verification": {
    "status": "INSUFFICIENT_EVIDENCE",
    "confidence": 0.0,
    "evidence": [],
    "reason": "No sufficient independent evidence was found."
  },
  "temporal_context": {
    "claim_time": null,
    "event_time": null,
    "verification_time": "2026-11-17T00:00:00Z",
    "evidence_time_policy": "Evidence available by verification time."
  },
  "limitations": [
    "The system cannot guarantee factual truth.",
    "Retrieved sources may be incomplete or conflicting."
  ]
}
```

## Status semantics

- `SUPPORTED`: direct, relevant evidence supports the claim.
- `REFUTED`: direct, relevant evidence contradicts the claim.
- `INSUFFICIENT_EVIDENCE`: no sufficiently relevant or strong evidence was
  found. It must not be converted to `REFUTED`.
- `CONFLICTING_EVIDENCE`: credible retrieved sources support and contradict
  the claim.
- `TEMPORALLY_UNCLEAR`: the result depends on date or changing circumstances.
- `VERIFICATION_UNAVAILABLE`: retrieval, parsing, source access, or safety
  checks prevented a defensible verification.

## Input support

The future request should support text, URL, image, OCR text, and multiple
claims. It must preserve the input type, language, OCR confidence, source URL,
claim identifiers, retrieval errors, and missing evidence. A URL fetcher or
image upload service must be separately authorized and isolated; no external
credentials are added by this planning change.

## Citation rules

Each evidence citation requires title, URL, publisher when available, source
category, publication date when available, retrieval timestamp, excerpt,
relevance score, entailment score, and support/contradiction relation. The API
must reject or omit citations that were not actually retrieved. A language
model may summarize recorded evidence but may not invent URLs, excerpts, or
source claims.

## Compatibility decision

Current endpoints that can remain unchanged: `/health`, `/api/v1/model-info`,
`/api/v1/predict`, `/api/v1/predict/batch`, `/api/v1/history`, and
`/api/v1/metrics`. A future verifier would likely use a separate
`/api/v2/verify` endpoint and possibly `/api/v2/claims`, `/api/v2/evidence`,
and upload/retrieval job endpoints. The existing history schema is not enough
for citations, temporal provenance, or multiple claims; a versioned evidence
table or separate verification store will be required. The frontend will need
an evidence panel, citation links, conflicting-evidence state, retrieval
errors, and OCR/image/video uncertainty. The frozen pattern classifier can
remain a fallback pattern signal, never a verification substitute.
