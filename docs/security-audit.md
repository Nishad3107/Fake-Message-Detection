# Security Audit

Audit scope: FastAPI input handling, SQLite persistence, frontend rendering,
configuration, dependencies, Docker/CI files, and tracked repository content.
The frozen PHEME model and research metrics were not changed.

## Findings and fixes

| Area | Severity | Finding | Status |
| --- | --- | --- | --- |
| Input validation | High | Text, whitespace-only input, maximum length, strict string types, and batch size were not uniformly enforced. | Fixed with Pydantic validators and configurable limits. |
| Error disclosure | High | Model-service exception details could expose internal paths through API responses. | Fixed with stable generic 503 responses. |
| Request size | Medium | Very large request bodies were not bounded before JSON parsing. | Fixed with a configurable 1 MB content-length guard. |
| CORS | Medium | Methods and headers were broader than needed. | Restricted to documented origins and GET/POST/OPTIONS plus Content-Type. Production rejects wildcard origins. |
| Response hardening | Low | Browser security headers were absent. | Added nosniff, frame denial, and no-referrer headers. |
| Privacy logging | Medium | Logging policy was implicit. | Logs exclude submitted text; production error logs record only exception type unless debug is explicitly enabled. |
| SQLite permissions | Medium | Database file permissions were not normalized. | Existing database files are best-effort restricted to owner read/write on supported systems. |
| SQL safety | Low | History queries use parameterized values; schema names are fixed constants. | Verified. |
| Frontend rendering | Medium | User text must not be interpreted as HTML. | React text rendering is used; no `dangerouslySetInnerHTML` or external inference call is used. |
| Secrets and paths | High | Tracked secrets, user-specific paths, raw PHEME data, model artifacts, and SQLite history must not enter Git. | CI repository checks and `.gitignore` rules cover these classes. |

## Remaining risks

- SQLite history is local and not encrypted by this prototype.
- The default development CORS origins are intentionally local; deployment
  operators must configure the exact frontend origin and use HTTPS.
- The model probabilities are not calibrated factual probabilities and the
  model can be overconfident on unseen events.
- Dependency ranges are resolved through the committed frontend lockfile and
  Python requirements constraints. A future production deployment should use
  a reviewed, regularly updated lock/constraints policy and vulnerability
  scanning.
- Rate limiting, authentication, centralized audit logging, and managed
  database controls are outside this academic prototype.

## Configuration expectations

Safe defaults are in `.env.example`. For production, set `ENVIRONMENT=production`
and an explicit `CORS_ORIGINS` value; wildcard origins are rejected. Review
`MAX_REQUEST_BYTES`, `MAX_POST_LENGTH`, `MAX_BATCH_SIZE`,
`HISTORY_RESULT_LIMIT`, `LOG_LEVEL`, and `DEBUG` for the deployment. Keep
`DEBUG=false` and do not put credentials in `.env` under version control.

## Verification

Run the local quality gates before merging:

```bash
make test
make lint
make format-check
make build
make repo-checks
git diff --check
```

The full PHEME dataset and ignored model artifacts are intentionally not
required by CI source/API-contract tests. Docker prediction verification does
require the locally generated frozen artifact.

