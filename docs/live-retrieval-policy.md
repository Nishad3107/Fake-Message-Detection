# Live Retrieval Policy

Live retrieval is disabled by default. The current FastAPI application does not
search the web, fetch user URLs, or require search credentials.

## Retrieval modes

1. **Offline benchmark:** fixed local documents and a locked retrieval index.
2. **Local corpus:** explicitly imported documents with recorded provenance.
3. **User-provided URL:** fetch only after explicit configuration and input
   validation.
4. **Real-time web search:** requires a separate approval and provider policy.
5. **Search API:** requires an approved provider, secret management, quotas, and
   terms review.
6. **Browser retrieval:** requires an isolated service and must respect robots,
   terms, rate limits, and access restrictions.
7. **Source-page fetching:** validates redirects, content type, size, timeout,
   and publication metadata.
8. **Evidence caching:** follows the storage policy and records URL, hash, and
   retrieval timestamp.

## Required safeguards

- Enable retrieval only through explicit configuration; never by default.
- Never commit or print search/API credentials.
- Return `VERIFICATION_UNAVAILABLE` for timeout, DNS, access, parsing, or
  policy failures.
- Validate redirects and reject private-network or unsupported destinations.
- Reject unsupported content types and excessive response sizes.
- Respect robots directives, publisher terms, copyright restrictions, and
  applicable law.
- Show the source URL and retrieval timestamp to users.
- Unknown domains may be candidates only when explicitly enabled and may not be
  cited automatically.
- A failed or unavailable fetch must not be represented as evidence against a
  claim.

The initial YAML policy sets live retrieval, external search APIs, and full-page
storage to `false`. No credentials are required for offline mode.
