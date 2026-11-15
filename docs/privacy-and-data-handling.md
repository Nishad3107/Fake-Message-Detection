# Privacy and Data Handling

This academic prototype accepts user-provided social-media-style text for local
NLP analysis. It is not a hosted fact-checking service and does not require an
API key or send PHEME records to an external inference API.

## What is accepted and stored

- The API accepts one text field, between 10 and 5,000 non-whitespace
  characters, or a batch of at most 50 such posts.
- Successful predictions are stored in SQLite by default, including the
  original text, cleaned text, label, class scores, confidence category,
  explanation summary, model version, and timestamp.
- The application does not collect accounts, names, IP addresses, browser
  fingerprints, or other personal metadata.
- Application logs do not include submitted message text. Error responses are
  sanitized and do not expose stack traces or local filesystem paths.

## Where data goes

In local development, SQLite is stored at `backend/data/history.db`. In Docker,
the database is stored in the named `history-data` volume at `/app/runtime`.
The default local frontend calls the local FastAPI service. No external API is
used for model inference. If the application is deployed to a remote server,
submitted text necessarily travels to that server and the operator becomes
responsible for its privacy and retention policy.

Raw PHEME data, generated model artifacts, reports, and local history are
excluded from Git by repository policy. The PHEME files remain subject to their
original source terms.

## Retention and deletion

There is no automatic retention or deletion job in this prototype: history is
kept until the operator deliberately removes the local SQLite file or resets a
Docker volume. Stop the local service before making a backup or deletion, and
confirm the path first:

```bash
ls -l backend/data/history.db
```

For Docker, inspect the configured volume with `docker volume ls` and use the
documented Docker volume-management process for your environment. Do not run a
volume removal command unless you intentionally want to erase all stored
history.

## User notice and limitations

The analyzer displays a notice that submitted text is stored in local SQLite
history. Users should avoid entering private, identifying, or confidential
content. The system produces statistical language-pattern predictions, not
verified facts. Local history is not encrypted by the application, so operating
system permissions, disk encryption, backups, and deployment access controls
remain important.

