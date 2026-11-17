# Evidence Storage Policy

This is the default policy for a future verifier. Evidence storage is disabled
in the current application.

## Default record

When storage is explicitly approved, retain only what is needed to reproduce a
citation decision:

- source URL;
- source title and publisher;
- short permitted excerpt;
- source category and metadata;
- content hash;
- publication date when available;
- retrieval timestamp;
- retrieval and entailment model versions;
- claim identifier and relation.

Do not store full webpages by default. Do not store user-submitted private text
or unrelated page content. Full-page storage requires a separate legal,
copyright, security, and retention approval.

## Privacy, copyright, and retention

URLs can reveal sensitive interests, and excerpts may contain personal or
copyrighted material. Store the minimum excerpt allowed by the source terms,
restrict access, avoid logging raw claim text, and keep a deletion mechanism.
The retention period is currently **not approved**; it must be configured by the
project owner before evidence storage is enabled. Deletion must remove stored
URLs, excerpts, metadata, hashes, and derived citation records where required.

The system must not send PHEME, HinFakeNews, or user content to an external
provider unless the provider, data processing terms, and user-facing notice
are approved.
