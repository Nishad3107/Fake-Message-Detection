# Verification Benchmark Preparation

FEVER and AVeriTeC are separate evidence-verification tracks. They are not
combined with PHEME rumor veracity or HinFakeNews article classification.

## Manual acquisition

Download the exact release from the official project source, review its license
and terms, and place only the local files in the matching directory:

- FEVER: `backend/data/raw/verification/fever/`
  - Official source: <https://fever.ai/dataset/fever.html>
- AVeriTeC: `backend/data/raw/verification/averitec/`
  - Official source: <https://fever.ai/dataset/averitec.html>

Do not paste URLs into the adapter as input data and do not ask the adapter to
download anything. Record the release/version, license, filenames, and hashes
after the files are supplied.

## Local commands

```bash
make prepare-verification-benchmark BENCHMARK=fever
make prepare-verification-benchmark BENCHMARK=averitec
```

Each command writes a separate preparation report under
`backend/data/reports/verification/` and, only when local files validate,
writes normalized JSONL under the corresponding processed directory.

For the locally supplied AVeriTeC train/dev release, the adapter preserves the
nested questions, answers, URLs, metadata, labels, justifications, and original
split. One source row has an empty claim and is therefore explicitly excluded
from normalized claims rather than being fabricated; this leaves the
preparation status `partial`. Run the follow-up offline checks with:

```bash
make build-verification-corpus BENCHMARK=averitec MODE=train_evidence_corpus
make audit-verification-corpus BENCHMARK=averitec
make verify-verification-readiness
```

The train evidence corpus is a development artifact only. It must not be
reported as independent retrieval performance. The current local audit marks
it unsafe for development/research metrics when train/dev claims, URLs, or
evidence content overlap. The unavailable AVeriTeC knowledge store and blind
test remain required for defensible retrieval benchmarking.

With no files present, the report remains `incomplete` and gives the required
manual action. No counts or labels are fabricated.

## Dataset separation

- FEVER records remain under `verification/fever/`.
- AVeriTeC records remain under `verification/averitec/`.
- PHEME and HinFakeNews paths are never read by these adapters.
- No model training or retrieval index construction is performed.
- Raw and generated benchmark data remains ignored unless redistribution is
  explicitly allowed by the dataset license.
