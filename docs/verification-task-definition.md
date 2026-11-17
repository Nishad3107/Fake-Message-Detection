# Verification Task Definition

This project contains a pattern classifier today. A future evidence-verification
system is a separate research track and must not be represented as a stronger
version of the current classifier.

## Separate tasks

| Task | Input | Output | Evidence required? |
|---|---|---|---:|
| Fake-news classification | Article or message text | Dataset label such as fake/real | No |
| Rumor-veracity classification | Social-media rumor thread | `true`, `false`, `unverified` | No, unless separately annotated |
| Claim verification | Atomic factual claim | Supported/refuted/uncertain status | Yes |
| Evidence retrieval | Claim and corpus/web index | Ranked passages/documents | Yes |
| Source credibility analysis | Source metadata and provenance | Quality factors, not truth | No single permanent truth score |
| Image manipulation detection | Image pixels | Visual-manipulation signal | No factual conclusion by itself |
| OCR text verification | Screenshot/image with text | OCR text plus OCR uncertainty | Evidence still required |
| Multimodal verification | Claim, image, OCR, and evidence | Evidence-grounded status | Yes |

These tasks have different units, labels, leakage risks, and evaluation
protocols. Their results must not be collapsed into one accuracy number.

## Dataset boundaries

- **PHEME** is the current social-media rumor-veracity benchmark with
  `true`, `false`, and `unverified` labels. It does not provide a live web
  evidence corpus and is not sufficient for real-time fact verification.
- **HinFakeNews** is a separate Indian Hindi-oriented news/article
  classification benchmark. Its unresolved content boundary and license make
  it unsuitable for verification training at present. It must not be merged
  with PHEME.
- **FEVER** is a claim-verification benchmark with claims and Wikipedia
  evidence annotations. See the [official dataset page](https://fever.ai/dataset/fever.html)
  and the [NAACL paper](https://aclanthology.org/N18-1074/).
- **AVeriTeC** is a real-world claim-verification benchmark built around
  claims checked by fact-checking organizations and evidence from the web.
  See the [NeurIPS dataset description](https://papers.neurips.cc/paper_files/paper/2023/hash/cd86a30526cd1aff61d6f89f107634e4-Abstract-Datasets_and_Benchmarks.html)
  and the [official task page](https://fever.ai/dataset/averitec.html).

FEVER and AVeriTeC are future verification datasets, not replacements for the
current PHEME experiment. Their licenses, versions, evidence formats, and
split policies must be checked before import.

## Non-goals

The future system will not call a classifier output factual truth, infer truth
from a domain allowlist, or turn missing evidence into `REFUTED`. A generated
explanation is not a citation unless the cited source and excerpt were actually
retrieved and recorded.
