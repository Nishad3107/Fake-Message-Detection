"""Run an offline BM25 benchmark on the derived AVeriTeC internal split."""

from __future__ import annotations

import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.averitec_utils import answer_records, content_hash, load_jsonl
from app.claim_extraction import extract_claims
from app.verification_canonicalize import (
    canonical_text,
    canonical_url,
    evidence_document_identity,
    record_signatures,
    stable_hash,
)
from app.verification_input import VerificationInput

ROOT = Path(__file__).resolve().parents[1]
CLEAN_ROOT = ROOT / "data/processed/verification/averitec_clean"
REPORT_ROOT = ROOT / "data/reports/verification"
CHART_ROOT = REPORT_ROOT / "averitec_internal_retrieval"
SEED = 42
TOP_KS = (1, 3, 5, 10)
QUERY_TYPES = (
    "original_claim",
    "normalized_claim",
    "claim_entities",
    "claim_location",
    "claim_date",
    "fact_check_questions",
    "combined_questions",
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class InternalDocument:
    document_id: str
    title: str
    text: str
    url: str | None
    content_hash: str
    evidence_hash: str
    evidence_document_identity: str


class FastBM25:
    def __init__(
        self, documents: list[InternalDocument], *, field: str, tokenizer: str
    ):
        self.documents = documents
        self.field = field
        self.tokenizer_name = tokenizer
        self.tokens = [self._tokenize(self._field_text(doc)) for doc in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.inverted: dict[str, dict[int, int]] = defaultdict(dict)
        for index, tokens in enumerate(self.tokens):
            for token in tokens:
                self.inverted[token][index] = self.inverted[token].get(index, 0) + 1
        self.idf = {
            token: math.log(
                1 + (len(documents) - len(postings) + 0.5) / (len(postings) + 0.5)
            )
            for token, postings in self.inverted.items()
        }

    def _tokenize(self, text: str) -> list[str]:
        tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
        if self.tokenizer_name == "normalized":
            tokens = [canonical_text(token) for token in tokens]
        if self.tokenizer_name.endswith("_stopwords"):
            tokens = [token for token in tokens if token not in STOPWORDS]
        if self.tokenizer_name.endswith("_stemmed"):
            tokens = [self._stem(token) for token in tokens]
        return [token for token in tokens if token]

    @staticmethod
    def _stem(token: str) -> str:
        for suffix in ("ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 3:
                return token[: -len(suffix)]
        return token

    def _field_text(self, document: InternalDocument) -> str:
        return (
            document.text
            if self.field == "evidence_only"
            else f"{document.title} {document.text}"
        )

    def retrieve(
        self, query: str, top_k: int = 10
    ) -> list[tuple[InternalDocument, float]]:
        query_tokens = self._tokenize(query)
        scores: dict[int, float] = defaultdict(float)
        k1, b = 1.5, 0.75
        for term in set(query_tokens):
            postings = self.inverted.get(term, {})
            idf = self.idf.get(term, 0.0)
            for index, frequency in postings.items():
                denominator = frequency + k1 * (
                    1 - b + b * self.lengths[index] / max(self.average_length, 1)
                )
                scores[index] += idf * frequency * (k1 + 1) / denominator
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self.documents[item[0]].document_id),
        )
        return [
            (self.documents[index], round(score, 8)) for index, score in ranked[:top_k]
        ]


def _documents(rows: list[dict[str, Any]]) -> list[InternalDocument]:
    documents: dict[str, InternalDocument] = {}
    for row in rows:
        for answer in answer_records(row):
            text = answer.get("answer")
            if not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()
            url = answer.get("source_url") or answer.get("cached_source_url")
            url = url.strip() if isinstance(url, str) and url.strip() else None
            digest = content_hash(text)
            document_id = "averitec-clean-train-" + digest.removeprefix("sha256:")[:20]
            if document_id not in documents:
                documents[document_id] = InternalDocument(
                    document_id=document_id,
                    title=str(answer.get("question") or "AVeriTeC evidence"),
                    text=text,
                    url=url,
                    content_hash=digest,
                    evidence_hash=stable_hash(canonical_text(text)),
                    evidence_document_identity=evidence_document_identity(text, url),
                )
    return list(documents.values())


def _gold_signatures(row: dict[str, Any]) -> dict[str, set[str]]:
    signatures = record_signatures(row)
    return {
        "evidence_hashes": set(signatures["evidence_hashes"]),
        "document_ids": set(signatures["evidence_document_hashes"]),
        "url_hashes": set(signatures["url_hashes"]),
    }


def _claim_features(row: dict[str, Any]) -> dict[str, Any]:
    result = extract_claims(
        VerificationInput(input_type="plain_text", text=str(row.get("claim", "")))
    )
    claim = result.claims[0] if result.claims else None
    extraction = claim.model_dump() if claim else {}
    questions = [
        str(item.get("question")).strip()
        for item in row.get("questions", [])
        if isinstance(item.get("question"), str) and item.get("question").strip()
    ]
    return {
        "language": result.language,
        "claim_type": extraction.get("claim_type", "unknown"),
        "entities": extraction.get("entities", []),
        "location": row.get("location"),
        "claim_date": row.get("claim_date"),
        "questions": questions,
    }


def _queries(row: dict[str, Any], features: dict[str, Any]) -> list[tuple[str, str]]:
    claim = str(row.get("claim", ""))
    normalized = canonical_text(claim)
    entities = " ".join(features["entities"])
    location = str(features["location"] or "").strip()
    date = str(features["claim_date"] or "").strip()
    questions = " ".join(features["questions"])
    values = {
        "original_claim": claim,
        "normalized_claim": normalized,
        "claim_entities": f"{claim} {entities}".strip(),
        "claim_location": f"{claim} {location}".strip(),
        "claim_date": f"{claim} {date}".strip(),
        "fact_check_questions": questions,
        "combined_questions": f"{claim} {questions}".strip(),
    }
    return [(query_type, query) for query_type, query in values.items() if query]


def _configurations() -> list[dict[str, str]]:
    configurations = []
    for field in ("evidence_only", "title_plus_evidence"):
        for tokenizer in ("original", "normalized"):
            for stopwords in (False, True):
                for stemming in (False, True):
                    suffix = "_stopwords" if stopwords else ""
                    suffix += "_stemmed" if stemming else ""
                    configurations.append(
                        {
                            "field": field,
                            "tokenizer": tokenizer + suffix,
                            "stopwords": str(stopwords).lower(),
                            "stemming": str(stemming).lower(),
                        }
                    )
    return configurations


def _metric(retrieved: list[InternalDocument], relevant: set[str], k: int) -> int:
    return int(bool({doc.document_id for doc in retrieved[:k]} & relevant))


def _failure_category(
    row: dict[str, Any],
    features: dict[str, Any],
    coverage: int,
    retrieved: list[InternalDocument],
) -> str:
    if not coverage:
        return "missing_evidence"
    if features["entities"] and not any(
        any(entity.casefold() in doc.text.casefold() for entity in features["entities"])
        for doc in retrieved
    ):
        return "entity_mismatch"
    if features["claim_date"] and not any(
        features["claim_date"] in doc.text for doc in retrieved
    ):
        return "date_mismatch"
    if re.search(r"\d|%", str(row.get("claim", ""))):
        return "numeric_claim"
    return "retrieval_rank_failure"


def _mean_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_count": len(items),
        **{
            metric: round(sum(float(item[metric]) for item in items) / len(items), 8)
            for metric in (
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "recall_at_10",
                "mrr",
                "evidence_hit_rate",
                "gold_evidence_coverage",
            )
        },
    }


def _save_charts(
    results: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHART_ROOT.mkdir(parents=True, exist_ok=True)
    by_query = {
        name: [row for row in aggregate if row["query_type"] == name]
        for name in QUERY_TYPES
    }
    query_names = [name for name in QUERY_TYPES if by_query[name]]
    for metric in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10"):
        values = [
            sum(item[metric] for item in by_query[name]) / len(by_query[name])
            for name in query_names
        ]
        plt.figure(figsize=(10, 5))
        plt.bar(query_names, values)
        plt.ylabel(metric)
        plt.xlabel("Query type")
        plt.title(
            f"AVeriTeC internal retrieval: {metric}\nInternal development result, not official performance"
        )
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(CHART_ROOT / f"{metric}_by_query_type.png", dpi=140)
        plt.close()
    values = [
        sum(item["mrr"] for item in by_query[name]) / len(by_query[name])
        for name in query_names
    ]
    plt.figure(figsize=(10, 5))
    plt.bar(query_names, values)
    plt.ylabel("MRR")
    plt.xlabel("Query type")
    plt.title(
        "MRR by query type\nInternal development result, not official performance"
    )
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(CHART_ROOT / "mrr_by_query_type.png", dpi=140)
    plt.close()
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[str(row["label"])].append(row)
    keys = sorted(grouped)
    plt.figure(figsize=(9, 4))
    plt.bar(
        keys,
        [
            sum(row["evidence_hit_rate"] for row in grouped[key]) / len(grouped[key])
            for key in keys
        ],
    )
    plt.ylabel("Evidence hit rate")
    plt.xlabel("Label")
    plt.title(
        "Evidence hit rate by label\nInternal development result, not official performance"
    )
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(CHART_ROOT / "evidence_hit_rate_by_label.png", dpi=140)
    plt.close()
    grouped = defaultdict(list)
    for row in results:
        grouped[str(row["claim_type"])].append(row)
    keys = sorted(grouped)
    plt.figure(figsize=(9, 4))
    plt.bar(
        keys,
        [
            sum(row["recall_at_10"] for row in grouped[key]) / len(grouped[key])
            for key in keys
        ],
    )
    plt.ylabel("Recall@10")
    plt.xlabel("Claim type")
    plt.title(
        "Recall@10 by claim type\nInternal development result, not official performance"
    )
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(CHART_ROOT / "recall_at_10_by_claim_type.png", dpi=140)
    plt.close()
    runtimes = {
        row["configuration"]: row["configuration_runtime_seconds"] for row in aggregate
    }
    keys = sorted(runtimes)
    plt.figure(figsize=(11, 4))
    plt.bar(keys, [runtimes[key] for key in keys])
    plt.ylabel("Seconds")
    plt.xlabel("Configuration")
    plt.title(
        "Runtime by configuration\nInternal development result, not official performance"
    )
    plt.xticks(rotation=70, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(CHART_ROOT / "runtime_by_configuration.png", dpi=140)
    plt.close()
    scores = [score for row in results for score in row["top_scores"]]
    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=20)
    plt.ylabel("Count")
    plt.xlabel("BM25 score")
    plt.title(
        "BM25 score distribution\nInternal development result, not official performance"
    )
    plt.tight_layout()
    plt.savefig(CHART_ROOT / "score_distribution.png", dpi=140)
    plt.close()
    counts = Counter(item["failure_category"] for item in failures)
    plt.figure(figsize=(8, 4))
    plt.bar(list(counts), list(counts.values()))
    plt.ylabel("Failure count")
    plt.xlabel("Failure category")
    plt.title(
        "Retrieval failures by category\nInternal development result, not official performance"
    )
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(CHART_ROOT / "failures_by_category.png", dpi=140)
    plt.close()


def run_benchmark() -> dict[str, Any]:
    train_path = CLEAN_ROOT / "internal_train.jsonl"
    dev_path = CLEAN_ROOT / "internal_dev.jsonl"
    if not train_path.is_file() or not dev_path.is_file():
        raise FileNotFoundError("Run make verify-averitec-clean-benchmark first")
    train_rows, dev_rows = load_jsonl(train_path), load_jsonl(dev_path)
    documents = _documents(train_rows)
    document_by_hash = {doc.evidence_hash: doc for doc in documents}
    document_by_identity = {doc.evidence_document_identity: doc for doc in documents}
    corpus_url_hashes = {
        stable_hash(canonical_url(doc.url)) for doc in documents if doc.url
    }
    for row in dev_rows:
        gold = _gold_signatures(row)
        if (
            gold["evidence_hashes"] & set(document_by_hash)
            or gold["document_ids"] & set(document_by_identity)
            or gold["url_hashes"] & corpus_url_hashes
        ):
            raise RuntimeError(
                f"Evaluation evidence leakage detected for {row['record_id']}"
            )
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    evaluation_items = [
        (row, _claim_features(row), _gold_signatures(row)) for row in dev_rows
    ]
    for config in _configurations():
        config_started = time.perf_counter()
        retriever = FastBM25(
            documents, field=config["field"], tokenizer=config["tokenizer"]
        )
        for row, features, gold in evaluation_items:
            relevant = {
                document_by_hash[item].document_id
                for item in gold["evidence_hashes"]
                if item in document_by_hash
            }
            for query_type, query in _queries(row, features):
                retrieved_scored = retriever.retrieve(query, top_k=10)
                retrieved = [item[0] for item in retrieved_scored]
                scores = [score for _, score in retrieved_scored]
                coverage = int(bool(relevant))
                result = {
                    "benchmark_status": "INTERNAL DEVELOPMENT RESULT",
                    "official_performance": False,
                    "live_web_performance": False,
                    "record_id": row["record_id"],
                    "query_type": query_type,
                    "query_text": query,
                    "config_field": config["field"],
                    "config_tokenizer": config["tokenizer"],
                    "stopwords": config["stopwords"],
                    "stemming": config["stemming"],
                    "corpus_size": len(documents),
                    "gold_evidence_coverage": coverage,
                    "retrieved_document_ids": [doc.document_id for doc in retrieved],
                    "top_scores": scores,
                    "recall_at_1": _metric(retrieved, relevant, 1),
                    "recall_at_3": _metric(retrieved, relevant, 3),
                    "recall_at_5": _metric(retrieved, relevant, 5),
                    "recall_at_10": _metric(retrieved, relevant, 10),
                    "mrr": next(
                        (
                            1.0 / (index + 1)
                            for index, doc in enumerate(retrieved)
                            if doc.document_id in relevant
                        ),
                        0.0,
                    ),
                    "evidence_hit_rate": int(
                        bool({doc.document_id for doc in retrieved} & relevant)
                    ),
                    "label": row.get("label"),
                    "claim_type": features["claim_type"],
                    "location": row.get("location") or "__missing__",
                    "language": features["language"],
                    "question_count": len(features["questions"]),
                    "evidence_count": len(answer_records(row)),
                    "claim_length": len(str(row.get("claim", ""))),
                    "failure_category": None,
                }
                if not result["evidence_hit_rate"]:
                    result["failure_category"] = _failure_category(
                        row, features, coverage, retrieved
                    )
                    failures.append(
                        {
                            "record_id": row["record_id"],
                            "claim_excerpt": str(row.get("claim", ""))[:240],
                            "query_type": query_type,
                            "query_text": query,
                            "gold_evidence_ids": sorted(gold["evidence_hashes"]),
                            "top_retrieved_ids": [doc.document_id for doc in retrieved],
                            "top_retrieval_scores": scores,
                            "claim_type": features["claim_type"],
                            "verdict_label": row.get("label"),
                            "failure_category": result["failure_category"],
                        }
                    )
                results.append(result)
        config_runtime = round(time.perf_counter() - config_started, 6)
        for result in results:
            if (
                result["config_field"] == config["field"]
                and result["config_tokenizer"] == config["tokenizer"]
            ):
                result["configuration_runtime_seconds"] = config_runtime
    runtime = round(time.perf_counter() - started, 6)
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[
            (
                result["config_field"] + ":" + result["config_tokenizer"],
                result["query_type"],
            )
        ].append(result)
    aggregate = []
    for (configuration, query_type), items in grouped.items():
        aggregate.append(
            {
                "configuration": configuration,
                "query_type": query_type,
                "query_count": len(items),
                "runtime_seconds": items[0]["configuration_runtime_seconds"],
                "configuration_runtime_seconds": items[0][
                    "configuration_runtime_seconds"
                ],
                **{
                    metric: round(
                        sum(float(item[metric]) for item in items) / len(items), 8
                    )
                    for metric in (
                        "recall_at_1",
                        "recall_at_3",
                        "recall_at_5",
                        "recall_at_10",
                        "mrr",
                        "evidence_hit_rate",
                        "gold_evidence_coverage",
                    )
                },
                "failure_count": sum(
                    item["failure_category"] is not None for item in items
                ),
            }
        )
    by_dimensions: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension in (
        "label",
        "claim_type",
        "location",
        "language",
        "question_count",
        "evidence_count",
        "claim_length",
        "query_type",
    ):
        groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in results:
            groups[str(item[dimension])].append(item)
        by_dimensions[dimension] = {
            key: _mean_metrics(items) for key, items in sorted(groups.items())
        }
    summary = {
        "benchmark_status": "INTERNAL DEVELOPMENT RESULT",
        "official_performance": False,
        "live_web_performance": False,
        "benchmark": "averitec_clean",
        "seed": SEED,
        "internal_train_size": len(train_rows),
        "internal_dev_size": len(dev_rows),
        "corpus_size": len(documents),
        "query_count": len(results),
        "runtime_seconds": runtime,
        "gold_evidence_coverage_rate": round(
            sum(item["gold_evidence_coverage"] for item in results) / len(results), 8
        ),
        "aggregate_results": aggregate,
        "by_dimensions": by_dimensions,
        "best_configuration": (
            max(
                aggregate,
                key=lambda item: (
                    item["recall_at_10"],
                    item["mrr"],
                    item["evidence_hit_rate"],
                    -item["runtime_seconds"],
                ),
            )
            if aggregate
            else None
        ),
        "main_limitation": "Gold development evidence is intentionally absent from the internal-train corpus; retrieval metrics are therefore coverage-limited and must not be interpreted as realistic independent retrieval performance.",
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "averitec_internal_retrieval_results.json").write_text(
        json.dumps(
            {"summary": summary, "results": results}, indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )
    with (REPORT_ROOT / "averitec_internal_retrieval_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (REPORT_ROOT / "averitec_internal_retrieval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (REPORT_ROOT / "averitec_internal_retrieval_failures.json").write_text(
        json.dumps(
            {
                "benchmark_status": "INTERNAL DEVELOPMENT RESULT",
                "official_performance": False,
                "failures": failures,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _save_charts(results, aggregate, failures)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, ensure_ascii=False))
