"""Audit HinFakeNews CONTENT for annotation leakage and near duplicates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
INPUT = BACKEND_ROOT / "data/processed/indian/hinfakenews_content.csv"
REPORT_DIR = BACKEND_ROOT / "data/reports/indian"
STRUCTURE_REPORT = REPORT_DIR / "hinfakenews_content_structure.json"
FLAGS_REPORT = REPORT_DIR / "hinfakenews_content_leakage_flags.csv"
VARIANT_REPORT = REPORT_DIR / "content_variant_comparison.json"
NEAR_DUP_REPORT = REPORT_DIR / "hinfakenews_near_duplicates.json"
CANDIDATE_DIR = BACKEND_ROOT / "data/processed/indian/candidates"
SPLIT_MANIFEST = BACKEND_ROOT / "data/splits/indian/hinfakenews_split_manifest.json"

PATTERNS: dict[str, tuple[str, ...]] = {
    "contains_verdict_marker": (
        r"\bverdict\b",
        r"\bconclusion\b",
        r"\bफैसला\b",
        r"\bनिष्कर्ष\b",
        r"\bनतीजा\b",
        r"\bसच या झूठ\b",
    ),
    "contains_fact_check_marker": (
        r"fact[- ]?check",
        r"फैक्ट\s*-?चेक",
        r"तथ्य\s*जांच",
        r"पड़ताल",
        r"सच्चाई",
    ),
    "contains_claim_marker": (r"\bclaim\b", r"दावा", r"दावे"),
    "contains_debunking_language": (
        r"\bfake\b",
        r"\bfalse\b",
        r"\bmisleading\b",
        r"\bdebunk",
        r"\bफर्जी\b",
        r"\bफेक\b",
        r"\bझूठ\b",
        r"\bभ्रामक\b",
        r"\bनकली\b",
    ),
    "contains_label_word": (
        r"\bfake\b",
        r"\breal\b",
        r"\bfalse\b",
        r"\btrue\b",
        r"\bफर्जी\b",
        r"\bफेक\b",
        r"\bसही\b",
        r"\bगलत\b",
        r"\bसच\b",
    ),
    "contains_source_metadata": (
        r"https?://",
        r"www\.",
        r"\bsource\b",
        r"\bस्रोत\b",
        r"\bसंवाददाता\b",
        r"\bरिपोर्टर\b",
    ),
    "contains_html": (r"<[^>]+>", r"&(?:amp|quot|lt|gt);"),
}


def _read_rows(path: Path = INPUT) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared Indian dataset not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"record_id", "text_original", "label", "title", "url", "domain"}
    missing = sorted(required - set(rows[0]) if rows else required)
    if missing:
        raise ValueError(
            f"Prepared dataset missing required columns: {', '.join(missing)}"
        )
    return rows


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def detect_flags(
    row: dict[str, str],
    repeated_templates: set[str] | None = None,
    normalized_value: str | None = None,
) -> dict[str, bool]:
    text = row["text_original"]
    flags = {
        name: any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
        for name, patterns in PATTERNS.items()
    }
    normalized_prefix = (normalized_value or normalize_text(text)).casefold()[:100]
    flags["contains_repeated_template"] = bool(
        repeated_templates and normalized_prefix in repeated_templates
    )
    return flags


def _flag_examples(
    rows: list[dict[str, str]], flags_by_id: dict[str, dict[str, bool]]
) -> dict[str, list[str]]:
    examples: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for name, enabled in flags_by_id[row["record_id"]].items():
            if enabled and len(examples[name]) < 3:
                examples[name].append(normalize_text(row["text_original"])[:100])
    return dict(examples)


def _label_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["label"] for row in rows).items()))


def _write_candidates(
    rows: list[dict[str, str]],
    flags_by_id: dict[str, dict[str, bool]],
    normalized_by_id: dict[str, str],
) -> dict[str, dict]:
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    variants = {
        "content_raw": lambda row: row["text_original"],
        "content_clean": lambda row: normalized_by_id[row["record_id"]],
        "content_article_only_candidate": lambda row: normalized_by_id[
            row["record_id"]
        ],
        "title_only": lambda row: row["title"],
        "title_content_candidate": lambda row: f"{row['title']} {row['text_original']}".strip(),
    }
    comparison: dict[str, dict] = {}
    fields = [
        "record_id",
        "text",
        "label",
        "label_original",
        "title",
        "url",
        "domain",
        "source_row",
        "dataset_name",
    ]
    for name, getter in variants.items():
        path = CANDIDATE_DIR / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            lengths: list[int] = []
            word_lengths: list[int] = []
            changed = 0
            empty = 0
            for row in rows:
                text = getter(row)
                lengths.append(len(text))
                word_lengths.append(len(text.split()))
                empty += not text.strip()
                changed += text != row["text_original"]
                writer.writerow(
                    {
                        "record_id": row["record_id"],
                        "text": text,
                        "label": row["label"],
                        "label_original": row.get("label_original", ""),
                        "title": row["title"],
                        "url": row["url"],
                        "domain": row["domain"],
                        "source_row": row.get("source_row", ""),
                        "dataset_name": "HinFakeNews",
                    }
                )
        comparison[name] = {
            "path": str(path),
            "records": len(rows),
            "empty_text_count": empty,
            "changed_records_vs_original_content": changed,
            "character_statistics": {
                "minimum": min(lengths),
                "maximum": max(lengths),
                "mean": round(sum(lengths) / len(lengths), 4),
            },
            "word_statistics": {
                "minimum": min(word_lengths),
                "maximum": max(word_lengths),
                "mean": round(sum(word_lengths) / len(word_lengths), 4),
            },
            "flagged_records": sum(
                any(flags_by_id[row["record_id"]].values()) for row in rows
            ),
            "label_distribution": _label_distribution(rows),
            "domain_distribution": dict(
                Counter(
                    row["domain"] or "__missing_domain__" for row in rows
                ).most_common()
            ),
            "leakage_status": (
                "manual review required"
                if name == "content_article_only_candidate"
                else "candidate only"
            ),
        }
    comparison["content_article_only_candidate"][
        "handling_note"
    ] = "No ambiguous fact-check boundary was removed automatically; this candidate currently equals normalized CONTENT."
    return comparison


def _shingles(text: str) -> set[str]:
    tokens = re.findall(r"\w+", text.casefold(), re.UNICODE)
    values = {
        " ".join(tokens[index : index + 3])
        for index in range(min(max(0, len(tokens) - 2), 200))
    }
    return values or {text.casefold()}


def _minhash_signature(text: str) -> tuple[int, ...]:
    shingles = _shingles(text)
    signature = [2**32 - 1] * 8
    for shingle in shingles:
        digest = hashlib.sha256(shingle.encode("utf-8")).digest()
        for index in range(8):
            value = int.from_bytes(digest[index * 4 : index * 4 + 4], "big")
            signature[index] = min(signature[index], value)
    return tuple(signature)


def _near_duplicate_report(
    rows: list[dict[str, str]],
    normalized_by_id: dict[str, str],
    split_manifest: Path = SPLIT_MANIFEST,
) -> dict:
    exact_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        exact_groups[normalized_by_id[row["record_id"]].casefold()].append(
            row["record_id"]
        )
    exact_duplicate_groups = [ids for ids in exact_groups.values() if len(ids) > 1]
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    signatures = []
    for index, row in enumerate(rows):
        signature = _minhash_signature(normalized_by_id[row["record_id"]])
        signatures.append(signature)
        for band in range(4):
            start = band * 2
            buckets[(band, signature[start : start + 2])].append(index)
    pairs: set[tuple[int, int]] = set()
    for indices in buckets.values():
        if len(indices) > 500:
            continue
        pairs.update(tuple(sorted(pair)) for pair in itertools.combinations(indices, 2))
    near_pairs = []
    for left, right in pairs:
        left_shingles = _shingles(normalized_by_id[rows[left]["record_id"]])
        right_shingles = _shingles(normalized_by_id[rows[right]["record_id"]])
        similarity = len(left_shingles & right_shingles) / len(
            left_shingles | right_shingles
        )
        if similarity >= 0.85 and left_shingles != right_shingles:
            near_pairs.append((left, right, round(similarity, 4)))
    split_by_domain: dict[str, str] = {}
    if split_manifest.is_file():
        manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
        for split, details in manifest.get("splits", {}).items():
            for domain in details.get("domains", []):
                split_by_domain[domain] = split
    cross_split = []
    cross_label = []
    cross_domain = []
    for left, right, similarity in near_pairs:
        first, second = rows[left], rows[right]
        pair = {
            "record_ids": [first["record_id"], second["record_id"]],
            "jaccard_similarity": similarity,
            "labels": [first["label"], second["label"]],
            "domains": [first["domain"], second["domain"]],
        }
        if first["label"] != second["label"]:
            cross_label.append(pair)
        if first["domain"] != second["domain"]:
            cross_domain.append(pair)
        first_split = split_by_domain.get(first["domain"] or "__missing_domain__")
        second_split = split_by_domain.get(second["domain"] or "__missing_domain__")
        if first_split and second_split and first_split != second_split:
            pair["splits"] = [first_split, second_split]
            cross_split.append(pair)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "method": "normalized exact matching plus deterministic 8-hash MinHash LSH over word trigrams, verified with Jaccard similarity",
        "minhash_bands": 4,
        "minhash_rows_per_band": 2,
        "jaccard_similarity_threshold": 0.85,
        "candidate_bucket_count": len(buckets),
        "exact_duplicate_group_count": len(exact_duplicate_groups),
        "exact_duplicate_record_count": sum(
            len(group) for group in exact_duplicate_groups
        ),
        "near_duplicate_pair_count": len(near_pairs),
        "near_duplicate_affected_record_count": len(
            {index for pair in near_pairs for index in pair[:2]}
        ),
        "cross_label_near_duplicate_pair_count": len(cross_label),
        "cross_domain_near_duplicate_pair_count": len(cross_domain),
        "cross_split_near_duplicate_pair_count": len(cross_split),
        "example_safe_near_duplicate_pairs": near_pairs[:20]
        and [
            {
                "record_ids": [rows[left]["record_id"], rows[right]["record_id"]],
                "jaccard_similarity": similarity,
                "labels": [rows[left]["label"], rows[right]["label"]],
                "domains": [rows[left]["domain"], rows[right]["domain"]],
            }
            for left, right, distance in near_pairs[:20]
        ],
        "split_status": (
            "unsafe_for_final_evaluation"
            if cross_split
            else "no_cross_split_pairs_detected_by_this_method"
        ),
        "limitations": [
            "MinHash/LSH is an approximate screening method, not semantic equivalence.",
            "Near-duplicate review should be repeated with a predeclared threshold before final training.",
        ],
    }


def audit(input_path: Path = INPUT) -> dict:
    rows = _read_rows(input_path)
    normalized_by_id = {
        row["record_id"]: normalize_text(row["text_original"]) for row in rows
    }
    prefixes = Counter(
        normalized_by_id[row["record_id"]].casefold()[:100] for row in rows
    )
    repeated_templates = {prefix for prefix, count in prefixes.items() if count > 1}
    flags_by_id = {
        row["record_id"]: detect_flags(
            row, repeated_templates, normalized_by_id[row["record_id"]]
        )
        for row in rows
    }
    structure = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": "HinFakeNews",
        "records": len(rows),
        "patterns": {},
        "recommendation": "Manual content-boundary review required",
    }
    for name in [*PATTERNS, "contains_repeated_template"]:
        matching = [
            row for row in rows if flags_by_id[row["record_id"]].get(name, False)
        ]
        structure["patterns"][name] = {
            "count": len(matching),
            "label_distribution": _label_distribution(matching),
            "example_safe_markers": list(PATTERNS.get(name, ("repeated_prefix",)))[:3],
            "likely_annotation_or_metadata": name
            in {
                "contains_verdict_marker",
                "contains_fact_check_marker",
                "contains_claim_marker",
                "contains_debunking_language",
                "contains_label_word",
            },
            "recommended_handling": "manual review before training; do not remove automatically",
        }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STRUCTURE_REPORT.write_text(
        json.dumps(structure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    flag_fields = [
        "record_id",
        "label",
        "domain",
        "title",
        "url",
        "text_original",
        *sorted(next(iter(flags_by_id.values()), {})),
    ]
    with FLAGS_REPORT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flag_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{
                        field: row.get(field, "")
                        for field in [
                            "record_id",
                            "label",
                            "domain",
                            "title",
                            "url",
                            "text_original",
                        ]
                    },
                    **flags_by_id[row["record_id"]],
                }
            )
    variants = _write_candidates(rows, flags_by_id, normalized_by_id)
    VARIANT_REPORT.write_text(
        json.dumps(variants, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    near_duplicates = _near_duplicate_report(rows, normalized_by_id)
    NEAR_DUP_REPORT.write_text(
        json.dumps(near_duplicates, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "structure": structure,
        "variants": variants,
        "near_duplicates": near_duplicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.input), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
