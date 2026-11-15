#!/usr/bin/env bash
set -euo pipefail

tracked_files="$(git ls-files)"

fail() {
  printf 'Repository check failed: %s\n' "$1" >&2
  exit 1
}

if printf '%s\n' "$tracked_files" | grep -E '(^|/)\.env$|(^|/)\.env\.[^e]' >/dev/null; then
  fail "tracked environment file detected"
fi
if printf '%s\n' "$tracked_files" | grep -E '(^|/)backend/data/raw/pheme/|(^|/)backend/models/.*\.(joblib|pkl)$|(^|/).*\.db$|(^|/)backend/data/experiments/' | grep -v '/\.gitkeep$' >/dev/null; then
  fail "tracked dataset, model, database, or experiment artifact detected"
fi

if git grep -n -I -E '/Users/|/home/|[A-Za-z]:\\\\Users\\\\' -- . >/dev/null; then
  fail "user-specific absolute path detected"
fi
if git grep -n -I -E 'sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}' -- . >/dev/null; then
  fail "possible hardcoded credential detected"
fi

printf '%s\n' "Repository checks passed: no tracked secrets, private paths, raw PHEME data, generated models, databases, or experiment outputs."
