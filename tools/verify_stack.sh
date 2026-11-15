#!/usr/bin/env bash
set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:5173}"
export POST_TEXT="${POST_TEXT:-A public report describes the latest event and evidence.}"

check_json() {
  local url="$1"
  curl --fail --silent --show-error "$url"
}

health="$(check_json "$BACKEND_URL/health")"
python3 -c 'import json,sys; x=json.load(sys.stdin); assert x.get("status") == "ok" and x.get("model_loaded") is True' <<<"$health"

prediction="$(curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,os; print(json.dumps({"text": os.environ["POST_TEXT"]}))')" \
  "$BACKEND_URL/api/v1/predict")"
export prediction
python3 -c 'import json,os; x=json.loads(os.environ["prediction"]); required={"prediction","class_probabilities","confidence_category","warning","model_version","explanation"}; assert required <= x.keys(); assert abs(sum(x["class_probabilities"].values())-1) < 1e-4'

history="$(check_json "$BACKEND_URL/api/v1/history")"
python3 -c 'import json,sys; x=json.load(sys.stdin); assert "items" in x and "summary" in x' <<<"$history"

curl --fail --silent --show-error "$FRONTEND_URL/" >/dev/null
printf '%s\n' "PASS: backend health and model readiness"
printf '%s\n' "PASS: real prediction response and probability sum"
printf '%s\n' "PASS: SQLite-backed history response"
printf '%s\n' "PASS: frontend availability"
