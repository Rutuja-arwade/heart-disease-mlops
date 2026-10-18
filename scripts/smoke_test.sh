#!/usr/bin/env bash
# Smoke-test a running API instance. Fails (non-zero exit) on any unexpected response.
# Usage: ./scripts/smoke_test.sh [BASE_URL] [extra curl args, e.g. -H "Host: heart-api.local"]
set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
shift || true
CURL_ARGS=("$@")
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Waiting for ${BASE_URL}/ready"
for i in $(seq 1 30); do
  if curl -fsS ${CURL_ARGS[@]+"${CURL_ARGS[@]}"} "${BASE_URL}/ready" >/dev/null 2>&1; then break; fi
  if [ "$i" -eq 30 ]; then echo "API did not become ready"; exit 1; fi
  sleep 2
done

echo "==> GET /health"
curl -fsS ${CURL_ARGS[@]+"${CURL_ARGS[@]}"} "${BASE_URL}/health"; echo

echo "==> POST /predict (sample_request.json)"
RESPONSE=$(curl -fsS ${CURL_ARGS[@]+"${CURL_ARGS[@]}"} -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" -d @"${HERE}/sample_request.json")
echo "${RESPONSE}"
echo "${RESPONSE}" | python3 -c "
import json, sys
body = json.load(sys.stdin)
assert body['prediction'] in (0, 1), body
assert 0.5 <= body['confidence'] <= 1.0, body
print(f\"OK prediction={body['prediction']} confidence={body['confidence']} risk={body['risk_level']}\")
"

echo "==> POST /predict with invalid payload (expect 422)"
CODE=$(curl -s -o /dev/null -w "%{http_code}" ${CURL_ARGS[@]+"${CURL_ARGS[@]}"} -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" -d '{"age": 300, "sex": 1}')
echo "HTTP ${CODE}"
[ "${CODE}" = "422" ] || { echo "expected 422"; exit 1; }

echo "==> GET /metrics (excerpt)"
curl -fsS ${CURL_ARGS[@]+"${CURL_ARGS[@]}"} "${BASE_URL}/metrics" | grep -E "^heart_api_(requests_total|predictions_total|model_loaded)" | head -8

echo "Smoke test passed for ${BASE_URL}"
