#!/usr/bin/env bash
# =====================================================================================
# Flood-request experiment: same Locust load against 1, 2 and 4 API containers.
#
# Usage (from the repository root, with Docker running):
#     bash scripts/run_load_tests.sh
#
# Optional settings (environment variables):
#     REPLICAS="1 2 4"   container counts to test
#     USERS=100          simulated users at the same time
#     SPAWN=10           users started per second
#     DURATION=2m        test length per container count
#
# Output (in results/):
#     locust_<N>_containers_stats.csv / _stats_history.csv / .html   raw Locust results
#     locust_<N>_containers_instances.json                         requests per replica
#     load_test_summary.md + load_test_comparison.png             table + chart for the README
# =====================================================================================
set -euo pipefail

REPLICAS="${REPLICAS:-1 2 4}"
USERS="${USERS:-100}"
SPAWN="${SPAWN:-10}"
DURATION="${DURATION:-2m}"
HOST="http://localhost:8000"

cd "$(dirname "$0")/.."            # always run from the repository root
mkdir -p results

echo "==> Building images (only slow the first time)"
docker compose build api

for n in $REPLICAS; do
  echo
  echo "=================================================================="
  echo "==> Testing with $n API container(s): $USERS users for $DURATION"
  echo "=================================================================="

  # Start (or resize to) n API replicas + nginx. The UI is not needed for load testing.
  docker compose up -d --scale api="$n" api nginx

  # Wait until n DIFFERENT replicas answer through nginx (models loaded, nginx sees them all)
  echo -n "    waiting for $n healthy replica(s)"
  for attempt in $(seq 1 60); do
    seen=$(for i in $(seq 1 $((n * 6))); do
             curl -s -m 3 "$HOST/health" | python3 -c "import sys, json; print(json.load(sys.stdin).get('instance', ''))" 2>/dev/null || true
           done | sort -u | grep -c . || true)
    if [ "$seen" -ge "$n" ]; then echo " ✓ ($seen replicas answering)"; break; fi
    echo -n "."; sleep 5
    if [ "$attempt" -eq 60 ]; then echo " ✗ only $seen of $n replicas answered, check 'docker compose ps'"; exit 1; fi
  done

  # Warm-up: a few predictions so first-request overhead is not counted in the results
  for i in $(seq 1 $((n * 3))); do
    img=$(find data/test -name '*.jpg' | head -1)
    curl -s -o /dev/null -F "file=@$img" "$HOST/predict" || true
  done

  # The actual test. --csv writes the numbers, --html a report with charts.
  LOCUST_INSTANCE_REPORT="results/locust_${n}_containers_instances.json" \
  locust -f locustfile.py --host "$HOST" --headless \
         -u "$USERS" -r "$SPAWN" -t "$DURATION" \
         --csv "results/locust_${n}_containers" \
         --html "results/locust_${n}_containers.html" \
         --only-summary

  sleep 5   # let the replicas settle before the next round
done

echo
echo "==> Summarising"
python3 scripts/summarize_locust.py --replicas $REPLICAS --users "$USERS" --duration "$DURATION"
echo "Done. Copy results/load_test_summary.md into the README and add results/load_test_comparison.png."
