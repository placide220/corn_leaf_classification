"""
Locust load test: simulate a flood of farmers sending leaf photos to the model.

Each simulated user loops forever:
    * sends a random maize leaf image to POST /predict       (10 out of 11 requests)
    * checks GET /health                                     (1 out of 11 requests)
    * waits 1-2 seconds ("think time") before the next request

Run with the web UI (charts in your browser at http://localhost:8089):
    locust -f locustfile.py --host http://localhost:8000

Run headless and save results (what scripts/run_load_tests.sh does):
    locust -f locustfile.py --host http://localhost:8000 --headless \
           -u 100 -r 10 -t 2m --csv results/locust_1_containers

Environment variables:
    LOCUST_IMAGE_DIR        folder of test images (default: data/test, then demo_upload_batch)
    LOCUST_INSTANCE_REPORT  if set, a JSON file listing how many requests each API replica
                            handled is written there at the end of the test
"""

from __future__ import annotations

import io
import json
import os
import random
from collections import Counter
from pathlib import Path

from locust import HttpUser, between, events, task

# --------------------------------------------------------------------------------------
# 1. Load a pool of real test images ONCE (not on every request), so Locust measures the
#    API, not disk reads on the testing machine.
# --------------------------------------------------------------------------------------
MAX_IMAGES = 60
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


REPO_ROOT = Path(__file__).resolve().parent      # works no matter which folder you run locust from


def load_image_pool() -> list[tuple[str, bytes]]:
    candidates = [os.getenv("LOCUST_IMAGE_DIR", ""), REPO_ROOT / "data" / "test",
                  REPO_ROOT / "demo_upload_batch", REPO_ROOT / "data" / "train"]
    for folder in filter(None, candidates):
        files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        if files:
            random.seed(42)
            chosen = random.sample(files, min(MAX_IMAGES, len(files)))
            print(f"[locust] using {len(chosen)} images from {folder}")
            return [(p.name, p.read_bytes()) for p in chosen]

    # No images found: fall back to a few generated green "leaves" so the test still runs.
    from PIL import Image

    print("[locust] WARNING: no images found in " + ", ".join(str(c) for c in candidates if c)
          + ". Using generated test images. Set LOCUST_IMAGE_DIR to a folder of leaf images.")
    pool = []
    for i in range(5):
        buf = io.BytesIO()
        Image.new("RGB", (256, 256), (40 + 20 * i, 150, 50)).save(buf, "JPEG")
        pool.append((f"generated_{i}.jpg", buf.getvalue()))
    return pool


IMAGE_POOL = load_image_pool()

# Count which API replica answered each request (the API sends an X-Instance header).
# With 4 containers you should see the requests spread over 4 different instance ids.
INSTANCE_COUNTS: Counter = Counter()


# --------------------------------------------------------------------------------------
# 2. The simulated user
# --------------------------------------------------------------------------------------
# A request that takes longer than this counts as a FAILURE instead of hanging forever
# (without a timeout, very slow requests never finish and silently disappear from the results).
REQUEST_TIMEOUT_S = float(os.getenv("LOCUST_TIMEOUT", 30))


class FarmerUser(HttpUser):
    # Wait 1-2 s between requests, like a person taking the next photo
    wait_time = between(1, 2)

    @task(10)
    def predict(self):
        name, data = random.choice(IMAGE_POOL)
        # catch_response=True lets us decide what counts as a failure:
        # not only HTTP errors, but also a 200 response without a prediction in it.
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")},
                              name="/predict", timeout=REQUEST_TIMEOUT_S, catch_response=True) as resp:
            INSTANCE_COUNTS[resp.headers.get("X-Instance", "unknown")] += 1
            if resp.status_code == 0:
                resp.failure(f"no response within {REQUEST_TIMEOUT_S:.0f}s (timeout)")
            elif resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:120]}")
            elif "predicted_class" not in resp.text:
                resp.failure("response has no predicted_class")
            else:
                resp.success()

    @task(1)
    def health(self):
        with self.client.get("/health", name="/health", timeout=REQUEST_TIMEOUT_S, catch_response=True) as resp:
            INSTANCE_COUNTS[resp.headers.get("X-Instance", "unknown")] += 1
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")


# --------------------------------------------------------------------------------------
# 3. At the end of the test: print (and optionally save) the requests per replica
# --------------------------------------------------------------------------------------
@events.test_stop.add_listener
def report_instances(environment, **kwargs):
    total = sum(INSTANCE_COUNTS.values()) or 1
    print("\n[locust] requests handled per API replica:")
    for instance, n in INSTANCE_COUNTS.most_common():
        print(f"    {instance:20s} {n:7d}  ({100 * n / total:5.1f}%)")
    out = os.getenv("LOCUST_INSTANCE_REPORT")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(dict(INSTANCE_COUNTS), indent=2))
