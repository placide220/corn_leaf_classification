"""
End-to-end self-test for the Corn Leaf Disease API: health -> predict -> zip upload -> retraining.

Run it against any running API (local Docker, `uvicorn`, or the Azure deployment):

    python scripts/smoke_test.py                                   # http://localhost:8000
    python scripts/smoke_test.py --url https://<your-api>.azurecontainerapps.io
    python scripts/smoke_test.py --skip-retrain                    # only health + predict + upload

Every step prints PASS or FAIL with the reason, so you know exactly which part is broken.

The zip uploaded for retraining is demo_upload_batch.zip (or the demo_upload_batch/ folder),
i.e. images the model has never seen. Test images are NEVER uploaded, because adding them to
the training set would make the test accuracy dishonest.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
RESULTS = {"pass": 0, "fail": 0}


def report(ok: bool, step: str, detail: str = "") -> bool:
    RESULTS["pass" if ok else "fail"] += 1
    print(f"  {'PASS' if ok else 'FAIL'}  {step}" + (f"  ->  {detail}" if detail else ""))
    return ok


def find_test_image() -> Path | None:
    for cls in ("Common_Rust", "Healthy", "Blight", "Gray_Leaf_Spot"):
        files = sorted((REPO / "data" / "test" / cls).glob("*.jpg"))
        if files:
            return files[0]
    return None


def non_leaf_images() -> list[tuple[str, bytes]]:
    """Pictures that are clearly NOT maize leaves: random noise and a blank white 'document'.
    Put your own extra test pictures (cat, car, ...) in scripts/not_leaf_samples/ to test those too."""
    from PIL import Image
    import numpy as np

    out = []
    rng = np.random.default_rng(0)
    for name, img in [("random_noise.jpg", Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))),
                      ("blank_page.jpg", Image.new("RGB", (256, 256), (240, 240, 240)))]:
        buf = io.BytesIO()
        img.save(buf, "JPEG")
        out.append((name, buf.getvalue()))
    extra = REPO / "scripts" / "not_leaf_samples"
    if extra.is_dir():
        out += [(f.name, f.read_bytes()) for f in sorted(extra.iterdir())
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return out


def demo_zip_bytes(zip_arg: str | None) -> tuple[str, bytes] | None:
    """The zip to upload: --zip, else demo_upload_batch.zip, else zip the demo_upload_batch/ folder."""
    candidates = [Path(zip_arg)] if zip_arg else [REPO / "demo_upload_batch.zip"]
    for c in candidates:
        if c.is_file():
            return c.name, c.read_bytes()
    folder = REPO / "demo_upload_batch"
    if folder.is_dir():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for f in folder.rglob("*"):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    zf.write(f, f.relative_to(folder))
        return "demo_upload_batch.zip", buf.getvalue()
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000", help="API base URL")
    ap.add_argument("--zip", help="zip with one folder per class (default: demo_upload_batch.zip)")
    ap.add_argument("--skip-retrain", action="store_true", help="do not upload/retrain")
    ap.add_argument("--retrain-timeout", type=int, default=45, help="minutes to wait for retraining")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    print(f"\nTesting {url}\n")

    # ---------------------------------------------------------------- 1. health
    print("1. Health")
    try:
        h = requests.get(f"{url}/health", timeout=30).json()
    except Exception as exc:  # noqa: BLE001
        report(False, "API reachable", f"{type(exc).__name__}: {exc}")
        return 1
    report(True, "API reachable", f"instance {h.get('instance')}, up {h.get('uptime')}")
    if not report(h.get("model_loaded", False), "model loaded",
                  f"version {h.get('model_version')}" if h.get("model_loaded") else str(h.get("startup_error"))):
        return 1
    data = h.get("dataset_on_server") or {}
    n_train = data.get("train", {}).get("total", 0)
    n_test = data.get("test", {}).get("total", 0)
    report(n_train >= 100 and n_test >= 20, "training/test data on the server",
           f"train={n_train}, test={n_test}" + ("" if n_train >= 100 else "  (data/train missing: rebuild the image)"))
    if h.get("retraining") == "running":
        print("  NOTE  a retraining is already running; predictions may be slow until it finishes")

    # ---------------------------------------------------------------- 2. predict
    print("\n2. Predict one image")
    img = find_test_image()
    if img is None:
        report(False, "find a test image", "no images in data/test/<class>/")
    else:
        try:
            t0 = time.time()
            r = requests.post(f"{url}/predict", files={"file": (img.name, img.read_bytes(), "image/jpeg")}, timeout=180)
            ok = r.status_code == 200 and "predicted_class" in r.text
            body = r.json() if ok else r.text[:200]
            report(ok, f"POST /predict ({img.parent.name}/{img.name})",
                   f"{body['predicted_class']} ({body['confidence']:.1%}) in {time.time() - t0:.1f}s, "
                   f"model {body['model_version']}" if ok else f"HTTP {r.status_code}: {body}")
        except Exception as exc:  # noqa: BLE001
            report(False, "POST /predict", f"{type(exc).__name__}: {exc}")
        # A corrupt file must be rejected cleanly with a 400, not crash the API
        r = requests.post(f"{url}/predict", files={"file": ("bad.jpg", b"not an image", "image/jpeg")}, timeout=60)
        report(r.status_code == 400, "corrupt image rejected with HTTP 400", f"got HTTP {r.status_code}")

    # ---------------------------------------------------------------- 2b. not a maize leaf
    print("\n2b. Reject pictures that are not maize leaves")
    lc = requests.get(f"{url}/health", timeout=30).json().get("leaf_check", {})
    if lc.get("state") != "ready":
        print(f"  INFO  leaf check is '{lc.get('state')}' ({lc.get('message')}); skipping. "
              "It calibrates in the background ~1 min after start-up: run the test again shortly.")
    else:
        print(f"  INFO  {lc.get('message')}")
        for name, img_bytes in non_leaf_images():
            r = requests.post(f"{url}/predict", files={"file": (name, img_bytes, "image/jpeg")}, timeout=180)
            detail = r.json().get("detail", {}) if r.headers.get("content-type", "").startswith("application/json") else {}
            if r.status_code == 422 and isinstance(detail, dict):
                report(True, f"{name} rejected as 'not a maize leaf'", detail.get("reason", ""))
            else:
                report(False, f"{name} rejected as 'not a maize leaf'",
                       f"HTTP {r.status_code}, predicted {r.json().get('predicted_class')}" if r.ok else f"HTTP {r.status_code}")

        # Real maize leaves must NOT be rejected: try up to 10 test images per class
        accepted, total, wrongly_rejected = 0, 0, []
        for cls in ("Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"):
            for f in sorted((REPO / "data" / "test" / cls).glob("*.jpg"))[:10]:
                r = requests.post(f"{url}/predict", files={"file": (f.name, f.read_bytes(), "image/jpeg")}, timeout=180)
                total += 1
                if r.status_code == 200:
                    accepted += 1
                elif r.status_code == 422:
                    wrongly_rejected.append(f"{cls}/{f.name}")
        if total:
            report(accepted / total >= 0.9, "real maize test leaves accepted",
                   f"{accepted}/{total}" + (f" (rejected: {', '.join(wrongly_rejected[:5])})" if wrongly_rejected else ""))

    if args.skip_retrain:
        return summary()

    # ---------------------------------------------------------------- 3. upload
    print("\n3. Upload a zip of labelled images")
    z = demo_zip_bytes(args.zip)
    if z is None:
        report(False, "find demo_upload_batch.zip", "pass --zip path/to/file.zip")
        return summary()
    before_version = h.get("model_version")
    r = requests.post(f"{url}/upload-data", files=[("files", (z[0], z[1], "application/zip"))],
                      data={"auto_retrain": "true"}, timeout=300)
    if not report(r.status_code == 200, f"POST /upload-data ({z[0]}, {len(z[1]) / 1e6:.1f} MB)",
                  "" if r.status_code == 200 else f"HTTP {r.status_code}: {r.text[:200]}"):
        return summary()
    up = r.json()
    report(up["saved"] > 0 or up["duplicates_skipped"] > 0, "images saved",
           f"saved {up['saved']} {up.get('saved_per_class', {})}, duplicates {up['duplicates_skipped']}, "
           f"rejected {len(up['rejected'])}")
    trig = up.get("retrain_trigger", {})
    print(f"  INFO  trigger: {trig.get('reason')}")
    batch_id = up.get("batch_id")
    db_rows = requests.get(f"{url}/uploads", params={"limit": 1000}, timeout=60).json()["uploads"]
    batch_rows = [r for r in db_rows if r["batch_id"] == batch_id]
    report(bool(batch_id) and len(batch_rows) == up["saved"] + up["duplicates_skipped"] + len(up["rejected"]),
           "every uploaded file recorded in the database",
           f"batch {batch_id}: {len(batch_rows)} rows "
           f"({', '.join(sorted({r['status'] for r in batch_rows}))})")

    # ---------------------------------------------------------------- 4. retrain
    print("\n4. Retraining")
    if not trig.get("started"):
        r = requests.post(f"{url}/retrain", timeout=60)
        if r.status_code == 409:
            print("  INFO  a retraining is already running; waiting for it")
        else:
            report(r.status_code == 200, "POST /retrain (manual trigger)",
                   "" if r.status_code == 200 else f"HTTP {r.status_code}: {r.text[:200]}")
    deadline = time.time() + args.retrain_timeout * 60
    last_msg, status = "", {}
    while time.time() < deadline:
        try:
            status = requests.get(f"{url}/retrain-status", timeout=30).json()
        except Exception as exc:  # noqa: BLE001
            print(f"  WAIT  status not reachable ({type(exc).__name__}); container may be restarting")
            time.sleep(15)
            continue
        if status.get("message") != last_msg:
            last_msg = status.get("message", "")
            print(f"  ....  [{status.get('state')}] {last_msg}")
        if status.get("state") in ("completed", "rejected", "failed"):
            break
        time.sleep(10)
    state = status.get("state")
    if state == "running":
        report(False, "retraining finished in time", f"still running after {args.retrain_timeout} min")
    elif state == "failed":
        report(False, "retraining", status.get("message", ""))
    else:
        res = status.get("result") or {}
        report(True, f"retraining {state}",
               f"{res.get('from_version')} -> {res.get('to_version')}, test accuracy "
               f"{res.get('previous_accuracy')} -> {res.get('metrics', {}).get('accuracy')}")
        if state == "rejected":
            print("  INFO  'rejected' is a valid outcome: the new model scored >2 points worse, so the old one was kept")

    # ---------------------------------------------------------------- 4b. database after retraining
    if state in ("completed", "rejected"):
        rows = [r for r in requests.get(f"{url}/uploads", params={"limit": 1000}, timeout=60).json()["uploads"]
                if r["batch_id"] == batch_id and r["status"] != "duplicate" and r["status"] != "rejected"]
        in_train = [r for r in rows if r["status"] == "in_training_set"]
        report(len(rows) > 0 and len(in_train) == len(rows), "database: uploaded images marked 'in_training_set'",
               f"{len(in_train)}/{len(rows)}, learned by model {in_train[0]['trained_into_version'] if in_train else '-'}")
        runs = requests.get(f"{url}/retrain-runs", timeout=60).json()["runs"]
        report(bool(runs) and runs[0]["status"] == state, "database: retraining run recorded",
               f"run #{runs[0]['id']}: {runs[0]['from_version']} -> {runs[0]['to_version']} ({runs[0]['status']})"
               if runs else "no runs")

        # Uploading the same zip again must be recognised as duplicates (database fingerprint check)
        r = requests.post(f"{url}/upload-data", files=[("files", (z[0], z[1], "application/zip"))],
                          data={"auto_retrain": "false"}, timeout=300).json()
        report(r["saved"] == 0 and r["duplicates_skipped"] > 0, "re-uploading the same zip detected as duplicates",
               f"saved {r['saved']}, duplicates {r['duplicates_skipped']}")

    # ---------------------------------------------------------------- 5. predict again
    print("\n5. Predict with the model that is live now")
    if img is not None:
        r = requests.post(f"{url}/predict", files={"file": (img.name, img.read_bytes(), "image/jpeg")}, timeout=180)
        ok = r.status_code == 200
        version = r.json().get("model_version") if ok else None
        report(ok, "POST /predict after retraining",
               f"model {version} (was {before_version})" if ok else f"HTTP {r.status_code}: {r.text[:200]}")
    return summary()


def summary() -> int:
    print(f"\nResult: {RESULTS['pass']} passed, {RESULTS['fail']} failed\n")
    return 0 if RESULTS["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
