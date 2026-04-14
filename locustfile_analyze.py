"""
Focused stress test for POST /analyze
======================================
Tracks per-request details: response time, status, fraud score,
merchant extracted, Gemini errors, rate limits, and needsRescan flags.

SETUP
-----
1.  pip install locust

2.  Edit the CONFIG block below (API_KEY + IMAGE_URLS are required).

RUN — headless with full HTML report
--------------------------------------
    locust -f locustfile_analyze.py --headless \
        -u 5 -r 1 --run-time 120s \
        --html analyze_report.html \
        --csv  analyze_csv

RUN — interactive Web UI (recommended first)
----------------------------------------------
    locust -f locustfile_analyze.py
    → open http://localhost:8089, set users, start, then Download Report

READING THE REPORT
-------------------
  analyze_report.html        — charts + percentile table
  analyze_csv_stats.csv      — per-endpoint aggregated stats
  analyze_csv_failures.csv   — all failed requests
  analyze_detail.log         — one line per request (written live)
"""

import os
import uuid
import json
import csv
import threading
from datetime import datetime
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner

# =============================================================================
#  CONFIG — edit these before running
# =============================================================================

API_KEY   = os.getenv("STRESS_API_KEY")
BASE_URL  = os.getenv("STRESS_BASE_URL")

# List of invoice image URLs to rotate through.
# Use at least one real invoice image (JPEG/PNG, publicly accessible).
IMAGE_URLS = []
# Skip the screen-photo check to isolate pure Gemini + extraction performance.
SKIP_SCREEN_CHECK = os.getenv("SKIP_SCREEN_CHECK", "false").lower() == "true"

# =============================================================================
#  Per-request detail log (written to analyze_detail.log)
# =============================================================================

_log_lock = threading.Lock()
_LOG_FILE = "analyze_detail.log"

# CSV header written once at startup
_log_initialized = True

def _init_log():
    global _log_initialized
    if _log_initialized:
        return
    with _log_lock:
        if _log_initialized:
            return
        with open(_LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "user_ref", "scan_ref", "image_url",
                "status_code", "response_ms",
                "merchant_name", "total", "fraud_score",
                "needs_rescan", "profile_matched",
                "screen_photo_warning", "gemini_error",
                "rate_limited", "error_detail",
            ])
        _log_initialized = True


def _log_request(row: dict):
    _init_log()
    with _log_lock:
        with open(_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "user_ref", "scan_ref", "image_url",
                "status_code", "response_ms",
                "merchant_name", "total", "fraud_score",
                "needs_rescan", "profile_matched",
                "screen_photo_warning", "gemini_error",
                "rate_limited", "error_detail",
            ])
            writer.writerow(row)


# =============================================================================
#  Counters (thread-safe)
# =============================================================================

class _Counters:
    def __init__(self):
        self._lock = threading.Lock()
        self.total        = 0
        self.success      = 0
        self.failed       = 0
        self.rate_limited = 0
        self.gemini_error = 0
        self.needs_rescan = 0
        self.screen_flag  = 0
        self.total_ms     = 0.0

    def record(self, success, rate_limited, gemini_error,
               needs_rescan, screen_flag, elapsed_ms):
        with self._lock:
            self.total += 1
            self.total_ms += elapsed_ms
            if success:
                self.success += 1
            else:
                self.failed += 1
            if rate_limited:
                self.rate_limited += 1
            if gemini_error:
                self.gemini_error += 1
            if needs_rescan:
                self.needs_rescan += 1
            if screen_flag:
                self.screen_flag += 1

    def summary(self) -> str:
        with self._lock:
            avg = (self.total_ms / self.total) if self.total else 0
            return (
                f"  Total requests : {self.total}\n"
                f"  Success        : {self.success}  ({100*self.success/max(self.total,1):.1f}%)\n"
                f"  Failed         : {self.failed}\n"
                f"  Rate limited   : {self.rate_limited}\n"
                f"  Gemini errors  : {self.gemini_error}\n"
                f"  needsRescan    : {self.needs_rescan}\n"
                f"  Screen flags   : {self.screen_flag}\n"
                f"  Avg resp time  : {avg:.0f} ms\n"
            )

COUNTERS = _Counters()

# =============================================================================
#  Locust User
# =============================================================================

_image_index = 0
_image_lock  = threading.Lock()

def _next_image_url() -> str:
    global _image_index
    with _image_lock:
        url = IMAGE_URLS[_image_index % len(IMAGE_URLS)]
        _image_index += 1
        return url


class AnalyzeUser(HttpUser):
    """
    Sends POST /analyze requests in a loop.
    Adjust -u (users) and wait_time to control concurrency and throughput.
    """

    host = BASE_URL

    # Time between consecutive requests per virtual user.
    # between(0, 0)  → fire as fast as possible (max throughput test)
    # between(2, 5)  → realistic paced test
    wait_time = between(1, 3)

    @task
    def analyze_invoice(self):
        user_ref = f"stress-{uuid.uuid4().hex[:8]}"
        scan_ref = f"scan-{uuid.uuid4().hex[:8]}"
        image_url = _next_image_url()

        payload = {
            "userReference":    user_ref,
            "scanReference":    scan_ref,
            "imageUrl":         image_url,
            "skip_screen_check": str(SKIP_SCREEN_CHECK).lower(),
        }

        log_row = {
            "timestamp":          datetime.utcnow().isoformat(),
            "user_ref":           user_ref,
            "scan_ref":           scan_ref,
            "image_url":          image_url,
            "status_code":        "",
            "response_ms":        "",
            "merchant_name":      "",
            "total":              "",
            "fraud_score":        "",
            "needs_rescan":       "",
            "profile_matched":    "",
            "screen_photo_warning": "",
            "gemini_error":       "",
            "rate_limited":       "",
            "error_detail":       "",
        }

        is_success     = False
        is_rate_limit  = False
        is_gemini_error = False
        is_needs_rescan = False
        is_screen_flag  = False

        with self.client.post(
            "/analyze",
            data=payload,
            headers={"X-API-Key": API_KEY},
            catch_response=True,
            timeout=180,          # Gemini can take up to ~60s under load
            name="/analyze",
        ) as resp:

            log_row["status_code"] = resp.status_code
            log_row["response_ms"] = round(resp.elapsed.total_seconds() * 1000)

            if resp.status_code == 200:
                try:
                    body = resp.json()
                    data = body.get("data", {})

                    merchant     = data.get("MerchantName") or ""
                    total        = data.get("Total")
                    fraud_score  = data.get("fraudScore")
                    needs_rescan = data.get("needsRescan", False)
                    profile_ok   = data.get("profileMatched", False)
                    screen_warn  = data.get("screenPhotoWarning", False)
                    reason       = data.get("reason", "")

                    # Detect Gemini errors embedded in a 200 response
                    gemini_err = (
                        "RATE_LIMIT" in reason.upper()
                        or "gemini" in reason.lower()
                        or fraud_score is None
                    )

                    log_row.update({
                        "merchant_name":       merchant,
                        "total":               total,
                        "fraud_score":         fraud_score,
                        "needs_rescan":        needs_rescan,
                        "profile_matched":     profile_ok,
                        "screen_photo_warning": screen_warn,
                        "gemini_error":        gemini_err,
                        "rate_limited":        False,
                        "error_detail":        reason if gemini_err else "",
                    })

                    is_success      = not gemini_err
                    is_gemini_error = gemini_err
                    is_needs_rescan = needs_rescan
                    is_screen_flag  = screen_warn

                    if gemini_err:
                        resp.failure(f"Gemini error in 200 body: {reason[:120]}")
                    else:
                        resp.success()

                except Exception as parse_err:
                    log_row["error_detail"] = f"parse error: {parse_err}"
                    resp.failure(f"Could not parse JSON: {parse_err}")

            elif resp.status_code == 429:
                is_rate_limit = True
                log_row["rate_limited"] = True
                log_row["error_detail"] = resp.text[:200]
                resp.failure("429 Rate Limited")

            elif resp.status_code == 422:
                log_row["error_detail"] = resp.text[:300]
                resp.failure(f"422 Validation error — check form fields: {resp.text[:200]}")

            elif resp.status_code == 401:
                log_row["error_detail"] = "Invalid API key"
                resp.failure("401 Unauthorized — check API_KEY")

            else:
                log_row["error_detail"] = resp.text[:300]
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:150]}")

        _log_request(log_row)
        COUNTERS.record(
            success=is_success,
            rate_limited=is_rate_limit,
            gemini_error=is_gemini_error,
            needs_rescan=is_needs_rescan,
            screen_flag=is_screen_flag,
            elapsed_ms=log_row["response_ms"] or 0,
        )


# =============================================================================
#  Event hooks — startup banner + final summary
# =============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n" + "=" * 65)
    print("  /analyze STRESS TEST STARTED")
    print("=" * 65)
    print(f"  Target  : {BASE_URL}/analyze")
    print(f"  Images  : {len(IMAGE_URLS)} URL(s) rotating")
    print(f"  Skip screen check : {SKIP_SCREEN_CHECK}")
    print(f"  Detail log: {_LOG_FILE}")
    print("=" * 65 + "\n")
    _init_log()


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    stats = environment.stats
    total_entry = stats.total

    print("\n" + "=" * 65)
    print("  /analyze STRESS TEST — FINAL REPORT")
    print("=" * 65)

    # Locust built-in stats
    print(f"\n  [Locust Stats]")
    print(f"  Requests         : {total_entry.num_requests}")
    print(f"  Failures         : {total_entry.num_failures}  "
          f"({100*total_entry.fail_ratio:.1f}%)")
    print(f"  Avg resp time    : {total_entry.avg_response_time:.0f} ms")
    print(f"  Median resp time : {total_entry.get_response_time_percentile(0.50):.0f} ms")
    print(f"  P95 resp time    : {total_entry.get_response_time_percentile(0.95):.0f} ms")
    print(f"  P99 resp time    : {total_entry.get_response_time_percentile(0.99):.0f} ms")
    print(f"  Max resp time    : {total_entry.max_response_time:.0f} ms")
    print(f"  Req/s (avg)      : {total_entry.total_rps:.2f}")

    # Custom counters
    print(f"\n  [/analyze Detail]")
    print(COUNTERS.summary())

    # Failure breakdown
    if stats.errors:
        print("  [Failure Breakdown]")
        for key, err in stats.errors.items():
            print(f"    {err.occurrences:>5}x  {err.name} — {err.error[:80]}")

    print(f"\n  Detail log saved to : {_LOG_FILE}")
    print("  HTML report         : analyze_report.html  (if --html flag used)")
    print("  CSV stats           : analyze_csv_stats.csv (if --csv flag used)")
    print("=" * 65 + "\n")
