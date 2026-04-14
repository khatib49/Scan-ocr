"""
Stress test for scan-invoice-api using Locust.

Usage:
  1. Install:   pip install locust
  2. Headless:  locust --headless -u 10 -r 2 --run-time 60s --html report.html
  3. Web UI:    locust  (then open http://localhost:8089)

Environment variables (or edit the constants below):
  STRESS_API_KEY   - your X-API-Key value
  STRESS_BASE_URL  - base URL of the running API (default: http://localhost:8000)
  STRESS_IMAGE_URL - a publicly accessible invoice image URL for /analyze tests
"""

import os
import uuid
import io
from locust import HttpUser, task, between, tag, events

# ---------------------------------------------------------------------------
# Configuration — override with env vars or edit directly
# ---------------------------------------------------------------------------
API_KEY = os.getenv("STRESS_API_KEY")
BASE_URL = os.getenv("STRESS_BASE_URL")

# A real invoice image URL to send to /analyze.
# Replace with a stable URL you control, or leave as None to skip /analyze tasks.
INVOICE_IMAGE_URL = os.getenv("STRESS_IMAGE_URL", None)

# Small sample JPEG bytes (1x1 white pixel) used when uploading a file directly.
_TINY_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
    b"\xc1\xff\xd9"
)

HEADERS = {"X-API-Key": API_KEY}

# ---------------------------------------------------------------------------
# User behaviour
# ---------------------------------------------------------------------------

class InvoiceApiUser(HttpUser):
    """Simulates a typical API client making a mix of requests."""

    host = BASE_URL
    # Wait 0.5–2 s between tasks to simulate realistic pacing.
    # Set to between(0, 0) for pure throughput testing.
    wait_time = between(0.5, 2)

    # ------------------------------------------------------------------
    # Lightweight tasks (weight=5 means called ~5x more often)
    # ------------------------------------------------------------------

    @task(5)
    @tag("health", "light")
    def health_check(self):
        """GET /health — no auth required."""
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(3)
    @tag("projects", "light")
    def get_my_project(self):
        """GET /projects/me — authenticated."""
        with self.client.get(
            "/projects/me", headers=HEADERS, catch_response=True
        ) as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"projects/me returned {resp.status_code}: {resp.text[:120]}")

    @task(2)
    @tag("venue-profiles", "light")
    def list_venue_profiles(self):
        """GET /venue-profiles/all — authenticated."""
        with self.client.get(
            "/venue-profiles/all", headers=HEADERS, catch_response=True
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"venue-profiles/all returned {resp.status_code}")

    # ------------------------------------------------------------------
    # Heavy tasks — invoice analysis (requires a valid image)
    # ------------------------------------------------------------------

    @task(1)
    @tag("analyze", "heavy")
    def analyze_with_url(self):
        """POST /analyze with imageUrl — calls Gemini AI (slow, rate-limited)."""
        if not INVOICE_IMAGE_URL:
            return  # skip if not configured

        data = {
            "userReference": f"stress-user-{uuid.uuid4().hex[:8]}",
            "scanReference": f"stress-scan-{uuid.uuid4().hex[:8]}",
            "imageUrl": INVOICE_IMAGE_URL,
        }
        with self.client.post(
            "/analyze",
            data=data,
            headers=HEADERS,
            catch_response=True,
            timeout=120,
            name="/analyze (url)",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            elif resp.status_code == 429:
                resp.failure("Rate limited (429)")
            else:
                resp.failure(f"/analyze returned {resp.status_code}: {resp.text[:200]}")


class LightLoadUser(InvoiceApiUser):
    """Only runs lightweight tasks — useful to isolate DB / auth layer performance."""

    weight = 3  # 3 of these for every 1 HeavyUser (below)

    @task
    @tag("health", "light")
    def health_only(self):
        self.client.get("/health")

    # Disable heavy tasks for this user type
    analyze_with_url = None  # type: ignore[assignment]


class HeavyUser(InvoiceApiUser):
    """Hammers the /analyze endpoint — use sparingly to avoid Gemini quota burn."""

    weight = 1
    wait_time = between(3, 8)  # slower pace; Gemini has rate limits


# ---------------------------------------------------------------------------
# Optional: print a summary banner when the test finishes
# ---------------------------------------------------------------------------

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    stats = environment.stats
    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    for name, entry in stats.entries.items():
        print(
            f"{entry.method:6} {entry.name:<40} "
            f"reqs={entry.num_requests:>6}  "
            f"fails={entry.num_failures:>4}  "
            f"avg={entry.avg_response_time:>7.1f}ms  "
            f"p95={entry.get_response_time_percentile(0.95):>7.1f}ms"
        )
    print("=" * 60)
