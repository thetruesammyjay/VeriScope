"""Verify a running local VeriScope API through its public HTTP contract."""

from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_ARTICLE = (
    "The local health authority reported that three public hospitals remain open "
    "for emergency care, maternity services, and routine treatment throughout "
    "the week for residents in the surrounding communities."
)


def _request(url: str, *, payload: dict[str, str] | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload else None
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{request.method} {url} returned {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach {url}: {error.reason}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--article", default=DEFAULT_ARTICLE)
    parser.add_argument("--allow-unavailable-model", action="store_true")
    args = parser.parse_args()
    api_url = args.api_url.rstrip("/")

    health = _request(f"{api_url}/health")
    if health.get("status") != "ok":
        raise RuntimeError(f"Unexpected health response: {health}")

    analysis = _request(f"{api_url}/api/v1/analyze", payload={"text": args.article})
    prediction = analysis.get("prediction", {})
    verification = analysis.get("verification", {})
    if not isinstance(prediction.get("available"), bool):
        raise RuntimeError(f"Malformed prediction response: {prediction}")
    if not args.allow_unavailable_model and not prediction["available"]:
        raise RuntimeError(f"Model is unavailable: {prediction.get('error', 'unknown error')}")
    if verification.get("status") not in {
        "supported",
        "contradicted",
        "mixed",
        "insufficient",
    }:
        raise RuntimeError(f"Malformed verification response: {verification}")

    print(f"Health: {health['status']} ({health['environment']})")
    print(f"Prediction available: {prediction['available']}")
    print(f"Evidence status: {verification['status']}")
    print("Local API smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
