"""Minimal black-box smoke test for a running Docker container."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen


BASE_URL = "http://localhost:8000"


def main() -> None:
    with urlopen(f"{BASE_URL}/health", timeout=15) as response:
        health = json.load(response)
    assert health["status"] == "ok"

    request = Request(
        f"{BASE_URL}/ask",
        data=json.dumps({"question": "What are the main contraindications for warfarin?"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        answer = json.load(response)
    assert answer["citations"]
    assert "not medical advice" in answer["answer"].lower()
    print("Smoke test passed:", health)


if __name__ == "__main__":
    main()

