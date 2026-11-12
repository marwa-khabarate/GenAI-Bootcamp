from fastapi.testclient import TestClient

from app.main import app


def test_health_and_ask(monkeypatch) -> None:
    monkeypatch.setenv("PHARMA_OFFLINE", "1")
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["corpus_chunks"] > 20
        assert health.json()["retrieval_mode"] == "pending"

        examples = client.get("/examples")
        assert examples.status_code == 200
        assert len(examples.json()["examples"]) == 4

        response = client.post("/ask", json={"question": "What are the main contraindications for warfarin?"})
        assert response.status_code == 200
        payload = response.json()
        assert payload["citations"]
        assert {event["tool"] for event in payload["tools_used"]} >= {"semantic_search", "read_evidence"}
        assert "not medical advice" in payload["answer"].lower()
        assert all("elapsed_ms" in event for event in payload["tools_used"])

