from fastapi.testclient import TestClient

from app.main import app


PUBLIC_EVALUATIONS = [
    ("What are the main contraindications for warfarin?", {"dailymed-warfarin.md"}),
    ("Compare the contraindications of warfarin and apixaban. What conditions are shared and which are unique to each drug?", {"dailymed-warfarin.md", "dailymed-apixaban.md"}),
    ("A patient currently on warfarin is prescribed sertraline. What is the risk of bleeding and what monitoring is recommended?", {"dailymed-warfarin.md", "dailymed-sertraline.md"}),
    ("A patient with an eGFR of 25 mL/min is being considered for metformin. Is metformin contraindicated at this level of renal function, and what is the threshold?", {"dailymed-metformin.md"}),
]


def test_public_evaluations_are_grounded(monkeypatch) -> None:
    monkeypatch.setenv("PHARMA_OFFLINE", "1")
    with TestClient(app) as client:
        for question, expected_sources in PUBLIC_EVALUATIONS:
            payload = client.post("/ask", json={"question": question}).json()
            cited_sources = {citation["source"] for citation in payload["citations"]}
            assert expected_sources.issubset(cited_sources)
            assert "not medical advice" in payload["answer"].lower()
            assert {event["tool"] for event in payload["tools_used"]} >= {"semantic_search", "read_evidence"}
