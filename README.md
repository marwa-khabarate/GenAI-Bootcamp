# Pharma Research Assistant

A compact, inspectable agentic RAG demonstration over the supplied pharmacology corpus. It is designed as a technical exercise only and never substitutes for clinical judgement.

## What it demonstrates

- **Plan:** a model-driven planner decomposes the question and selects from a local tool catalogue. The enforced workflow requires at least two tools appropriate to grounded research.
- **Retrieve:** an in-memory cosine-similarity index over Markdown sections retrieves evidence without a vector database or any remote data store.
- **Use tools:** \`semantic_search\`, \`read_evidence\`, \`source_overview\`, and \`compare_evidence\` have explicit input/output boundaries and appear in the UI trace.
- **Review:** a separate critic checks the draft's citation set and medical disclaimer; local checks remain active if the model endpoint is unavailable.
- **Answer:** every response exposes source, section, and line-range evidence alongside an explicit disclaimer.
- **Examples:** the frontend requests four LLM-generated, corpus-aware demo questions; a validated local fallback is used when the endpoint is unavailable.

The OpenAI-compatible model is used for planning, synthesis, and critique. If no reachable model endpoint is configured, the same API remains demonstrable in a transparent local-fallback mode: retrieval, evidence reading, citation validation, and the disclaimer still run.
Set `PHARMA_OFFLINE=1` to force that deterministic local mode during development or testing.

## Architecture

\`\`\`text
Browser → POST /ask → Planner → Local tools → Evidence set → Grounded writer → Critic → cited answer
                         │                │                              │
                         └── in-memory Markdown similarity index ────────┘
\`\`\`

The corpus is bundled in the image. No question or corpus content is persisted, uploaded, or sent to any service except the configured LLM endpoint when LLM-assisted mode is enabled.

## Run locally

Use Python 3.11+.

\`\`\`bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
\`\`\`

Open \`http://localhost:8000\`. API documentation is at \`http://localhost:8000/docs\`.

Copy \`.env.example\` to \`.env\` and add the supplied credentials. The app accepts both \`OPENAI_BASE_URL\` and the \`OPENAI_BASE_UR\` spelling supplied in the original exercise archive. Do not commit \`.env\`.

Quoted values are also supported when the file is passed to Docker with \`--env-file\`.

## Docker

\`\`\`bash
docker build -t agent .
docker run --rm -p 8000:8000 --env-file .env agent
\`\`\`

Verify it with:

\`\`\`bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\\"question\\":\\"What are the main contraindications for warfarin?\\"}"
\`\`\`

## Tests

\`\`\`bash
pytest -q
\`\`\`

## Grounding and observability

- Retrieval builds an in-memory hybrid index on the first research request: OpenAI-compatible embeddings, lexical cosine similarity, explicit medicine-name matching, and section-heading boosts.
- If the embedding endpoint is unavailable, the API transparently uses lexical retrieval and reports the mode through `/health`.
- The interface displays each selected tool, its bounded input summary, result count, elapsed time, citation evidence, and whether the critic requested one revision.
- The critic can request one evidence-bounded rewrite. It never enters an unbounded self-correction loop.

The public evaluation scenarios are covered in `tests/test_evaluations.py`. After a Docker launch, run `python scripts/smoke_test.py`.

