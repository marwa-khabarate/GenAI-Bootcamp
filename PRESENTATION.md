# Pharma Research Assistant — 10-minute demo

## 0:00–1:00 — Problem

Clinicians and researchers often need to answer questions that span several medication labels: contraindications, interactions, monitoring, and renal thresholds. A generic answer is unsafe if it cannot show the evidence that supports each claim.

This project is a technical demonstration over a fixed local pharmacology corpus. It is not a clinical decision tool.

## 1:00–2:00 — Product demo

1. Open the web application at \`/\`.
2. Ask: “A patient currently on warfarin is prescribed sertraline. What is the risk of bleeding and what monitoring is recommended?”
3. Point out the response, the disclaimer, and expandable line-level evidence.
4. Show the agent trace below the answer.

## 2:00–4:00 — Architecture

\`\`\`text
Question
  ↓
LLM planner (or explicit local fallback)
  ↓ chooses local tools
Similarity search → evidence reader → grounded writer → independent critic
  ↓
Cited answer + evidence + trace
\`\`\`

The browser calls \`POST /ask\`. The FastAPI backend loads the Markdown corpus into memory at startup. There is no external vector database and no persistence layer.

## 4:00–6:00 — Why these GenAI choices

- The project uses the supplied OpenAI-compatible endpoint for planning, answer writing, and a separate critic.
- The planner selects from four explicit local tools; tools are not hidden inside a monolithic prompt.
- The writer is constrained to retrieved excerpts and a fixed citation format.
- The critic checks citation membership and the medical disclaimer after drafting.
- \`PHARMA_OFFLINE=1\` gives a reproducible, no-network fallback for demos and CI.

## 6:00–7:30 — Retrieval and grounding

Show \`app/retrieval.py\`.

- Markdown is segmented by headings and line ranges.
- Search uses transparent in-memory cosine similarity.
- Matching a drug in a source filename and matching a clinically meaningful section heading are weighted, so “warfarin contraindications” retrieves the relevant DailyMed section rather than an unrelated short page.
- Each cited excerpt retains source, heading, and original line range.

## 7:30–9:00 — Challenges and solutions

| Challenge | Solution |
| --- | --- |
| Compound multi-drug questions | Preserve every drug named in the original question during retrieval. |
| Hallucinated or ambiguous clinical claims | Evidence-only writer prompt, citation validation, critic node, and a visible caution state. |
| Endpoint unavailable during a demo | Local retrieval/citation fallback remains operational and clearly labels its mode. |
| Small corpus but need inspectability | Use a zero-infrastructure in-memory index instead of adding an opaque vector database. |

## 9:00–10:00 — Close

Run the provided multi-hop evaluation questions, open the source evidence, and show:

\`\`\`bash
docker build -t agent .
docker run --rm -p 8000:8000 --env-file .env agent
curl http://localhost:8000/health
\`\`\`

Conclude with the key trade-off: this design favours inspectable grounding and reliable local operation over hidden complexity.

