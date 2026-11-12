from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from app.config import get_settings


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")
STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "being", "by", "for", "from", "how", "in",
    "is", "of", "on", "or", "the", "to", "what", "which", "with", "would", "patient",
}
SECTION_CUES = {"contraindication", "contraindications", "interaction", "interactions", "warning", "warnings", "dosage", "renal", "mechanism"}


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    heading: str
    start_line: int
    end_line: int
    text: str

    @property
    def label(self) -> str:
        return f"{self.source} lines {self.start_line}-{self.end_line}"


class CorpusIndex:
    """Small, transparent in-memory vector index over Markdown sections."""

    def __init__(self, corpus_dir: Path) -> None:
        self.chunks = self._load(corpus_dir)
        self._vectors = [Counter(tokens(chunk.text + " " + chunk.heading)) for chunk in self.chunks]
        self._source_terms = [set(tokens(chunk.source)) for chunk in self.chunks]
        self._heading_terms = [set(tokens(chunk.heading)) for chunk in self.chunks]
        self._norms = [self._norm(vector) for vector in self._vectors]
        self._embedding_vectors: list[list[float]] | None = None
        self._embedding_client: OpenAI | None = None
        self._embedding_attempted = False
        self.embedding_mode = "pending"

    @staticmethod
    def _norm(vector: Counter[str]) -> float:
        return math.sqrt(sum(value * value for value in vector.values())) or 1.0

    @staticmethod
    def _load(corpus_dir: Path) -> list[Chunk]:
        chunks: list[Chunk] = []
        for path in sorted(corpus_dir.glob("*.md")):
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            sections: list[tuple[str, int, list[str]]] = []
            heading = "Introduction"
            start = 1
            body: list[str] = []
            for line_no, line in enumerate(lines, start=1):
                match = HEADER_RE.match(line)
                if match:
                    if body:
                        sections.append((heading, start, body))
                    heading = match.group(2).strip()
                    start = line_no
                    body = [line]
                else:
                    body.append(line)
            if body:
                sections.append((heading, start, body))

            for section_number, (title, section_start, section_lines) in enumerate(sections):
                window, overlap = 42, 8
                for offset in range(0, len(section_lines), window - overlap):
                    part = section_lines[offset : offset + window]
                    if not part:
                        continue
                    chunks.append(
                        Chunk(
                            id=f"{path.stem}:{section_number}:{offset}",
                            source=path.name,
                            heading=title,
                            start_line=section_start + offset,
                            end_line=section_start + offset + len(part) - 1,
                            text="\n".join(part).strip(),
                        )
                    )
                    if offset + window >= len(section_lines):
                        break
        if not chunks:
            raise RuntimeError(f"No Markdown corpus found in {corpus_dir}")
        return chunks

    def search(self, query: str, limit: int = 8) -> list[tuple[Chunk, float]]:
        query_embedding = self._query_embedding(query)
        query_vector = Counter(tokens(query))
        query_norm = self._norm(query_vector)
        query_terms = set(query_vector)
        scored: list[tuple[Chunk, float]] = []
        embeddings = self._embedding_vectors or []
        for index, (chunk, vector, source_terms, heading_terms, norm) in enumerate(zip(
            self.chunks, self._vectors, self._source_terms, self._heading_terms, self._norms
        )):
            cosine = sum(query_vector[term] * vector[term] for term in query_terms) / (query_norm * norm)
            # A direct generic-drug match in a filename is more specific than a common
            # label word such as "contraindications" appearing across every document.
            source_boost = 2.0 * len(query_terms.intersection(source_terms))
            # Label section headings express high-value clinical intents (for example,
            # "Contraindications" or "Drug Interactions") much more precisely than prose.
            heading_boost = 4.0 * len(query_terms.intersection(heading_terms))
            lexical_score = cosine + source_boost + heading_boost
            if query_embedding and embeddings:
                semantic_score = self._dense_cosine(query_embedding, embeddings[index])
                # Dense similarity supplies semantic recall; section/file boosts retain
                # precision for named medicines and regulatory label sections.
                score = 0.75 * semantic_score + 0.25 * lexical_score
            else:
                score = lexical_score
            scored.append((chunk, score))
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)

        # For an explicit, named-label question, ensure the matching DailyMed section
        # is represented. Dense retrieval remains the ranking engine, while this guard
        # prevents a tangentially similar interaction section from displacing a formal
        # "Contraindications" section needed for a regulated-label comparison.
        requested_sections = query_terms.intersection(SECTION_CUES)
        named_drugs = {
            term
            for term in query_terms
            if any(term in source_terms and "dailymed" in source_terms for source_terms in self._source_terms)
        }
        required: list[tuple[Chunk, float]] = []
        if requested_sections and named_drugs:
            for drug in sorted(named_drugs):
                candidate = next(
                    (
                        item
                        for item in ranked
                        if "dailymed" in set(tokens(item[0].source))
                        and drug in set(tokens(item[0].source))
                        and requested_sections.intersection(set(tokens(item[0].heading)))
                    ),
                    None,
                )
                if candidate:
                    required.append(candidate)
        required_ids = {chunk.id for chunk, _ in required}
        return (required + [item for item in ranked if item[0].id not in required_ids])[:limit]

    @staticmethod
    def _dense_cosine(left: list[float], right: list[float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
        right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

    def _query_embedding(self, query: str) -> list[float] | None:
        if not self._ensure_embeddings() or not self._embedding_client:
            return None
        try:
            response = self._embedding_client.embeddings.create(
                model=get_settings().embedding_model,
                input=[query],
            )
            return list(response.data[0].embedding)
        except Exception:
            # A transient embedding outage must not make the research API unavailable.
            self.embedding_mode = "lexical-fallback"
            return None

    def _ensure_embeddings(self) -> bool:
        if self._embedding_attempted:
            return self._embedding_vectors is not None
        self._embedding_attempted = True
        settings = get_settings()
        if settings.offline or not settings.api_key:
            self.embedding_mode = "lexical-fallback"
            return False
        try:
            self._embedding_client = OpenAI(api_key=settings.api_key, base_url=settings.base_url or None, timeout=30.0, max_retries=1)
            vectors: list[list[float]] = []
            batch_size = 64
            # A few source labels contain exceptionally long unwrapped patient-guide
            # lines. Keep embedding requests bounded without changing the evidence
            # preserved for citations and the writer.
            texts = [f"{chunk.source}\n{chunk.heading}\n{chunk.text[:12_000]}" for chunk in self.chunks]
            for offset in range(0, len(texts), batch_size):
                response = self._embedding_client.embeddings.create(
                    model=settings.embedding_model,
                    input=texts[offset : offset + batch_size],
                )
                vectors.extend(list(item.embedding) for item in response.data)
            if len(vectors) != len(self.chunks):
                raise RuntimeError("Embedding response count did not match corpus chunks")
            self._embedding_vectors = vectors
            self.embedding_mode = "hybrid-embeddings"
            return True
        except Exception:
            self._embedding_client = None
            self.embedding_mode = "lexical-fallback"
            return False

    def by_ids(self, ids: list[str]) -> list[Chunk]:
        wanted = set(ids)
        return [chunk for chunk in self.chunks if chunk.id in wanted]

    def source_overview(self, query: str, limit: int = 6) -> list[str]:
        sources: list[str] = []
        for chunk, _ in self.search(query, limit=limit * 3):
            if chunk.source not in sources:
                sources.append(chunk.source)
            if len(sources) >= limit:
                break
        return sources

