from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)


class Citation(BaseModel):
    source: str
    heading: str
    lines: str
    excerpt: str


class ToolEvent(BaseModel):
    tool: str
    purpose: str
    result_count: int
    input_summary: str = ""
    elapsed_ms: int = 0


class Review(BaseModel):
    status: Literal["approved", "cautious"]
    notes: list[str]
    revised: bool = False


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    plan: list[str]
    tools_used: list[ToolEvent]
    review: Review
    mode: Literal["llm", "local-fallback"]


class ExampleQuestion(BaseModel):
    label: str
    question: str


class ExamplesResponse(BaseModel):
    examples: list[ExampleQuestion]
    mode: Literal["llm", "local-fallback"]

