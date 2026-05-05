"""Shared schemas."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


ModeName = Literal["general", "ml"]


class ConstraintProfile(BaseModel):
    model_scale: str = ""
    dataset_scope: str = ""
    compute_budget: str = ""
    time_budget: str = ""
    extra_notes: str = ""


class ResearchBrief(BaseModel):
    mode: ModeName
    user_goal: str
    reframed_goal: str
    scope: str
    deliverable: str
    time_cutoff: str
    key_questions: list[str] = Field(default_factory=list)
    constraints: ConstraintProfile = Field(default_factory=ConstraintProfile)
    human_clarifications: list[str] = Field(default_factory=list)
    preferred_idea_id: str = ""
    human_idea_feedback: str = ""
    report_emphasis: str = ""


class SourceNote(BaseModel):
    title: str
    url: str
    snippet: str
    excerpt: str = ""


class ResearchIdea(BaseModel):
    idea_id: str
    title: str
    summary: str
    method_outline: str
    evidence_use: str
    risks: list[str] = Field(default_factory=list)
    novelty: float = 0.0
    feasibility: float = 0.0
    relevance: float = 0.0
    clarity: float = 0.0
    total_score: float = 0.0
    elo_rating: float = 1000.0
    review_feedback: str = ""
    weakest_dimension: str = ""
    relation_to_parent: str = "root"
    parent_id: str | None = None
    depth: int = 0


class EvidenceSynthesis(BaseModel):
    findings: list[str] = Field(default_factory=list)
    tensions: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)


class ReportSections(BaseModel):
    title: str
    abstract: str
    problem_and_goal: str
    evidence_base: str
    proposed_direction: str
    plan_or_analysis: str
    risks_and_limits: str
    conclusion: str
    references: list[str] = Field(default_factory=list)


class MemoryEntry(BaseModel):
    entry_id: str
    kind: str
    summary: str
    goal: str
    details: str
    tags: list[str] = Field(default_factory=list)
    created_at: str


class EloMatch(BaseModel):
    idea_a_id: str
    idea_b_id: str
    winner_id: str
    rationale: str
    rating_a_before: float
    rating_b_before: float
    rating_a_after: float
    rating_b_after: float


class GraphState(TypedDict, total=False):
    run_id: str
    run_dir: str
    mode: ModeName
    goal: str
    brief: dict
    sources: list[dict]
    memory_context: dict
    idea_tree: list[dict]
    elo_matches: list[dict]
    top_ideas: list[dict]
    evidence: dict
    report: dict
    artifacts: dict
    memory_updates: dict
    model_routing: dict
