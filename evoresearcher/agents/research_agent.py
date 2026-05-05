"""Research agent with memory retrieval, review-guided tree search, and Elo ranking."""

from __future__ import annotations

from itertools import count

from typing import Literal

from pydantic import BaseModel, Field

from evoresearcher.config import AppConfig
from evoresearcher.llm import LLMClient
from evoresearcher.memory.store import JSONMemoryStore
from evoresearcher.research.elo_tournament import run_elo_tournament
from evoresearcher.research.tree_search import build_tree
from evoresearcher.retrieval.search import WebResearcher
from evoresearcher.schemas import EloMatch, EvidenceSynthesis, ResearchBrief, ResearchIdea, SourceNote


class SearchPlan(BaseModel):
    queries: list[str] = Field(default_factory=list)


class RootIdea(BaseModel):
    title: str
    summary: str
    method_outline: str
    evidence_use: str
    risks: list[str] = Field(default_factory=list)


class IdeaReview(BaseModel):
    novelty: float
    feasibility: float
    relevance: float
    clarity: float
    weakest_dimension: Literal["novelty", "feasibility", "relevance", "clarity"]
    feedback: str


class ChildIdea(BaseModel):
    title: str
    summary: str
    method_outline: str
    evidence_use: str
    risks: list[str] = Field(default_factory=list)
    relation_to_parent: Literal["refine_weak_dimension", "alternative_direction"]


class IdeaExpansion(BaseModel):
    refine_weak_dimension_child: ChildIdea
    alternative_direction_child: ChildIdea


class PairwiseJudgement(BaseModel):
    winner_id: str
    rationale: str


class ResearchRunResult(BaseModel):
    sources: list[SourceNote]
    idea_tree: list[ResearchIdea]
    leaf_ideas: list[ResearchIdea]
    ranked_ideas: list[ResearchIdea]
    elo_matches: list[EloMatch]
    evidence: EvidenceSynthesis
    memory_context: dict


class ResearchAgent:
    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        ideation_memory: JSONMemoryStore,
        proposal_memory: JSONMemoryStore,
    ):
        self.config = config
        self.llm = llm
        self.ideation_memory = ideation_memory
        self.proposal_memory = proposal_memory
        self.web = WebResearcher()
        self._counter = count(1)

    def run(self, brief: ResearchBrief, observer=None) -> ResearchRunResult:
        memory_hits = self.ideation_memory.query(brief.reframed_goal, top_k=3)
        proposal_hits = self.proposal_memory.query(brief.reframed_goal, top_k=3)
        if observer is not None:
            observer.phase_log(
                "research",
                "Retrieved "
                f"{len(memory_hits)} ideation memories ({self.ideation_memory.last_query_backend}) "
                f"and {len(proposal_hits)} proposal memories ({self.proposal_memory.last_query_backend}).",
            )
            observer.metric("memory_hits", len(memory_hits) + len(proposal_hits))
            if hasattr(observer, "memories_ready"):
                observer.memories_ready(
                    memory_hits=memory_hits,
                    proposal_hits=proposal_hits,
                    ideation_backend=self.ideation_memory.last_query_backend,
                    proposal_backend=self.proposal_memory.last_query_backend,
                )
        sources = self._collect_sources(brief=brief, observer=observer)
        idea_tree, leaf_ideas = self._grow_tree(
            brief=brief,
            sources=sources,
            memory_hits=memory_hits,
            proposal_hits=proposal_hits,
            observer=observer,
        )
        if observer is not None:
            observer.set_phase("ranking", "running elo tournament over leaf proposals")
            observer.metric("tree_nodes", len(idea_tree))
            observer.metric("leaf_nodes", len(leaf_ideas))
        ranked_ideas, elo_matches = run_elo_tournament(
            leaf_ideas,
            judge_fn=lambda idea_a, idea_b: self._judge_pair(
                brief=brief,
                idea_a=idea_a,
                idea_b=idea_b,
                sources=sources,
            ),
        )
        if observer is not None:
            observer.metric("elo_matches", len(elo_matches))
            if ranked_ideas:
                observer.metric("top_elo", f"{ranked_ideas[0].elo_rating:.2f}")
        evidence = self._synthesize_evidence(brief=brief, sources=sources, observer=observer)
        return ResearchRunResult(
            sources=sources,
            idea_tree=idea_tree,
            leaf_ideas=leaf_ideas,
            ranked_ideas=ranked_ideas,
            elo_matches=elo_matches,
            evidence=evidence,
            memory_context={
                "ideation_hits": [entry.model_dump() for entry in memory_hits],
                "proposal_hits": [entry.model_dump() for entry in proposal_hits],
                "ideation_backend": self.ideation_memory.last_query_backend,
                "proposal_backend": self.proposal_memory.last_query_backend,
            },
        )

    def _collect_sources(self, *, brief: ResearchBrief, observer=None) -> list[SourceNote]:
        if not self.config.search_enabled:
            return []
        search_plan = self.llm.structured(
            SearchPlan,
            label="research_search_plan",
            system_prompt=(
                "You are a research planning agent. Produce concise web search queries that maximize "
                "coverage and grounding for a deep research task."
            ),
            user_prompt=(
                f"Goal: {brief.reframed_goal}\n"
                f"Mode: {brief.mode}\n"
                f"Time cutoff: {brief.time_cutoff}\n"
                f"Key questions: {brief.key_questions}"
            ),
        )
        gathered: list[SourceNote] = []
        for query in search_plan.queries[:3]:
            if observer is not None:
                observer.phase_log("research", f"Searching web for: {query}")
            for result in self.web.search(query, limit=2):
                if result.url not in {item.url for item in gathered}:
                    gathered.append(self.web.enrich(result))
                if len(gathered) >= self.config.max_sources:
                    return gathered
        return gathered

    def _grow_tree(
        self,
        *,
        brief: ResearchBrief,
        sources: list[SourceNote],
        memory_hits,
        proposal_hits,
        observer=None,
    ) -> tuple[list[ResearchIdea], list[ResearchIdea]]:
        root_idea = self._build_root_idea(
            brief=brief,
            sources=sources,
            memory_hits=memory_hits,
            proposal_hits=proposal_hits,
        )
        self._score_idea(
            brief=brief,
            idea=root_idea,
            sources=sources,
            memory_hits=memory_hits,
            proposal_hits=proposal_hits,
        )
        if observer is not None and hasattr(observer, "candidate_ideas_ready"):
            observer.candidate_ideas_ready([root_idea], stage="initial candidate")
        nodes, leaves = build_tree(
            [root_idea],
            depth=self.config.tree_depth,
            expand_fn=lambda node, current_depth: self._expand_node(
                brief=brief,
                sources=sources,
                node=node,
                depth=current_depth,
                memory_hits=memory_hits,
                proposal_hits=proposal_hits,
            ),
        )
        for depth in range(1, self.config.tree_depth + 1):
            layer_count = sum(1 for node in nodes if node.depth == depth)
            if observer is not None:
                observer.phase_log("research", f"Expanded tree depth {depth} with {layer_count} nodes.")
                if hasattr(observer, "candidate_ideas_ready"):
                    observer.candidate_ideas_ready(
                        [node for node in nodes if node.depth == depth],
                        stage=f"depth {depth} candidates",
                    )
        return nodes, leaves

    def _build_root_idea(
        self,
        *,
        brief: ResearchBrief,
        sources: list[SourceNote],
        memory_hits,
        proposal_hits,
    ) -> ResearchIdea:
        root = self.llm.structured(
            RootIdea,
            label="research_root_idea",
            system_prompt=(
                "You are the research ideation agent for EvoResearcher. Build exactly one strong initial idea "
                "from the user goal, retrieved memory, and gathered evidence."
            ),
            user_prompt=(
                f"Brief: {brief.model_dump_json(indent=2)}\n"
                f"Ideation memory hits: {[entry.model_dump() for entry in memory_hits]}\n"
                f"Proposal memory hits: {[entry.model_dump() for entry in proposal_hits]}\n"
                f"Sources: {[source.model_dump() for source in sources[:4]]}\n"
                "Return one initial research idea draft."
            ),
        )
        return self._make_idea(
            idea_id=f"idea-{next(self._counter)}",
            depth=0,
            parent_id=None,
            relation_to_parent="root",
            title=root.title,
            summary=root.summary,
            method_outline=root.method_outline,
            evidence_use=root.evidence_use,
            risks=root.risks,
        )

    def _expand_node(
        self,
        *,
        brief: ResearchBrief,
        sources: list[SourceNote],
        node: ResearchIdea,
        depth: int,
        memory_hits,
        proposal_hits,
    ) -> list[ResearchIdea]:
        children = self._expand_from_review(
            brief=brief,
            sources=sources,
            parent=node,
            depth=depth,
            memory_hits=memory_hits,
            proposal_hits=proposal_hits,
        )
        scored_children: list[ResearchIdea] = []
        for child in children[: max(1, min(self.config.branching_factor, 2))]:
            self._score_idea(
                brief=brief,
                idea=child,
                sources=sources,
                memory_hits=memory_hits,
                proposal_hits=proposal_hits,
            )
            scored_children.append(child)
        return scored_children

    def _expand_from_review(
        self,
        *,
        brief: ResearchBrief,
        sources: list[SourceNote],
        parent: ResearchIdea,
        depth: int,
        memory_hits,
        proposal_hits,
    ) -> list[ResearchIdea]:
        expansion = self.llm.structured(
            IdeaExpansion,
            label=f"research_expansion_depth_{depth}_{parent.idea_id}",
            system_prompt=(
                "You are the review-guided tree search expander for EvoResearcher. "
                "Use the parent's review feedback to produce exactly two children: "
                "one child that directly fixes the weakest dimension, and one child that pursues an "
                "alternative but still aligned direction."
            ),
            user_prompt=(
                f"Brief: {brief.model_dump_json(indent=2)}\n"
                f"Parent idea: {parent.model_dump_json(indent=2)}\n"
                f"Parent weakest dimension: {parent.weakest_dimension}\n"
                f"Parent review feedback: {parent.review_feedback}\n"
                f"Ideation memory hits: {[entry.model_dump() for entry in memory_hits]}\n"
                f"Proposal memory hits: {[entry.model_dump() for entry in proposal_hits]}\n"
                f"Sources: {[source.model_dump() for source in sources[:4]]}\n"
                "Return two children. The first must refine the weak dimension. "
                "The second must explore a nearby alternative direction."
            ),
        )
        refine_child = self._make_idea(
            idea_id=f"idea-{next(self._counter)}",
            depth=depth,
            parent_id=parent.idea_id,
            relation_to_parent="refine_weak_dimension",
            title=expansion.refine_weak_dimension_child.title,
            summary=expansion.refine_weak_dimension_child.summary,
            method_outline=expansion.refine_weak_dimension_child.method_outline,
            evidence_use=expansion.refine_weak_dimension_child.evidence_use,
            risks=expansion.refine_weak_dimension_child.risks,
        )
        alternative_child = self._make_idea(
            idea_id=f"idea-{next(self._counter)}",
            depth=depth,
            parent_id=parent.idea_id,
            relation_to_parent="alternative_direction",
            title=expansion.alternative_direction_child.title,
            summary=expansion.alternative_direction_child.summary,
            method_outline=expansion.alternative_direction_child.method_outline,
            evidence_use=expansion.alternative_direction_child.evidence_use,
            risks=expansion.alternative_direction_child.risks,
        )
        return [refine_child, alternative_child]

    def _score_idea(
        self,
        *,
        brief: ResearchBrief,
        idea: ResearchIdea,
        sources: list[SourceNote],
        memory_hits,
        proposal_hits,
    ) -> None:
        review = self.llm.structured(
            IdeaReview,
            label=f"research_review_{idea.idea_id}",
            system_prompt=(
                "Review a research proposal idea. Score novelty, feasibility, relevance, and clarity from 1 to 10. "
                "Also identify the single weakest dimension and provide feedback that can guide refinement."
            ),
            user_prompt=(
                f"Brief: {brief.model_dump_json(indent=2)}\n"
                f"Idea: {idea.model_dump_json(indent=2)}\n"
                f"Ideation memory hits: {[entry.model_dump() for entry in memory_hits]}\n"
                f"Proposal memory hits: {[entry.model_dump() for entry in proposal_hits]}\n"
                f"Sources: {[source.model_dump() for source in sources[:4]]}\n"
            ),
        )
        total = round(
            review.novelty * 0.25
            + review.feasibility * 0.25
            + review.relevance * 0.25
            + review.clarity * 0.25,
            2,
        )
        idea.novelty = review.novelty
        idea.feasibility = review.feasibility
        idea.relevance = review.relevance
        idea.clarity = review.clarity
        idea.total_score = total
        idea.review_feedback = review.feedback
        idea.weakest_dimension = review.weakest_dimension

    def _judge_pair(
        self,
        *,
        brief: ResearchBrief,
        idea_a: ResearchIdea,
        idea_b: ResearchIdea,
        sources: list[SourceNote],
    ) -> tuple[str, str]:
        judgement = self.llm.structured(
            PairwiseJudgement,
            label=f"elo_judge_{idea_a.idea_id}_vs_{idea_b.idea_id}",
            system_prompt=(
                "You are an impartial research committee chair. Choose the stronger proposal candidate "
                "for a deep research system and explain the choice briefly."
            ),
            user_prompt=(
                f"Brief: {brief.model_dump_json(indent=2)}\n"
                f"Idea A: {idea_a.model_dump_json(indent=2)}\n"
                f"Idea B: {idea_b.model_dump_json(indent=2)}\n"
                f"Sources: {[source.model_dump() for source in sources[:4]]}\n"
                "Return the winner_id exactly matching one of the input idea ids."
            ),
        )
        if judgement.winner_id not in {idea_a.idea_id, idea_b.idea_id}:
            raise ValueError(
                f"Elo judge returned invalid winner_id '{judgement.winner_id}' for "
                f"{idea_a.idea_id} vs {idea_b.idea_id}."
            )
        return judgement.winner_id, judgement.rationale

    def _synthesize_evidence(self, *, brief: ResearchBrief, sources: list[SourceNote], observer=None) -> EvidenceSynthesis:
        if observer is not None:
            observer.phase_log("research", f"Synthesizing evidence across {len(sources)} sources.")
            observer.metric("sources", len(sources))
        return self.llm.structured(
            EvidenceSynthesis,
            label="research_evidence_synthesis",
            system_prompt=(
                "You are a deep research synthesis agent. Extract high-signal findings, tensions, "
                "and opportunities from the gathered evidence."
            ),
            user_prompt=(
                f"Brief: {brief.model_dump_json(indent=2)}\n"
                f"Sources: {[source.model_dump() for source in sources]}\n"
            ),
        )

    def _make_idea(
        self,
        *,
        idea_id: str,
        depth: int,
        parent_id: str | None,
        relation_to_parent: str,
        title: str,
        summary: str,
        method_outline: str,
        evidence_use: str,
        risks: list[str],
    ) -> ResearchIdea:
        return ResearchIdea(
            idea_id=idea_id,
            title=title,
            summary=summary,
            method_outline=method_outline,
            evidence_use=evidence_use,
            risks=risks,
            parent_id=parent_id,
            depth=depth,
            relation_to_parent=relation_to_parent,
        )
