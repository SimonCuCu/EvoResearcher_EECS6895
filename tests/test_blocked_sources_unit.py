from pathlib import Path

from evoresearcher.agents.research_agent import (
    ResearchAgent,
    _blocked_title_markers_from_text,
    _blocked_url_keys_from_text,
    _source_matches_blocked_rule,
)
from evoresearcher.config import AppConfig
from evoresearcher.memory.store import JSONMemoryStore
from evoresearcher.schemas import ConstraintProfile, ResearchBrief, SourceNote


class SearchOnlyLLM:
    def structured(self, model, *, label, system_prompt, user_prompt, temperature=0.2, model_override=None):
        assert label == "research_search_plan"
        assert model_override == "deepseek-v4-pro"
        return model(queries=["south asia social protection failures"])


class FakeWeb:
    def __init__(self):
        self.enriched_urls = []

    def search(self, query, limit=3):
        return [
            SourceNote(
                title="South Asia's unprotected poor: A systematic review",
                url="https://journals.plos.org/plosglobalpublichealth/article?id=10.1371/journal.pgph.0002710",
                snippet="Blocked source.",
            ),
            SourceNote(
                title="Allowed social protection evaluation",
                url="https://example.org/social-protection-evaluation",
                snippet="Allowed source.",
            ),
        ]

    def enrich(self, source, char_limit=1600):
        self.enriched_urls.append(source.url)
        return source.model_copy(update={"excerpt": "Allowed excerpt."})


def build_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        workspace_dir=tmp_path,
        outputs_dir=tmp_path / "outputs",
        memory_dir=tmp_path / "memory",
        author_line="Test",
        deepseek_api_key="test",
        deepseek_model="test-model",
        deepseek_base_url="https://example.com",
        deepseek_reasoning_model="deepseek-v4-pro",
        search_enabled=True,
        tree_depth=1,
        branching_factor=2,
    )


def blocked_prompt() -> str:
    return (
        "You are not allowed to view or cite South Asia's unprotected poor: A systematic review. "
        "Blocked URL: https://journals.plos.org/globalpublichealth/article?id=10.1371/journal.pgph.0002710"
    )


def test_blocked_source_rule_matches_title_and_doi_url():
    blocked_url_keys = _blocked_url_keys_from_text(blocked_prompt())
    blocked_title_markers = _blocked_title_markers_from_text(blocked_prompt())
    source = SourceNote(
        title="South Asia's unprotected poor: A systematic review",
        url="https://journals.plos.org/plosglobalpublichealth/article?id=10.1371/journal.pgph.0002710",
        snippet="",
    )

    assert _source_matches_blocked_rule(
        source,
        blocked_url_keys=blocked_url_keys,
        blocked_title_markers=blocked_title_markers,
    )


def test_collect_sources_skips_blocked_search_results(tmp_path: Path):
    config = build_config(tmp_path)
    config.ensure_directories()
    brief = ResearchBrief(
        mode="general",
        user_goal=blocked_prompt(),
        reframed_goal="Analyze South Asian social protection program failures.",
        scope="South Asia",
        deliverable="report",
        time_cutoff="as of September 2023",
        constraints=ConstraintProfile(),
    )
    agent = ResearchAgent(
        config,
        SearchOnlyLLM(),
        JSONMemoryStore(config.memory_dir / "ideation.json"),
        JSONMemoryStore(config.memory_dir / "proposal.json"),
    )
    agent.web = FakeWeb()

    sources = agent._collect_sources(brief=brief)

    assert [source.title for source in sources] == ["Allowed social protection evaluation"]
    assert agent.web.enriched_urls == ["https://example.org/social-protection-evaluation"]


def test_fallback_queries_include_short_program_specific_searches(tmp_path: Path):
    config = build_config(tmp_path)
    agent = ResearchAgent(
        config,
        SearchOnlyLLM(),
        JSONMemoryStore(config.memory_dir / "ideation.json"),
        JSONMemoryStore(config.memory_dir / "proposal.json"),
    )
    brief = ResearchBrief(
        mode="general",
        user_goal="Study MGNREGA, Pakistan Zakat, BISP, BRAC CFPR/TUP, and Nepal.",
        reframed_goal="Study South Asian social protection failures.",
        scope="South Asia",
        deliverable="report",
        time_cutoff="as of September 2023",
        constraints=ConstraintProfile(),
    )

    queries = agent._fallback_search_queries(brief)

    assert "MGNREGA failures India" in queries
    assert "Pakistan Zakat programme targeting poor" in queries
    assert "BISP targeting errors Pakistan" in queries
    assert "BRAC ultra poor programme evaluation" in queries
    assert "Nepal social security allowances exclusion poor" in queries
