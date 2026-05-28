"""
Tests for backend/services/matching_service.py — orchestration + pure helpers.

Does NOT require a running database. Uses MagicMock for the SQLAlchemy Session
and SimpleNamespace for domain objects, consistent with the existing algorithm
test pattern.

Functions tested
----------------
_is_eligible            (6) — no-deadline, expired, future, cost+barrier,
                               cost without financial barrier, None deadline
_build_pathway_summary  (5) — high risk step, mentor step, no-mentor fallback,
                               step count, always returns list
generate_matches        (8) — MatchResult type, TOP_N caps, score ordering,
                               expired opportunity excluded, no opps, no mentors,
                               risk_level string, pathway_summary non-empty

Total: 19 tests, 3 classes.
"""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.models.mentor import Mentor as MentorORM
from backend.models.opportunity import Opportunity as OppORM
from backend.schemas.matching import MatchResult
from backend.services.matching_service import (
    TOP_N_MENTORS,
    TOP_N_OPPORTUNITIES,
    _build_pathway_summary,
    _is_eligible,
    generate_matches,
)


# ─────────────────────────────────────────────────────────────────────────────
# Factories — SimpleNamespace objects satisfy duck-typed algorithm functions
# ─────────────────────────────────────────────────────────────────────────────

def make_user(**kw):
    defaults = dict(
        id=1,
        country="Kenya",
        region="Nairobi",
        language="English",
        interests=["machine learning", "education"],
        skills=["python", "data analysis"],
        barriers=[],
        goals="study CS and build products",
        education_level="undergraduate",
        mobility_intent=0.5,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def make_opp(opp_id: int = 1, **kw):
    defaults = dict(
        id=opp_id,
        title=f"Scholarship {opp_id}",
        organization="Global NGO",
        type="scholarship",
        mode="remote",
        region=None,
        country=None,
        field_tags=["machine learning", "python"],
        language_requirements=["English"],
        eligibility_rules={},
        deadline=date.today() + timedelta(days=30),
        cost=0.0,
        impact_score=7.0,
        link=f"https://example.com/{opp_id}",
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def make_mentor(mentor_id: int = 1, **kw):
    defaults = dict(
        id=mentor_id,
        name=f"Dr. Mentor {mentor_id}",
        field="Computer Science",
        country="Kenya",
        region="Nairobi",
        language="English",
        diaspora_status=False,
        availability=True,
        expertise_tags=["machine learning", "python"],
        mentorship_capacity=4,
        rating=8.5,
        experience_years=6,
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def make_db(opps: list, mentors: list) -> MagicMock:
    """Return a MagicMock Session that routes db.query by ORM class."""
    mock_db = MagicMock()

    opp_query = MagicMock()
    opp_query.all.return_value = opps

    mentor_query = MagicMock()
    mentor_query.filter.return_value.all.return_value = mentors

    mock_db.query.side_effect = lambda model: (
        opp_query if model is OppORM else mentor_query
    )
    return mock_db


# ─────────────────────────────────────────────────────────────────────────────
# _is_eligible
# ─────────────────────────────────────────────────────────────────────────────

class TestIsEligible:
    def test_no_rules_eligible(self):
        assert _is_eligible(make_user(), make_opp()) is True

    def test_expired_deadline_ineligible(self):
        opp = make_opp(deadline=date.today() - timedelta(days=1))
        assert _is_eligible(make_user(), opp) is False

    def test_future_deadline_eligible(self):
        opp = make_opp(deadline=date.today() + timedelta(days=30))
        assert _is_eligible(make_user(), opp) is True

    def test_cost_exceeds_max_with_financial_barrier_ineligible(self):
        opp = make_opp(cost=1000.0, eligibility_rules={"max_cost": 500})
        user = make_user(barriers=["financial"])
        assert _is_eligible(user, opp) is False

    def test_cost_exceeds_max_without_financial_barrier_eligible(self):
        opp = make_opp(cost=1000.0, eligibility_rules={"max_cost": 500})
        user = make_user(barriers=[])
        assert _is_eligible(user, opp) is True

    def test_none_deadline_eligible(self):
        opp = make_opp(deadline=None)
        assert _is_eligible(make_user(), opp) is True


# ─────────────────────────────────────────────────────────────────────────────
# _build_pathway_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPathwaySummary:
    def _sopp(self, title="Scholarship A", mode="remote"):
        return SimpleNamespace(title=title, organization="NGO", mode=mode)

    def _smentor(self, name="Dr. A", field="CS"):
        return SimpleNamespace(name=name, field=field)

    def test_always_returns_list(self):
        result = _build_pathway_summary(make_user(), [], [], 0.3)
        assert isinstance(result, list)

    def test_high_risk_step_mentions_brain_drain(self):
        result = _build_pathway_summary(make_user(), [], [], 0.8)
        assert any("brain-drain" in s.lower() for s in result)

    def test_mentor_name_in_step_when_mentor_available(self):
        result = _build_pathway_summary(
            make_user(), [], [self._smentor("Prof. Kamau", "AI")], 0.3
        )
        assert any("Prof. Kamau" in s for s in result)

    def test_no_mentor_fallback_message(self):
        result = _build_pathway_summary(make_user(), [], [], 0.3)
        assert any("No mentors" in s for s in result)

    def test_step_count_at_least_3_with_data(self):
        opps = [self._sopp("Fellowship"), self._sopp("Workshop")]
        mentors = [self._smentor("Dr. B")]
        result = _build_pathway_summary(make_user(), opps, mentors, 0.3)
        assert len(result) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# generate_matches
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateMatches:
    def test_returns_match_result_instance(self):
        db = make_db([make_opp()], [make_mentor()])
        result = generate_matches(make_user(), db)
        assert isinstance(result, MatchResult)

    def test_top_opportunities_capped_at_top_n(self):
        opps = [make_opp(i) for i in range(1, 12)]
        db = make_db(opps, [])
        result = generate_matches(make_user(), db)
        assert len(result.top_opportunities) <= TOP_N_OPPORTUNITIES

    def test_top_mentors_capped_at_top_n(self):
        mentors = [make_mentor(i) for i in range(1, 10)]
        db = make_db([make_opp()], mentors)
        result = generate_matches(make_user(), db)
        assert len(result.top_mentors) <= TOP_N_MENTORS

    def test_opportunities_sorted_descending_by_final_score(self):
        opps = [
            make_opp(1, field_tags=["machine learning", "python"]),
            make_opp(2, field_tags=["interpretive dance"]),
        ]
        db = make_db(opps, [])
        result = generate_matches(make_user(), db)
        scores = [o.final_score for o in result.top_opportunities]
        assert scores == sorted(scores, reverse=True)

    def test_expired_opportunity_excluded(self):
        expired = make_opp(1, deadline=date.today() - timedelta(days=1))
        valid = make_opp(2, deadline=date.today() + timedelta(days=10))
        db = make_db([expired, valid], [])
        result = generate_matches(make_user(), db)
        assert all(o.opportunity_id != 1 for o in result.top_opportunities)

    def test_no_opportunities_yields_empty_top_list(self):
        db = make_db([], [make_mentor()])
        result = generate_matches(make_user(), db)
        assert result.top_opportunities == []

    def test_no_mentors_yields_empty_top_list(self):
        db = make_db([make_opp()], [])
        result = generate_matches(make_user(), db)
        assert result.top_mentors == []

    def test_risk_level_is_valid_string(self):
        db = make_db([], [])
        result = generate_matches(make_user(mobility_intent=0.9), db)
        assert result.risk_level in ("low", "medium", "high")
