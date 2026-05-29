"""Tests for opportunity_fit.score_reasons — explainability output.

score_reasons(user, opp) → (reasons: list[str], barriers: list[str])

Covers:
  TestScoreReasonsStructure   (4) — return type and mutually-exclusive polarity
  TestPositiveReasons         (8) — each reason trigger condition
  TestBarrierReasons          (6) — each barrier trigger condition
  TestReasonsNoFalsePositives (4) — conditions that must NOT produce reasons

Total: 22 tests.
"""
from __future__ import annotations

import sys
import os
from datetime import date, timedelta
from types import SimpleNamespace

from backend.algorithms.opportunity_fit import score_reasons


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_user(**kwargs):
    defaults = dict(
        country="Kenya",
        region="Nairobi",
        language="English",
        interests=["machine learning", "education"],
        skills=["python", "data analysis"],
        barriers=[],
        goals="I want to research and publish science in a lab at university",
        education_level="graduate",
        mobility_intent=0.5,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def make_opp(**kwargs):
    defaults = dict(
        type="research",
        mode="remote",
        region=None,
        country=None,
        field_tags=["machine learning", "python"],
        language_requirements=["English"],
        eligibility_rules={},
        deadline=None,
        cost=0.0,
        impact_score=8.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# TestScoreReasonsStructure
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreReasonsStructure:
    def test_returns_tuple_of_two_lists(self):
        result = score_reasons(make_user(), make_opp())
        assert isinstance(result, tuple)
        assert len(result) == 2
        reasons, barriers = result
        assert isinstance(reasons, list)
        assert isinstance(barriers, list)

    def test_all_reasons_are_strings(self):
        reasons, barriers = score_reasons(make_user(), make_opp())
        assert all(isinstance(r, str) for r in reasons)
        assert all(isinstance(b, str) for b in barriers)

    def test_ineligible_no_overlap_all_barriers(self):
        u = make_user(
            skills=[], interests=[], barriers=["language", "financial"],
            goals=None, education_level="high_school",
        )
        opp = make_opp(
            field_tags=["biomedical"],
            language_requirements=["French"],
            eligibility_rules={"education_level": "phd"},
            cost=500.0,
            deadline=date.today() + timedelta(days=5),
        )
        reasons, barriers = score_reasons(u, opp)
        assert len(barriers) > 0

    def test_perfect_fit_no_barriers(self):
        u = make_user(
            skills=["machine learning"],
            interests=["machine learning"],
            barriers=[],
            goals="I want to research and publish science in a lab",
            education_level="graduate",
        )
        opp = make_opp(
            field_tags=["machine learning"],
            mode="remote",
            cost=0.0,
            deadline=date.today() + timedelta(days=60),
        )
        _, barriers = score_reasons(u, opp)
        assert len(barriers) == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestPositiveReasons
# ─────────────────────────────────────────────────────────────────────────────

class TestPositiveReasons:
    def test_skill_match_reason_present(self):
        u = make_user(skills=["python", "machine learning"])
        opp = make_opp(field_tags=["python", "machine learning"])
        reasons, _ = score_reasons(u, opp)
        assert any("skill" in r.lower() or "matches" in r.lower() for r in reasons)

    def test_interest_alignment_reason_present(self):
        u = make_user(interests=["machine learning"])
        opp = make_opp(field_tags=["machine learning"])
        reasons, _ = score_reasons(u, opp)
        assert any("interest" in r.lower() for r in reasons)

    def test_remote_mode_reason_present(self):
        opp = make_opp(mode="remote")
        reasons, _ = score_reasons(make_user(), opp)
        assert any("remote" in r.lower() for r in reasons)

    def test_free_cost_reason_present(self):
        opp = make_opp(cost=0.0)
        reasons, _ = score_reasons(make_user(), opp)
        assert any("free" in r.lower() for r in reasons)

    def test_aspiration_alignment_reason_research(self):
        u = make_user(goals="I want to research and publish science in a lab")
        opp = make_opp(type="research")
        reasons, _ = score_reasons(u, opp)
        assert any("goal" in r.lower() or "fit" in r.lower() or "research" in r.lower() for r in reasons)

    def test_non_remote_no_remote_reason(self):
        opp = make_opp(mode="local")
        reasons, _ = score_reasons(make_user(), opp)
        assert not any("remote" in r.lower() for r in reasons)

    def test_paid_opportunity_no_free_reason(self):
        opp = make_opp(cost=200.0)
        reasons, _ = score_reasons(make_user(), opp)
        assert not any("free" in r.lower() for r in reasons)

    def test_skill_match_mention_tags(self):
        u = make_user(skills=["python"])
        opp = make_opp(field_tags=["python", "deep learning"])
        reasons, _ = score_reasons(u, opp)
        skill_reasons = [r for r in reasons if "skill" in r.lower() or "matches" in r.lower()]
        # At least one skill reason should mention a tag from the opportunity
        if skill_reasons:
            assert any("python" in r.lower() or "deep learning" in r.lower()
                       for r in skill_reasons)


# ─────────────────────────────────────────────────────────────────────────────
# TestBarrierReasons
# ─────────────────────────────────────────────────────────────────────────────

class TestBarrierReasons:
    def test_eligibility_barrier_present_when_ineligible(self):
        u = make_user(education_level="high_school")
        opp = make_opp(eligibility_rules={"education_level": "graduate"})
        _, barriers = score_reasons(u, opp)
        assert any("eligib" in b.lower() for b in barriers)

    def test_no_eligibility_barrier_when_eligible(self):
        u = make_user(education_level="graduate")
        opp = make_opp(eligibility_rules={"education_level": "graduate"})
        _, barriers = score_reasons(u, opp)
        assert not any("eligib" in b.lower() for b in barriers)

    def test_tight_deadline_barrier_within_14_days(self):
        opp = make_opp(deadline=date.today() + timedelta(days=10))
        _, barriers = score_reasons(make_user(), opp)
        assert any("deadline" in b.lower() for b in barriers)

    def test_no_deadline_barrier_when_far(self):
        opp = make_opp(deadline=date.today() + timedelta(days=30))
        _, barriers = score_reasons(make_user(), opp)
        assert not any("deadline" in b.lower() for b in barriers)

    def test_financial_barrier_with_cost(self):
        u = make_user(barriers=["financial"])
        opp = make_opp(cost=100.0)
        _, barriers = score_reasons(u, opp)
        assert any("financial" in b.lower() or "cost" in b.lower() for b in barriers)

    def test_language_barrier_present(self):
        u = make_user(barriers=["language"], language="Swahili")
        _, barriers = score_reasons(u, make_opp())
        assert any("language" in b.lower() for b in barriers)


# ─────────────────────────────────────────────────────────────────────────────
# TestReasonsNoFalsePositives
# ─────────────────────────────────────────────────────────────────────────────

class TestReasonsNoFalsePositives:
    def test_no_skill_reason_when_no_overlap(self):
        u = make_user(skills=["carpentry"])
        opp = make_opp(field_tags=["biomedical", "neuroscience"])
        reasons, _ = score_reasons(u, opp)
        assert not any("skill" in r.lower() and ("match" in r.lower() or "overlap" in r.lower())
                       for r in reasons)

    def test_no_interest_reason_when_no_interest_overlap(self):
        u = make_user(interests=["agriculture"])
        opp = make_opp(field_tags=["aerospace"])
        reasons, _ = score_reasons(u, opp)
        assert not any("interest" in r.lower() for r in reasons)

    def test_no_financial_barrier_when_no_financial_barrier_flag(self):
        u = make_user(barriers=[])
        opp = make_opp(cost=500.0)
        _, barriers = score_reasons(u, opp)
        assert not any("financial" in b.lower() for b in barriers)

    def test_no_language_barrier_when_no_language_barrier_flag(self):
        u = make_user(barriers=[])
        _, barriers = score_reasons(u, make_opp())
        assert not any("language" in b.lower() for b in barriers)
