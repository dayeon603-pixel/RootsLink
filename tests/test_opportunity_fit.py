import sys
import os

from datetime import date, timedelta
from types import SimpleNamespace

from backend.algorithms.opportunity_fit import (
    _tag_overlap,
    _eligibility,
    _location_feasibility,
    _aspiration_alignment,
    _barrier_compatibility,
    _timing_readiness,
    opportunity_fit_score,
)


def make_user(**kwargs):
    defaults = dict(
        country="Kenya",
        region="Nairobi",
        language="English",
        interests=["machine learning", "education"],
        skills=["python", "data analysis"],
        barriers=[],
        goals="I want to study computer science at university and build products.",
        education_level="undergraduate",
        mobility_intent=0.5,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def make_opp(**kwargs):
    defaults = dict(
        type="scholarship",
        mode="remote",
        region=None,
        country=None,
        field_tags=["machine learning", "python"],
        language_requirements=["English"],
        eligibility_rules={},
        deadline=None,
        cost=0.0,
        impact_score=7.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestTagOverlap:
    def test_empty_opp_tags_returns_neutral(self):
        assert _tag_overlap(["python"], []) == 0.5

    def test_empty_user_tags_returns_zero(self):
        assert _tag_overlap([], ["python"]) == 0.0

    def test_perfect_overlap(self):
        assert _tag_overlap(["python", "ml"], ["python", "ml"]) == 1.0

    def test_partial_overlap_jaccard(self):
        # intersection=1, union=3 -> 1/3
        assert abs(_tag_overlap(["python", "ml"], ["python", "r"]) - 1 / 3) < 1e-9

    def test_no_overlap(self):
        assert _tag_overlap(["python"], ["java"]) == 0.0

    def test_case_insensitive(self):
        assert _tag_overlap(["Python"], ["python"]) == 1.0


class TestEligibility:
    def test_no_rules_passes(self):
        u = make_user()
        opp = make_opp(eligibility_rules={})
        assert _eligibility(u, opp) == 1.0

    def test_education_level_fail(self):
        u = make_user(education_level="high_school")
        opp = make_opp(eligibility_rules={"education_level": "graduate"})
        assert _eligibility(u, opp) == 0.0

    def test_education_level_pass(self):
        u = make_user(education_level="graduate")
        opp = make_opp(eligibility_rules={"education_level": "undergraduate"})
        assert _eligibility(u, opp) == 1.0

    def test_cost_exceeds_max_with_financial_barrier(self):
        u = make_user(barriers=["financial"])
        opp = make_opp(cost=5000.0, eligibility_rules={"max_cost": 100})
        assert _eligibility(u, opp) == 0.0

    def test_cost_exceeds_max_without_financial_barrier(self):
        u = make_user(barriers=[])
        opp = make_opp(cost=5000.0, eligibility_rules={"max_cost": 100})
        assert _eligibility(u, opp) == 1.0


class TestLocationFeasibility:
    def test_remote_always_feasible(self):
        u = make_user(barriers=["financial"])
        opp = make_opp(mode="remote")
        assert _location_feasibility(u, opp) == 1.0

    def test_local_same_country(self):
        u = make_user(country="Kenya")
        opp = make_opp(mode="local", country="Kenya")
        assert _location_feasibility(u, opp) == 1.0

    def test_local_different_country(self):
        u = make_user(country="Kenya")
        opp = make_opp(mode="local", country="Nigeria", region=None)
        assert _location_feasibility(u, opp) == 0.3

    def test_local_same_region(self):
        u = make_user(country="Uganda", region="East Africa")
        opp = make_opp(mode="local", country="Kenya", region="East Africa")
        assert _location_feasibility(u, opp) == 0.85

    def test_international_with_financial_barrier(self):
        u = make_user(barriers=["financial"])
        opp = make_opp(mode="international")
        assert _location_feasibility(u, opp) == 0.4

    def test_international_no_barrier(self):
        u = make_user(barriers=[])
        opp = make_opp(mode="international")
        assert _location_feasibility(u, opp) == 0.9

    def test_hybrid_mode(self):
        u = make_user()
        opp = make_opp(mode="hybrid")
        assert _location_feasibility(u, opp) == 0.7


class TestAspirationAlignment:
    def test_no_goals_returns_neutral(self):
        u = make_user(goals=None)
        opp = make_opp(type="scholarship")
        assert _aspiration_alignment(u, opp) == 0.5

    def test_empty_goals_returns_neutral(self):
        u = make_user(goals="")
        opp = make_opp(type="scholarship")
        assert _aspiration_alignment(u, opp) == 0.5

    def test_scholarship_keyword_partial_match(self):
        u = make_user(goals="I want to study for a degree")
        opp = make_opp(type="scholarship")
        # keywords: ["study","degree","university","academic"] -- 2 match -> 2/4 = 0.5
        assert _aspiration_alignment(u, opp) == 0.5

    def test_research_full_match(self):
        u = make_user(goals="I want to research and publish science in a lab")
        opp = make_opp(type="research")
        # keywords: ["research","science","publish","lab"] -- all 4 match -> 1.0
        assert _aspiration_alignment(u, opp) == 1.0

    def test_unknown_type_returns_neutral(self):
        u = make_user(goals="Any goal here")
        opp = make_opp(type="unknown_type")
        assert _aspiration_alignment(u, opp) == 0.5


class TestBarrierCompatibility:
    def test_no_barriers_full_score(self):
        u = make_user(barriers=[])
        opp = make_opp(language_requirements=["English"], cost=0.0, mode="remote")
        assert _barrier_compatibility(u, opp) == 1.0

    def test_language_barrier_wrong_language(self):
        u = make_user(language="Swahili", barriers=["language"])
        opp = make_opp(language_requirements=["French"], cost=0.0, mode="local")
        # penalty=0.4 -> 0.6
        assert _barrier_compatibility(u, opp) == 0.6

    def test_financial_barrier_with_cost(self):
        u = make_user(barriers=["financial"])
        opp = make_opp(cost=500.0, mode="local", language_requirements=[])
        # penalty=0.3 -> 0.7
        assert _barrier_compatibility(u, opp) == 0.7

    def test_internet_barrier_remote(self):
        u = make_user(barriers=["internet"])
        opp = make_opp(mode="remote", language_requirements=[], cost=0.0)
        # penalty=0.3 -> 0.7
        assert _barrier_compatibility(u, opp) == 0.7

    def test_all_barriers_clamps_to_zero(self):
        u = make_user(language="Swahili", barriers=["language", "financial", "internet"])
        opp = make_opp(language_requirements=["French"], cost=200.0, mode="remote")
        # penalty=0.4+0.3+0.3=1.0 -> max(0.0, 0.0) = 0.0
        assert _barrier_compatibility(u, opp) == 0.0


class TestTimingReadiness:
    def test_no_deadline_returns_partial(self):
        opp = make_opp(deadline=None)
        assert _timing_readiness(opp) == 0.8

    def test_expired_deadline(self):
        opp = make_opp(deadline=date(2020, 1, 1))
        assert _timing_readiness(opp) == 0.0

    def test_very_tight_deadline(self):
        opp = make_opp(deadline=date.today() + timedelta(days=3))
        assert _timing_readiness(opp) == 0.3

    def test_short_deadline(self):
        opp = make_opp(deadline=date.today() + timedelta(days=14))
        assert _timing_readiness(opp) == 0.7

    def test_comfortable_deadline(self):
        opp = make_opp(deadline=date.today() + timedelta(days=30))
        assert _timing_readiness(opp) == 1.0


class TestOpportunityFitScore:
    def test_score_in_range(self):
        u = make_user()
        opp = make_opp()
        score = opportunity_fit_score(u, opp)
        assert 0.0 <= score <= 1.0

    def test_perfect_fit_near_one(self):
        u = make_user(
            skills=["machine learning", "python"],
            interests=["machine learning", "python"],
            barriers=[],
            goals="I want to research and publish science in a lab at university",
            education_level="graduate",
        )
        opp = make_opp(
            type="research",
            mode="remote",
            field_tags=["machine learning", "python"],
            eligibility_rules={},
            cost=0.0,
            deadline=date.today() + timedelta(days=30),
        )
        assert opportunity_fit_score(u, opp) >= 0.9

    def test_ineligible_user_lower_score(self):
        u_eligible = make_user(education_level="graduate")
        u_ineligible = make_user(education_level="high_school")
        opp = make_opp(eligibility_rules={"education_level": "graduate"})
        assert opportunity_fit_score(u_ineligible, opp) < opportunity_fit_score(u_eligible, opp)
