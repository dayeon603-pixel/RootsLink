import sys
import os

from types import SimpleNamespace
import pytest

from backend.algorithms.brain_drain_risk import (
    _aspiration_mismatch,
    _mentorship_scarcity,
    _financial_constraint,
    _digital_gap,
    _community_weakness,
    _opportunity_desert,
    _local_visibility,
    brain_drain_risk_score,
    risk_level,
)


def make_user(**kwargs):
    defaults = dict(
        country="Kenya",
        region="Nairobi",
        goals="I want to study computer science and build tech in my community.",
        barriers=[],
        mobility_intent=0.5,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestAspirationMismatch:
    def test_long_goals_high_mobility(self):
        u = make_user(
            goals="I really want to pursue advanced research abroad in machine learning.",
            mobility_intent=1.0,
        )
        assert _aspiration_mismatch(u) == 1.0

    def test_short_goals_high_mobility(self):
        # len("Get out") = 7 <= 20 -> has_goals=0.3
        u = make_user(goals="Get out", mobility_intent=1.0)
        assert _aspiration_mismatch(u) == pytest.approx(0.3)

    def test_long_goals_low_mobility(self):
        u = make_user(
            goals="I want to build a technology company in Kenya.",
            mobility_intent=0.0,
        )
        assert _aspiration_mismatch(u) == 0.0

    def test_no_goals_uses_low_weight(self):
        u = make_user(goals=None, mobility_intent=0.6)
        assert _aspiration_mismatch(u) == pytest.approx(0.3 * 0.6)


class TestMentorshipScarcity:
    def test_no_mentors(self):
        u = make_user()
        assert _mentorship_scarcity(u, mentor_count=0) == 1.0

    def test_five_mentors(self):
        u = make_user()
        assert _mentorship_scarcity(u, mentor_count=5) == 0.0

    def test_one_mentor(self):
        u = make_user()
        assert _mentorship_scarcity(u, mentor_count=1) == pytest.approx(0.8)

    def test_ten_mentors_clamped(self):
        u = make_user()
        assert _mentorship_scarcity(u, mentor_count=10) == 0.0


class TestFinancialConstraint:
    def test_no_financial_barrier(self):
        u = make_user(barriers=[])
        assert _financial_constraint(u) == 0.0

    def test_has_financial_barrier(self):
        u = make_user(barriers=["financial"])
        assert _financial_constraint(u) == 1.0


class TestDigitalGap:
    def test_no_internet_barrier(self):
        u = make_user(barriers=[])
        assert _digital_gap(u) == 0.0

    def test_has_internet_barrier(self):
        u = make_user(barriers=["internet"])
        assert _digital_gap(u) == 1.0


class TestCommunityWeakness:
    def test_no_social_barriers(self):
        u = make_user(barriers=[])
        assert _community_weakness(u) == 0.0

    def test_two_social_barriers(self):
        u = make_user(barriers=["social", "family"])
        assert _community_weakness(u) == pytest.approx(0.5)

    def test_all_social_barriers(self):
        u = make_user(barriers=["social", "family", "isolation", "no role models"])
        assert _community_weakness(u) == 1.0


class TestOpportunityDesert:
    def test_rural_region(self):
        u = make_user(region="rural", country="Kenya")
        assert _opportunity_desert(u) == 0.9

    def test_sub_saharan_region(self):
        u = make_user(region="sub-saharan africa", country="Ethiopia")
        assert _opportunity_desert(u) == 0.9

    def test_urban_region_default(self):
        u = make_user(region="Nairobi", country="Kenya")
        assert _opportunity_desert(u) == 0.3


class TestLocalVisibility:
    def test_high_mobility_low_visibility(self):
        u = make_user(mobility_intent=1.0)
        assert _local_visibility(u) == 0.0

    def test_low_mobility_high_visibility(self):
        u = make_user(mobility_intent=0.0)
        assert _local_visibility(u) == 1.0


class TestRiskLevel:
    def test_high_risk(self):
        assert risk_level(0.7) == "high"

    def test_medium_risk(self):
        assert risk_level(0.5) == "medium"

    def test_low_risk(self):
        assert risk_level(0.2) == "low"

    def test_boundary_high(self):
        assert risk_level(0.65) == "high"

    def test_boundary_medium(self):
        assert risk_level(0.35) == "medium"


class TestBrainDrainRiskScore:
    def test_score_in_range(self):
        u = make_user()
        score = brain_drain_risk_score(u, mentor_count=0)
        assert 0.0 <= score <= 1.0

    def test_low_mobility_no_barriers_low_risk(self):
        u = make_user(
            mobility_intent=0.0,
            barriers=[],
            goals="short",
            region="Nairobi",
        )
        score = brain_drain_risk_score(u, mentor_count=5)
        assert score <= 0.35

    def test_high_mobility_all_barriers_high_risk(self):
        u = make_user(
            mobility_intent=1.0,
            barriers=["financial", "internet"],
            goals="I desperately want to leave my country and study abroad in Europe.",
            region="rural",
        )
        score = brain_drain_risk_score(u, mentor_count=0)
        assert score >= 0.60

    def test_raw_negative_clamps_to_zero(self):
        u = make_user(
            mobility_intent=0.0,
            barriers=[],
            goals=None,
            region="Nairobi",
        )
        score = brain_drain_risk_score(u, mentor_count=10)
        assert score >= 0.0
