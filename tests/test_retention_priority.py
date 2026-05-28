import sys
import os

from types import SimpleNamespace
import pytest

from backend.algorithms.retention_priority import (
    _growth_potential,
    _regional_relevance,
    _knowledge_return,
    _network_effect,
    _contribution_potential,
    _local_continuation,
    _user_preference,
    retention_priority_score,
)


def make_user(**kwargs):
    defaults = dict(
        country="Kenya",
        region="Nairobi",
        mobility_intent=0.3,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def make_opp(**kwargs):
    defaults = dict(
        type="fellowship",
        mode="local",
        region=None,
        country="Kenya",
        impact_score=7.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestGrowthPotential:
    def test_impact_ten(self):
        opp = make_opp(impact_score=10.0)
        assert _growth_potential(opp) == 1.0

    def test_impact_five(self):
        opp = make_opp(impact_score=5.0)
        assert _growth_potential(opp) == pytest.approx(0.5)


class TestRegionalRelevance:
    def test_local_same_country(self):
        u = make_user(country="Kenya")
        opp = make_opp(mode="local", country="Kenya", region=None)
        assert _regional_relevance(u, opp) == 1.0

    def test_local_same_region(self):
        u = make_user(country="Uganda", region="East Africa")
        opp = make_opp(mode="local", country="Kenya", region="East Africa")
        assert _regional_relevance(u, opp) == 0.9

    def test_remote_keeps_physical(self):
        u = make_user()
        opp = make_opp(mode="remote")
        assert _regional_relevance(u, opp) == 0.7

    def test_international_low(self):
        u = make_user()
        opp = make_opp(mode="international")
        assert _regional_relevance(u, opp) == 0.3


class TestKnowledgeReturn:
    def test_research_type(self):
        opp = make_opp(type="research", mode="local")
        assert _knowledge_return(opp) == 0.9

    def test_remote_non_research(self):
        opp = make_opp(type="internship", mode="remote")
        assert _knowledge_return(opp) == 0.7

    def test_local_non_research(self):
        opp = make_opp(type="internship", mode="local")
        assert _knowledge_return(opp) == 0.4


class TestNetworkEffect:
    def test_community_type_fellowship(self):
        opp = make_opp(type="fellowship")
        assert _network_effect(opp) == 0.85

    def test_non_community_type(self):
        opp = make_opp(type="scholarship")
        assert _network_effect(opp) == 0.4


class TestContributionPotential:
    def test_local_high_impact_capped(self):
        opp = make_opp(mode="local", impact_score=10.0)
        # base=1.0 * 1.2 = 1.2 -> capped at 1.0
        assert _contribution_potential(opp) == 1.0

    def test_remote_low_impact(self):
        opp = make_opp(mode="remote", impact_score=5.0)
        # base=0.5 * 0.7 = 0.35
        assert _contribution_potential(opp) == pytest.approx(0.35)

    def test_local_medium_impact(self):
        opp = make_opp(mode="local", impact_score=5.0)
        # base=0.5 * 1.2 = 0.6
        assert _contribution_potential(opp) == pytest.approx(0.6)


class TestLocalContinuation:
    def test_local_mode(self):
        opp = make_opp(mode="local")
        assert _local_continuation(opp) == 1.0

    def test_remote_mode(self):
        opp = make_opp(mode="remote")
        assert _local_continuation(opp) == 0.8

    def test_hybrid_mode(self):
        opp = make_opp(mode="hybrid")
        assert _local_continuation(opp) == 0.8

    def test_international_mode(self):
        opp = make_opp(mode="international")
        assert _local_continuation(opp) == 0.3


class TestUserPreference:
    def test_zero_mobility(self):
        u = make_user(mobility_intent=0.0)
        assert _user_preference(u) == 1.0

    def test_full_mobility(self):
        u = make_user(mobility_intent=1.0)
        assert _user_preference(u) == 0.0

    def test_mid_mobility(self):
        u = make_user(mobility_intent=0.4)
        assert _user_preference(u) == pytest.approx(0.6)


class TestRetentionPriorityScore:
    def test_score_in_range(self):
        u = make_user()
        opp = make_opp()
        score = retention_priority_score(u, opp)
        assert 0.0 <= score <= 1.0

    def test_local_stayer_high_score(self):
        u = make_user(country="Kenya", region="Nairobi", mobility_intent=0.0)
        opp = make_opp(
            type="fellowship",
            mode="local",
            country="Kenya",
            region=None,
            impact_score=9.0,
        )
        assert retention_priority_score(u, opp) >= 0.8

    def test_local_beats_international_same_user(self):
        u = make_user(mobility_intent=0.9)
        opp_local = make_opp(mode="local")
        opp_intl = make_opp(mode="international")
        assert retention_priority_score(u, opp_local) > retention_priority_score(u, opp_intl)
