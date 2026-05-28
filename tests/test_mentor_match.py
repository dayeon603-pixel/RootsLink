import sys
import os

from types import SimpleNamespace

from backend.algorithms.mentor_match import (
    _field_alignment,
    _goal_alignment,
    _regional_relevance,
    _language_compatibility,
    _capacity_score,
    _quality_score,
    _experience_score,
    mentor_match_score,
    mentor_match_reasons,
)


def make_user(**kwargs):
    defaults = dict(
        country="Kenya",
        language="English",
        interests=["machine learning", "education"],
        skills=["python", "data analysis"],
        goals="I want to pursue a career in machine learning and build impactful technology.",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def make_mentor(**kwargs):
    defaults = dict(
        field="machine learning",
        country="Kenya",
        language="English",
        diaspora_status=False,
        experience_years=8,
        mentorship_capacity=3,
        availability=True,
        rating=8.0,
        expertise_tags=["python", "ml", "data science"],
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestFieldAlignment:
    def test_no_overlap_returns_zero(self):
        u = make_user(interests=["art"], skills=["painting"])
        m = make_mentor(expertise_tags=["finance"], field="banking")
        assert _field_alignment(u, m) == 0.0

    def test_full_overlap(self):
        u = make_user(interests=["python"], skills=["ml"])
        m = make_mentor(expertise_tags=["python"], field="ml")
        # user_tags={"python","ml"}, mentor_tags={"python","ml"} -> 2/2 = 1.0
        assert _field_alignment(u, m) == 1.0

    def test_partial_overlap_jaccard(self):
        u = make_user(interests=["python", "ml"], skills=[])
        m = make_mentor(expertise_tags=["python", "R"], field="stats")
        # user={"python","ml"}, mentor={"python","r","stats"} -> intersection=1, union=4
        assert abs(_field_alignment(u, m) - 1 / 4) < 1e-9

    def test_case_insensitive(self):
        u = make_user(interests=["Python"], skills=[])
        m = make_mentor(expertise_tags=["python"], field="other")
        assert _field_alignment(u, m) > 0.0


class TestGoalAlignment:
    def test_no_goals_returns_neutral(self):
        u = make_user(goals=None)
        m = make_mentor(field="machine learning")
        assert _goal_alignment(u, m) == 0.5

    def test_goals_match_field(self):
        u = make_user(goals="I want to work in machine learning and AI")
        m = make_mentor(field="machine learning")
        # field_words=["machine","learning"] -- both in goals -> 2/2 = 1.0
        assert _goal_alignment(u, m) == 1.0

    def test_goals_no_match(self):
        u = make_user(goals="I want to become an artist")
        m = make_mentor(field="machine learning")
        # "machine" and "learning" not in goals -> 0/2 = 0.0
        assert _goal_alignment(u, m) == 0.0


class TestRegionalRelevance:
    def test_same_country(self):
        u = make_user(country="Kenya")
        m = make_mentor(country="Kenya", diaspora_status=False)
        assert _regional_relevance(u, m) == 1.0

    def test_diaspora_mentor(self):
        u = make_user(country="Kenya")
        m = make_mentor(country="USA", diaspora_status=True)
        assert _regional_relevance(u, m) == 0.85

    def test_different_country_non_diaspora(self):
        u = make_user(country="Kenya")
        m = make_mentor(country="Nigeria", diaspora_status=False)
        assert _regional_relevance(u, m) == 0.4


class TestLanguageCompatibility:
    def test_same_language(self):
        u = make_user(language="English")
        m = make_mentor(language="English")
        assert _language_compatibility(u, m) == 1.0

    def test_one_english_fallback(self):
        u = make_user(language="Swahili")
        m = make_mentor(language="English")
        assert _language_compatibility(u, m) == 0.7

    def test_neither_english(self):
        u = make_user(language="Swahili")
        m = make_mentor(language="French")
        assert _language_compatibility(u, m) == 0.3


class TestCapacityScore:
    def test_unavailable_mentor(self):
        m = make_mentor(availability=False, mentorship_capacity=5)
        assert _capacity_score(m) == 0.0

    def test_capacity_five_is_max(self):
        m = make_mentor(availability=True, mentorship_capacity=5)
        assert _capacity_score(m) == 1.0

    def test_capacity_one(self):
        m = make_mentor(availability=True, mentorship_capacity=1)
        assert _capacity_score(m) == 0.2


class TestQualityScore:
    def test_rating_ten(self):
        m = make_mentor(rating=10.0)
        assert _quality_score(m) == 1.0

    def test_rating_five(self):
        m = make_mentor(rating=5.0)
        assert _quality_score(m) == 0.5


class TestExperienceScore:
    def test_zero_experience(self):
        m = make_mentor(experience_years=0)
        assert _experience_score(m) == 0.0

    def test_ten_years_is_max(self):
        m = make_mentor(experience_years=10)
        assert _experience_score(m) == 1.0

    def test_fifteen_years_capped(self):
        m = make_mentor(experience_years=15)
        assert _experience_score(m) == 1.0

    def test_five_years(self):
        m = make_mentor(experience_years=5)
        assert _experience_score(m) == 0.5


class TestMentorMatchScore:
    def test_score_in_range(self):
        u = make_user()
        m = make_mentor()
        score = mentor_match_score(u, m)
        assert 0.0 <= score <= 1.0

    def test_available_beats_unavailable(self):
        u = make_user()
        m_avail = make_mentor(availability=True)
        m_unavail = make_mentor(availability=False)
        assert mentor_match_score(u, m_avail) > mentor_match_score(u, m_unavail)

    def test_higher_rating_higher_score(self):
        u = make_user()
        m_high = make_mentor(rating=9.0)
        m_low = make_mentor(rating=3.0)
        assert mentor_match_score(u, m_high) > mentor_match_score(u, m_low)


class TestMentorMatchReasons:
    def test_same_language_reason(self):
        u = make_user(language="English")
        m = make_mentor(
            language="English",
            experience_years=3,
            rating=6.0,
            country="Nigeria",
            diaspora_status=False,
            expertise_tags=[],
            field="other",
        )
        reasons = mentor_match_reasons(u, m)
        assert any("English" in r for r in reasons)

    def test_high_experience_reason(self):
        u = make_user()
        m = make_mentor(experience_years=7, rating=5.0)
        reasons = mentor_match_reasons(u, m)
        assert any("7 years" in r for r in reasons)

    def test_high_rating_reason(self):
        u = make_user()
        m = make_mentor(rating=9.0, experience_years=3)
        reasons = mentor_match_reasons(u, m)
        assert any("9.0/10" in r for r in reasons)
