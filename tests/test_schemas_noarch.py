"""
Noarch Pydantic v2 schema validation tests.

Covers all models in backend/schemas/: user, mentor, opportunity, matching.
No torch, database, or network access required.
"""

import sys
import pathlib
import unittest
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "backend"))

from pydantic import ValidationError
from schemas.user import UserCreate, UserRead, UserUpdate
from schemas.mentor import MentorCreate, MentorRead, MentorUpdate
from schemas.opportunity import OpportunityCreate, OpportunityFilter
from schemas.matching import (
    ScoredOpportunity, ScoredMentor, MatchResult, InteractionCreate
)


class TestUserCreate(unittest.TestCase):
    def _valid(self, **kw):
        base = dict(name="Alice", email="alice@example.com", country="Kenya")
        base.update(kw)
        return UserCreate(**base)

    def test_valid_minimal(self):
        u = self._valid()
        self.assertEqual(u.name, "Alice")
        self.assertEqual(u.country, "Kenya")

    def test_default_mobility_intent(self):
        self.assertAlmostEqual(self._valid().mobility_intent, 0.5)

    def test_default_language_english(self):
        self.assertEqual(self._valid().language, "English")

    def test_default_interests_empty_list(self):
        self.assertEqual(self._valid().interests, [])

    def test_optional_region_none(self):
        self.assertIsNone(self._valid().region)

    def test_optional_goals_none(self):
        self.assertIsNone(self._valid().goals)

    def test_name_empty_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(name="")

    def test_name_too_long_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(name="x" * 121)

    def test_invalid_email_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(email="not-an-email")

    def test_mobility_intent_below_zero_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(mobility_intent=-0.1)

    def test_mobility_intent_above_one_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(mobility_intent=1.1)

    def test_country_empty_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(country="")


class TestUserUpdate(unittest.TestCase):
    def test_all_none_valid(self):
        u = UserUpdate()
        self.assertIsNone(u.name)
        self.assertIsNone(u.mobility_intent)

    def test_partial_update_valid(self):
        u = UserUpdate(name="Bob", mobility_intent=0.8)
        self.assertEqual(u.name, "Bob")
        self.assertAlmostEqual(u.mobility_intent, 0.8)

    def test_mobility_intent_out_of_range_raises(self):
        with self.assertRaises(ValidationError):
            UserUpdate(mobility_intent=1.5)


class TestMentorCreate(unittest.TestCase):
    def _valid(self, **kw):
        base = dict(
            name="Dr. Kim",
            email="kim@uni.edu",
            field="Biomedical Engineering",
            country="South Korea",
        )
        base.update(kw)
        return MentorCreate(**base)

    def test_valid_minimal(self):
        m = self._valid()
        self.assertEqual(m.field, "Biomedical Engineering")

    def test_default_capacity_three(self):
        self.assertEqual(self._valid().mentorship_capacity, 3)

    def test_default_availability_true(self):
        self.assertTrue(self._valid().availability)

    def test_default_diaspora_false(self):
        self.assertFalse(self._valid().diaspora_status)

    def test_name_empty_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(name="")

    def test_capacity_zero_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(mentorship_capacity=0)

    def test_capacity_above_twenty_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(mentorship_capacity=21)

    def test_experience_years_negative_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(experience_years=-1)

    def test_invalid_email_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(email="bad")


class TestMentorUpdate(unittest.TestCase):
    def test_all_none_valid(self):
        u = MentorUpdate()
        self.assertIsNone(u.field)
        self.assertIsNone(u.experience_years)

    def test_capacity_bounds_enforced(self):
        with self.assertRaises(ValidationError):
            MentorUpdate(mentorship_capacity=25)


class TestOpportunityCreate(unittest.TestCase):
    def _valid(self, **kw):
        base = dict(
            title="Rhodes Scholarship",
            organization="Rhodes Trust",
            type="scholarship",
            mode="international",
        )
        base.update(kw)
        return OpportunityCreate(**base)

    def test_valid_minimal(self):
        o = self._valid()
        self.assertEqual(o.title, "Rhodes Scholarship")

    def test_default_impact_score(self):
        self.assertAlmostEqual(self._valid().impact_score, 5.0)

    def test_default_cost_zero(self):
        self.assertAlmostEqual(self._valid().cost, 0.0)

    def test_title_empty_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(title="")

    def test_cost_negative_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(cost=-1.0)

    def test_impact_score_below_one_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(impact_score=0.9)

    def test_impact_score_above_ten_raises(self):
        with self.assertRaises(ValidationError):
            self._valid(impact_score=10.1)


class TestOpportunityFilter(unittest.TestCase):
    def test_all_none_valid(self):
        f = OpportunityFilter()
        self.assertIsNone(f.type)
        self.assertIsNone(f.mode)

    def test_partial_filter_valid(self):
        f = OpportunityFilter(type="scholarship", mode="international")
        self.assertEqual(f.type, "scholarship")


class TestScoredOpportunity(unittest.TestCase):
    def _valid(self, **kw):
        base = dict(
            opportunity_id=1, title="T", organization="O",
            type="fellowship", mode="remote",
            opportunity_fit_score=0.8, retention_priority_score=0.7,
            final_score=0.75, reasons=["reason"], barriers=[],
        )
        base.update(kw)
        return ScoredOpportunity(**base)

    def test_valid_construction(self):
        s = self._valid()
        self.assertEqual(s.opportunity_id, 1)

    def test_link_defaults_to_none(self):
        self.assertIsNone(self._valid().link)

    def test_link_set(self):
        s = self._valid(link="https://example.com")
        self.assertEqual(s.link, "https://example.com")


class TestMatchResult(unittest.TestCase):
    def test_valid_construction(self):
        scored_opp = ScoredOpportunity(
            opportunity_id=1, title="T", organization="O",
            type="scholarship", mode="remote",
            opportunity_fit_score=0.9, retention_priority_score=0.8,
            final_score=0.85, reasons=[], barriers=[],
        )
        scored_mentor = ScoredMentor(
            mentor_id=1, name="M", field="CS", country="NG",
            language="English", diaspora_status=True,
            mentor_match_score=0.95, reasons=[],
        )
        mr = MatchResult(
            user_id=42, brain_drain_risk=0.65, risk_level="medium",
            top_opportunities=[scored_opp], top_mentors=[scored_mentor],
            pathway_summary=["Step 1"],
        )
        self.assertEqual(mr.user_id, 42)
        self.assertEqual(mr.risk_level, "medium")
        self.assertEqual(len(mr.top_opportunities), 1)


class TestInteractionCreate(unittest.TestCase):
    def test_minimal_valid(self):
        ic = InteractionCreate(user_id=1, action="clicked")
        self.assertEqual(ic.action, "clicked")
        self.assertIsNone(ic.mentor_id)
        self.assertIsNone(ic.opportunity_id)

    def test_with_mentor_id(self):
        ic = InteractionCreate(user_id=1, mentor_id=5, action="meeting_completed")
        self.assertEqual(ic.mentor_id, 5)

    def test_with_opportunity_id(self):
        ic = InteractionCreate(user_id=2, opportunity_id=10, action="applied")
        self.assertEqual(ic.opportunity_id, 10)


if __name__ == "__main__":
    unittest.main()
