"""30 FastAPI TestClient route tests for backend routers."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def client():
    from backend.database import Base, get_db
    import backend.models.user         # noqa: F401  — registers ORM models
    import backend.models.mentor       # noqa: F401
    import backend.models.opportunity  # noqa: F401
    import backend.models.interaction  # noqa: F401
    from backend.main import app

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    with patch("backend.main.init_db"):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()
    engine.dispose()


VALID_USER = {
    "name": "Amara Diallo",
    "email": "amara.route.test@example.com",
    "country": "Senegal",
}


# ─── Health endpoints (3) ────────────────────────────────────────────────────
class TestHealthEndpoints:
    def test_health_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_info_algorithms_list(self, client):
        r = client.get("/info")
        assert r.status_code == 200
        data = r.json()
        assert "algorithms" in data
        assert isinstance(data["algorithms"], list)
        assert len(data["algorithms"]) > 0


# ─── Overview stats (5) ──────────────────────────────────────────────────────
class TestOverviewStats:
    def test_overview_200(self, client):
        r = client.get("/match/stats/overview")
        assert r.status_code == 200

    def test_total_users_initially_zero(self, client):
        r = client.get("/match/stats/overview")
        assert r.json()["total_users"] == 0

    def test_total_mentors_initially_zero(self, client):
        r = client.get("/match/stats/overview")
        assert r.json()["total_mentors"] == 0

    def test_total_opportunities_initially_zero(self, client):
        r = client.get("/match/stats/overview")
        assert r.json()["total_opportunities"] == 0

    def test_total_interactions_initially_zero(self, client):
        r = client.get("/match/stats/overview")
        assert r.json()["total_interactions"] == 0


# ─── User creation (6) ───────────────────────────────────────────────────────
class TestUserCreate:
    def test_post_users_201(self, client):
        r = client.post("/users/", json=VALID_USER)
        assert r.status_code == 201

    def test_id_is_integer(self, client):
        r = client.post("/users/", json=VALID_USER)
        assert isinstance(r.json()["id"], int)

    def test_name_matches(self, client):
        r = client.post("/users/", json=VALID_USER)
        assert r.json()["name"] == VALID_USER["name"]

    def test_email_matches(self, client):
        r = client.post("/users/", json=VALID_USER)
        assert r.json()["email"] == VALID_USER["email"]

    def test_duplicate_email_409(self, client):
        client.post("/users/", json=VALID_USER)
        r = client.post("/users/", json=VALID_USER)
        assert r.status_code == 409

    def test_missing_name_422(self, client):
        r = client.post("/users/", json={"email": "x@example.com", "country": "UK"})
        assert r.status_code == 422


# ─── Get user (3) ────────────────────────────────────────────────────────────
class TestGetUser:
    def test_get_existing_user_200(self, client):
        created = client.post("/users/", json=VALID_USER).json()
        r = client.get(f"/users/{created['id']}")
        assert r.status_code == 200

    def test_get_user_name_correct(self, client):
        created = client.post("/users/", json=VALID_USER).json()
        r = client.get(f"/users/{created['id']}")
        assert r.json()["name"] == VALID_USER["name"]

    def test_get_nonexistent_user_404(self, client):
        r = client.get("/users/999")
        assert r.status_code == 404


# ─── List users (2) ──────────────────────────────────────────────────────────
class TestListUsers:
    def test_list_users_returns_list(self, client):
        r = client.get("/users/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_list_users_after_create(self, client):
        client.post("/users/", json=VALID_USER)
        r = client.get("/users/")
        assert len(r.json()) == 1


# ─── Update user (4) ─────────────────────────────────────────────────────────
class TestUpdateUser:
    def test_patch_user_200(self, client):
        created = client.post("/users/", json=VALID_USER).json()
        r = client.patch(f"/users/{created['id']}", json={"name": "Updated Name"})
        assert r.status_code == 200

    def test_patch_updates_name(self, client):
        created = client.post("/users/", json=VALID_USER).json()
        r = client.patch(f"/users/{created['id']}", json={"name": "Updated Name"})
        assert r.json()["name"] == "Updated Name"

    def test_patch_leaves_email_unchanged(self, client):
        created = client.post("/users/", json=VALID_USER).json()
        r = client.patch(f"/users/{created['id']}", json={"name": "New Name"})
        assert r.json()["email"] == VALID_USER["email"]

    def test_patch_nonexistent_404(self, client):
        r = client.patch("/users/999", json={"name": "Ghost"})
        assert r.status_code == 404


# ─── Interaction logging (5) ─────────────────────────────────────────────────
class TestInteractionLog:
    def test_post_interaction_201(self, client):
        r = client.post("/match/interaction", json={"user_id": 1, "action": "clicked"})
        assert r.status_code == 201

    def test_response_has_status_logged(self, client):
        r = client.post("/match/interaction", json={"user_id": 1, "action": "saved"})
        assert r.json()["status"] == "logged"

    def test_response_has_action_key(self, client):
        r = client.post("/match/interaction", json={"user_id": 1, "action": "applied"})
        assert "action" in r.json()

    def test_action_value_matches_payload(self, client):
        r = client.post("/match/interaction", json={"user_id": 1, "action": "applied"})
        assert r.json()["action"] == "applied"

    def test_interaction_with_optional_fields(self, client):
        r = client.post("/match/interaction", json={
            "user_id": 1, "action": "clicked",
            "mentor_id": 2, "opportunity_id": 3,
        })
        assert r.status_code == 201


# ─── Match not found (2) ─────────────────────────────────────────────────────
class TestMatchNotFound:
    def test_match_unknown_user_404(self, client):
        r = client.get("/match/999")
        assert r.status_code == 404

    def test_match_404_detail_message(self, client):
        r = client.get("/match/999")
        assert "not found" in r.json()["detail"].lower()
