"""Tests for the projects API endpoints."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from backend.db.database import get_session
from backend.main import app


@pytest.fixture
def client():
    """Create a test client with an in-memory SQLite database."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session

    # Patch the engine used by on_startup so it doesn't try to open ./data/app.db
    with patch("backend.db.database.engine", engine):
        with TestClient(app) as c:
            yield c
    app.dependency_overrides.clear()


class TestCreateProject:
    def test_create_project(self, client):
        response = client.post(
            "/api/projects/",
            json={"name": "Test Project", "description": "A test", "task": "classification"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Project"
        assert data["task"] == "classification"
        assert "id" in data

    def test_create_project_detection(self, client):
        response = client.post(
            "/api/projects/",
            json={"name": "Det Project", "task": "detection"},
        )
        assert response.status_code == 200
        assert response.json()["task"] == "detection"

    def test_create_project_invalid_task(self, client):
        response = client.post(
            "/api/projects/",
            json={"name": "Bad", "task": "invalid_task"},
        )
        assert response.status_code == 400


class TestListProjects:
    def test_list_empty(self, client):
        response = client.get("/api/projects/")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_after_create(self, client):
        client.post(
            "/api/projects/",
            json={"name": "P1", "task": "classification"},
        )
        client.post(
            "/api/projects/",
            json={"name": "P2", "task": "detection"},
        )
        response = client.get("/api/projects/")
        assert response.status_code == 200
        assert len(response.json()) == 2


class TestGetProject:
    def test_get_existing(self, client):
        create = client.post(
            "/api/projects/",
            json={"name": "P1", "task": "segmentation"},
        )
        pid = create.json()["id"]
        response = client.get(f"/api/projects/{pid}")
        assert response.status_code == 200
        assert response.json()["name"] == "P1"

    def test_get_not_found(self, client):
        response = client.get("/api/projects/9999")
        assert response.status_code == 404


class TestUpdateProject:
    def test_update(self, client):
        create = client.post(
            "/api/projects/",
            json={"name": "Original", "task": "classification"},
        )
        pid = create.json()["id"]
        response = client.put(
            f"/api/projects/{pid}",
            json={"name": "Updated", "task": "detection"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Updated"
        assert response.json()["task"] == "detection"

    def test_update_not_found(self, client):
        response = client.put(
            "/api/projects/9999",
            json={"name": "X", "task": "classification"},
        )
        assert response.status_code == 404


class TestDeleteProject:
    def test_delete(self, client):
        create = client.post(
            "/api/projects/",
            json={"name": "ToDelete", "task": "classification"},
        )
        pid = create.json()["id"]
        response = client.delete(f"/api/projects/{pid}")
        assert response.status_code == 200
        # Verify it's gone
        get_response = client.get(f"/api/projects/{pid}")
        assert get_response.status_code == 404

    def test_delete_not_found(self, client):
        response = client.delete("/api/projects/9999")
        assert response.status_code == 404
