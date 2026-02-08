"""Tests for health, encoder, and decoder listing endpoints."""

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


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestEncodersEndpoint:
    def test_list_encoders(self, client):
        response = client.get("/api/encoders")
        assert response.status_code == 200
        data = response.json()
        assert "dinov2_vits14" in data
        assert "dinov2_vitb14" in data
        assert "dinov2_vitl14" in data
        assert "dinov2_vitg14" in data

    def test_encoder_has_config(self, client):
        response = client.get("/api/encoders")
        data = response.json()
        vitb = data["dinov2_vitb14"]
        assert vitb["embed_dim"] == 768
        assert vitb["patch_size"] == 14
        assert vitb["num_heads"] == 12

    def test_register_variants_listed(self, client):
        response = client.get("/api/encoders")
        data = response.json()
        assert "dinov2_vits14_reg" in data
        assert "dinov2_vitb14_reg" in data


class TestDecodersEndpoint:
    def test_list_decoders(self, client):
        response = client.get("/api/decoders")
        assert response.status_code == 200
        data = response.json()
        assert "classification" in data
        assert "detection" in data
        assert "segmentation" in data

    def test_classification_decoders(self, client):
        response = client.get("/api/decoders")
        data = response.json()
        names = [d["name"] for d in data["classification"]]
        assert "linear_probe" in names
        assert "mlp" in names
        assert "transformer" in names

    def test_detection_decoders(self, client):
        response = client.get("/api/decoders")
        data = response.json()
        names = [d["name"] for d in data["detection"]]
        assert "detr_lite" in names
        assert "fpn" in names

    def test_segmentation_decoders(self, client):
        response = client.get("/api/decoders")
        data = response.json()
        names = [d["name"] for d in data["segmentation"]]
        assert "linear_seg" in names
        assert "upernet" in names
        assert "mask_transformer" in names

    def test_decoder_has_description(self, client):
        response = client.get("/api/decoders")
        data = response.json()
        for task_decoders in data.values():
            for decoder in task_decoders:
                assert "name" in decoder
                assert "description" in decoder
                assert len(decoder["description"]) > 0
