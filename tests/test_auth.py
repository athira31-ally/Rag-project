from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_dev_token_and_me_roundtrip():
    token_resp = client.post("/auth/dev-token")
    assert token_resp.status_code == 200
    token = token_resp.json()["access_token"]

    me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    body = me_resp.json()
    assert body["email"] == "dev@example.com"


def test_me_requires_auth():
    resp = client.get("/auth/me")
    assert resp.status_code == 401


def test_login_returns_authorize_url():
    resp = client.get("/auth/login")
    assert resp.status_code == 200
    body = resp.json()
    assert "authorize_url" in body
    assert body["state"]
