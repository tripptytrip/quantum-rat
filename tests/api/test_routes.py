from app.server import create_app


def test_step_endpoint_returns_parity_keys():
    app = create_app()
    client = app.test_client()
    resp = client.post("/step", json={"batch_size": 2})
    assert resp.status_code == 200
    data = resp.get_json()
    for key in ["rat", "brain", "stats", "whiskers", "vision"]:
        assert key in data
    assert "dopamine" in data["stats"]


def test_determinism_check_endpoint_matches_baseline():
    app = create_app()
    client = app.test_client()
    resp = client.post("/determinism_check")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["match"] is True
