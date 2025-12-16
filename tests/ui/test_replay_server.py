
import json
import pytest
from pathlib import Path
from ui.replay_server import app
import tempfile

import tempfile
import json
import pytest
from pathlib import Path
from ui.replay_server import app

@pytest.fixture
def client(monkeypatch):
    app.config['TESTING'] = True
    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)
        monkeypatch.setenv("CRITICAL_RAT_RUNS_DIR", str(run_root))

        # Create dummy tournament run
        tourn_dir = run_root / "tourn_1"
        tourn_ep_dir = tourn_dir / "agents" / "agent_0" / "protocol_0"
        tourn_ep_dir.mkdir(parents=True)
        (tourn_ep_dir / "summary.json").write_text(json.dumps({"score": 100}))
        (tourn_ep_dir / "ticks.jsonl").write_text("\n".join([json.dumps({"tick": i}) for i in range(10)]))

        # Create dummy experiment run
        exp_dir = run_root / "exp_1"
        exp_dir.mkdir()
        (exp_dir / "summary.json").write_text(json.dumps({"score": 50}))
        (exp_dir / "ticks.jsonl").write_text("\n".join([json.dumps({"tick": i}) for i in range(5)]))
        
        with app.test_client() as client:
            yield client


def test_list_runs_returns_json(client):
    """Test that listing runs returns a JSON object with a 'runs' key."""
    rv = client.get('/api/runs')
    assert rv.is_json
    json_data = rv.get_json()
    assert 'runs' in json_data
    assert len(json_data['runs']) == 2
    run_ids = {r['id'] for r in json_data['runs']}
    assert 'tourn_1' in run_ids
    assert 'exp_1' in run_ids

def test_list_episodes_for_tournament_shape(client):
    """Test listing episodes for a tournament-shaped run."""
    rv = client.get('/api/runs/tourn_1/episodes')
    assert rv.is_json
    json_data = rv.get_json()
    assert 'episodes' in json_data
    assert len(json_data['episodes']) == 1
    episode = json_data['episodes'][0]
    assert episode['agent_id'] == 'agent_0'
    assert episode['protocol'] == 'protocol_0'

def test_list_episodes_for_experiment_shape(client):
    """Test listing episodes for an experiment-shaped run."""
    rv = client.get('/api/runs/exp_1/episodes')
    assert rv.is_json
    json_data = rv.get_json()
    assert 'episodes' in json_data
    assert len(json_data['episodes']) == 1
    episode = json_data['episodes'][0]
    assert episode['episode_id'] == 'default'

def test_get_summary(client):
    """Test fetching a summary for an episode."""
    rv = client.get('/api/runs/tourn_1/agent_0/protocol_0/summary')
    assert rv.is_json
    data = rv.get_json()
    assert data['score'] == 100


def test_ticks_paging(client):
    """Test that tick paging works correctly."""
    # First page
    rv = client.get('/api/runs/tourn_1/agent_0/protocol_0/ticks?start=0&limit=5')
    assert rv.is_json
    data = rv.get_json()
    assert data['start'] == 0
    assert data['limit'] == 5
    assert data['total'] == 10
    assert len(data['ticks']) == 5
    assert data['ticks'][0]['tick'] == 0

    # Second page
    rv = client.get('/api/runs/tourn_1/agent_0/protocol_0/ticks?start=5&limit=5')
    assert rv.is_json
    data = rv.get_json()
    assert data['start'] == 5
    assert data['limit'] == 5
    assert data['total'] == 10
    assert len(data['ticks']) == 5
    assert data['ticks'][0]['tick'] == 5

def test_path_traversal_blocked(client):
    """Test that path traversal attempts are blocked."""
    # This will be caught by Werkzeug's routing and return a 404
    rv = client.get('/api/runs/../etc/passwd/episodes')
    assert rv.status_code == 404

    # This will also be normalized by Werkzeug and result in a 404
    rv = client.get('/api/runs/tourn_1/../../bad/protocol_0/summary')
    assert rv.status_code == 404
