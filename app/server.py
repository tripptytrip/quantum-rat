"""Flask app factory exposing read-only routes over the engine."""

from typing import Any, Dict

from flask import Flask

from app.routes import bp
from app.routes.api import init_state


class Application:
    """Lightweight container for non-Flask context (kept for parity)."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config: Dict[str, Any] = config or {}
        self.services: Dict[str, Any] = {}

    def register_service(self, name: str, service: Any) -> None:
        self.services[name] = service

    def start(self) -> Dict[str, Any]:
        return {"status": "ready", "services": list(self.services)}


def create_app(config: Dict[str, Any] | None = None) -> Flask:
    """Create the Flask application and initialize engine state."""
    init_state(seed=(config or {}).get("seed", 1337) if config else 1337)
    flask_app = Flask(__name__)
    flask_app.register_blueprint(bp)
    return flask_app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
