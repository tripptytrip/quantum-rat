from __future__ import annotations

from flask import Blueprint

bp = Blueprint("api", __name__)

# Import routes to register handlers on blueprint
from . import api  # noqa: E402,F401

__all__ = ["bp"]
