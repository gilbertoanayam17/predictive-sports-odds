"""Configuración común de los tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Los tests nunca leen el .env del usuario ni usan claves reales (no gastan créditos)."""
    monkeypatch.setenv("PREDICTIVE_ODDS_NO_DOTENV", "1")
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
