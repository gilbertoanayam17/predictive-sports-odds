"""Configuración común de los tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_api_keys(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Los tests nunca leen el .env del usuario ni usan claves reales (no gastan créditos).

    Además se ejecutan en una carpeta temporal: los archivos locales del usuario
    (league_profiles.json, bets.sqlite, .cache/) no afectan a los tests ni se modifican.
    """
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    monkeypatch.setenv("PREDICTIVE_ODDS_NO_DOTENV", "1")
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
