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
    from predictive_odds import data_fetcher

    data_fetcher._API_FOOTBALL_PLAN_BLOCKED.clear()
    monkeypatch.setenv("PREDICTIVE_ODDS_NO_DOTENV", "1")
    for key in ("ODDS_API_KEY", "API_FOOTBALL_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        # setenv + delenv: monkeypatch recuerda el estado original y lo restaura aunque el código
        # bajo prueba (p. ej. la app al leer secrets) escriba la variable durante el test.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
