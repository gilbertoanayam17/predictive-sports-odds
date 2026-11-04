"""Prueba de humo de la app Streamlit (se omite si Streamlit no está instalado)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_renders_and_registers_bets(tmp_path: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert [m.label for m in at.metric][:3] == ["Partidos analizados", "Value bets", "Parlays"]
    assert len(at.tabs) == 9

    at.text_input[0].set_value(str(tmp_path / "bets.sqlite")).run()  # ruta del registro
    register = next(b for b in at.button if b.label == "Registrar las apuestas sugeridas")
    register.click().run()
    assert not at.exception, at.exception
    assert any("registrada" in s.value for s in at.success)


def test_app_risk_and_slate_tabs() -> None:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    next(b for b in at.button if b.label == "Simular riesgo").click().run()
    assert not at.exception, at.exception
    assert any("p_ruina" in str(c.value) for c in at.caption)
    next(b for b in at.button if b.label == "Analizar jornada").click().run()
    assert not at.exception, at.exception
    assert any(m.label == "Exposición total" for m in at.metric)


def test_app_real_mode_without_key_shows_onboarding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.radio[0].set_value("Real").run()
    assert not at.exception, at.exception
    assert any("Cómo usar datos reales" in i.value for i in at.info)
    assert any(b.label == "Probar conexión" for b in at.button)
    assert any(b.label == "Ejecutar backtest" for b in at.button)  # backtest real disponible sin clave


def test_app_settle_button(tmp_path: Path) -> None:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.text_input[0].set_value(str(tmp_path / "bets.sqlite")).run()
    next(b for b in at.button if b.label == "Registrar las apuestas sugeridas").click().run()
    next(b for b in at.button if b.label == "Liquidar con resultados").click().run()
    assert not at.exception, at.exception
    # En la simulación la jornada aún no se jugó: no hay nada que liquidar todavía
    assert any("pendientes" in i.value for i in at.info)


def test_app_connection_button_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """El botón 'Probar conexión' muestra un mensaje por fuente sin romper la app (red simulada caída)."""
    import predictive_odds.connectivity as conn

    monkeypatch.setattr(conn, "check_all", lambda *a, **k: [
        conn.ConnectionCheck("A", True, "ok"), conn.ConnectionCheck("B", False, "mal"),
        conn.ConnectionCheck("C", False, "opcional", required=False)])
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.radio[0].set_value("Real").run()
    next(b for b in at.button if b.label == "Probar conexión").click().run()
    assert not at.exception, at.exception


def test_app_league_study_panel() -> None:
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    ms = next(m for m in at.multiselect if m.label == "Ligas a estudiar")
    ms.set_value(["EPL"]).run()
    next(s for s in at.slider if s.label == "Temporadas anteriores").set_value(1).run()
    next(b for b in at.button if b.label == "Ejecutar estudio").click().run()
    assert not at.exception, at.exception
    assert any("no se guardan" in i.value for i in at.info)  # simulado: no guarda perfiles


def test_app_performance_tab_with_settled_bets(tmp_path: Path) -> None:
    import pandas as pd

    from predictive_odds.bet_tracker import BetTracker
    from predictive_odds.domain import Selection
    from predictive_odds.parlay_builder import Pick

    db = tmp_path / "perf.sqlite"
    kick = pd.Timestamp("2026-11-08 15:00", tz="UTC")
    with BetTracker(db) as t:
        for i in range(3):
            t.record_pick(Pick(f"m{i}", "EPL", "H", "A", kick, Selection.home(), 2.1, "b", 0.55), 10)
        t.settle(pd.DataFrame([{"match_id": f"m{i}", "home_team": "H", "away_team": "A", "date": kick,
                                "home_goals": i % 2, "away_goals": 0, "status": "FT"} for i in range(3)]))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.text_input[0].set_value(str(db)).run()
    assert not at.exception, at.exception
    assert any(m.label == "Beneficio esperado" for m in at.metric)
