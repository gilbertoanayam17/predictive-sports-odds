"""Preparación para la nube: claves fuera del navegador, secrets y copias del registro."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from predictive_odds.bet_tracker import BetTracker, TrackerError, restore_backup  # noqa: E402
from predictive_odds.domain import Selection  # noqa: E402
from predictive_odds.parlay_builder import Pick  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SECRET = "clave-super-secreta-123"


@pytest.fixture
def offline_real_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Modo Real sin red: las fuentes reales se sustituyen por la simulación."""
    import predictive_odds.sources as sources
    from predictive_odds.data_fetcher import DataFetcher, SimulatedDataProvider, SimulationConfig

    def fake(live: bool, league_key: str, seasons, *a, **k):  # type: ignore[no-untyped-def]
        sim = SimulatedDataProvider(SimulationConfig(seasons=tuple(seasons)))
        return DataFetcher(sim, odds_provider=sim)

    monkeypatch.setattr(sources, "make_fetcher", fake)


def _all_text(at: AppTest) -> str:
    parts = [str(getattr(el, "value", "")) for kind in ("text_input", "caption", "markdown", "info", "success")
             for el in getattr(at, kind)]
    return " ".join(parts) + " ".join(str(getattr(t, "placeholder", "")) for t in at.text_input)


def test_server_key_never_sent_to_browser(monkeypatch: pytest.MonkeyPatch, offline_real_mode: None) -> None:
    monkeypatch.setenv("ODDS_API_KEY", SECRET)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    next(r for r in at.radio if r.label == "Fuente").set_value("Real").run()
    assert not at.exception, at.exception
    key_input = next(t for t in at.text_input if t.label == "Clave de The Odds API")
    assert key_input.value == "" and SECRET not in _all_text(at)
    assert any("configurada en el servidor" in c.value for c in at.caption)
    assert any(m.label == "Partidos analizados" for m in at.metric)  # usa la clave del servidor


def test_streamlit_secrets_are_loaded(offline_real_mode: None) -> None:
    at = AppTest.from_file(APP, default_timeout=180)
    at.secrets["ODDS_API_KEY"] = SECRET
    at.run()
    next(r for r in at.radio if r.label == "Fuente").set_value("Real").run()
    assert not at.exception, at.exception
    assert os.environ.get("ODDS_API_KEY") == SECRET  # disponible para el servidor
    assert SECRET not in _all_text(at)


def test_restore_backup(tmp_path: Path) -> None:
    src, dst = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    with BetTracker(src) as t:
        t.record_pick(Pick("m1", "EPL", "H", "A", pd.Timestamp("2026-11-08", tz="UTC"), Selection.home(),
                           2.0, "b", 0.55), 10)
    with BetTracker(dst) as t:
        pass
    assert restore_backup(src.read_bytes(), dst) == 1
    with BetTracker(dst) as t:
        assert len(t.bets()) == 1
    original = dst.read_bytes()
    with pytest.raises(TrackerError, match="SQLite"):
        restore_backup(b"no es sqlite", dst)
    other = tmp_path / "otra.sqlite"
    import sqlite3
    sqlite3.connect(other).execute("CREATE TABLE x (a INT)").connection.commit()
    with pytest.raises(TrackerError, match="no es un registro"):
        restore_backup(other.read_bytes(), dst)
    assert dst.read_bytes() == original  # una copia inválida no toca el registro actual
