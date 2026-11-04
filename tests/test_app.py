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
    assert len(at.tabs) == 8

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
