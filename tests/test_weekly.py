"""Ciclo semanal automático (simulado, sin red)."""

from __future__ import annotations

from pathlib import Path

import pytest

from predictive_odds import cli
from predictive_odds.bet_tracker import BetTracker
from predictive_odds.notifier import TelegramNotifier


def test_weekly_cycle_registers_settles_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[str] = []
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setattr(TelegramNotifier, "send", lambda self, text: sent.append(text))
    paper, real = tmp_path / "s.sqlite", tmp_path / "r.sqlite"
    base = ["--weekly", "--leagues", "EPL", "--no-ml", "--season", "2025", "--max-singles", "4",
            "--track", str(paper), "--real-track", str(real)]

    # Semana 1: jornada 29 por jugar → se registran sugerencias; el usuario anota una apuesta real
    assert cli.main(base + ["--played-rounds", "28"]) == 0
    with BetTracker(paper) as t:
        pending = len(t.bets())
        pick_row = t.bets().iloc[0]
    assert pending > 0 and "Resumen semanal" in sent[-1] and "mín." in sent[-1] and "Caliente" in sent[-1]
    from predictive_odds.bet_tracker import Leg  # apuesta real a partir de la primera sugerencia
    import json

    leg = Leg(**json.loads(pick_row["legs"])[0])
    with BetTracker(real) as t:
        t._insert("single", "EPL", "real", [leg], 2.5, 0.5, 20)

    # Semana 2: la jornada 29 ya se jugó → se liquidan ambos registros y el resumen trae rendimiento
    assert cli.main(base + ["--played-rounds", "29"]) == 0
    with BetTracker(paper) as t:
        statuses = set(t.bets()["status"])
    with BetTracker(real) as t:
        assert t.bets()["status"].iloc[0] in ("won", "lost", "void")
    assert statuses & {"won", "lost", "void"}
    assert "Liquidadas esta semana" in sent[-1] and "Tus apuestas reales" in sent[-1]


def test_weekly_without_telegram(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--weekly", "--leagues", "EPL", "--no-ml", "--season", "2025",
                     "--track", str(tmp_path / "s.sqlite"), "--real-track", str(tmp_path / "r.sqlite")]) == 0
    assert "Telegram no configurado" in capsys.readouterr().out
