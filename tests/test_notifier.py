"""Tests de las alertas de Telegram (sin red)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from predictive_odds.domain import Selection
from predictive_odds.notifier import (
    NotifierError,
    TelegramNotifier,
    format_alert,
    notify_value_bets,
    select_alerts,
)
from predictive_odds.parlay_builder import Pick

TOKEN = "123456:SECRETO-ABC"


class _Resp:
    def __init__(self, status: int, payload: Any) -> None:
        self.status_code, self._payload = status, payload

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Session:
    def __init__(self, responses: list[Any]) -> None:
        self.responses, self.calls = responses, []

    def post(self, url: str, json: Any = None, timeout: float = 0) -> _Resp:
        self.calls.append((url, json))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _pick(mid: str, edge_prob: float) -> Pick:
    return Pick(mid, "EPL", "Arsenal", "Chelsea <b>", pd.Timestamp("2026-11-08 15:00", tz="UTC"),
                Selection.home(), 2.0, "pinnacle", edge_prob, 0.5, 0.01, decision_prob=edge_prob)


def test_token_never_leaks_in_errors() -> None:
    url_err = requests.ConnectionError(f"Max retries exceeded with url: /bot{TOKEN}/sendMessage")
    for item in (url_err, _Resp(401, {"ok": False}), _Resp(400, {"ok": False, "description": f"bad {TOKEN}"}),
                 _Resp(500, ValueError("html"))):
        n = TelegramNotifier(TOKEN, "42", session=_Session([item]))
        with pytest.raises(NotifierError) as info:
            n.send("hola")
        assert TOKEN not in str(info.value) and info.value.__cause__ is None


def test_send_splits_long_messages_and_uses_html() -> None:
    session = _Session([_Resp(200, {"ok": True, "result": {}})] * 3)
    TelegramNotifier(TOKEN, "42", session=session).send("x" * 9000)
    assert len(session.calls) == 3 and session.calls[0][1]["parse_mode"] == "HTML"
    with pytest.raises(NotifierError, match="TELEGRAM_CHAT_ID"):
        TelegramNotifier(TOKEN, "").send("hola")


def test_discover_chat_ids_and_bot_name() -> None:
    updates = [{"message": {"chat": {"id": 42, "first_name": "Gil"}}},
               {"channel_post": {"chat": {"id": -100, "title": "Canal"}}}, {"edited": {}}]
    session = _Session([_Resp(200, {"ok": True, "result": {"username": "mi_bot"}}),
                        _Resp(200, {"ok": True, "result": updates})])
    n = TelegramNotifier(TOKEN, session=session)
    assert n.bot_name() == "mi_bot"
    assert n.discover_chat_ids() == [("42", "Gil"), ("-100", "Canal")]


def test_select_format_and_dedupe(tmp_path: Path) -> None:
    sent = tmp_path / "sent.json"
    picks = [_pick("m1", 0.56), _pick("m2", 0.51)]  # EV ajustado: +12% y +2%
    assert [p.match_id for p in select_alerts(picks, 0.05, sent)] == ["m1"]
    text = format_alert(picks[:1], 1000)
    assert "Chelsea &lt;b&gt;" in text and "+12.0%" in text and "08/11 15:00" in text  # contenido escapado
    session = _Session([_Resp(200, {"ok": True, "result": {}})])
    n = TelegramNotifier(TOKEN, "42", session=session)
    assert notify_value_bets(n, picks, 1000, "EPL", 0.05, sent) == 1
    assert notify_value_bets(n, picks, 1000, "EPL", 0.05, sent) == 0  # no repite
    assert len(session.calls) == 1


def test_failed_send_is_not_marked(tmp_path: Path) -> None:
    sent = tmp_path / "sent.json"
    n = TelegramNotifier(TOKEN, "42", session=_Session([_Resp(401, {"ok": False})]))
    with pytest.raises(NotifierError):
        notify_value_bets(n, [_pick("m1", 0.56)], 1000, "EPL", 0.05, sent)
    assert not sent.exists()


def test_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert TelegramNotifier.from_env() is None
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    n = TelegramNotifier.from_env()
    assert n is not None and n.chat_id == "42"


def test_cli_alert_and_setup_without_token(capsys: pytest.CaptureFixture[str]) -> None:
    from predictive_odds import cli

    assert cli.main(["--no-ml", "--alert"]) == 9
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().err
    assert cli.main(["--telegram-setup"]) == 9
    assert "@BotFather" in capsys.readouterr().out
