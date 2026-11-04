"""Ahorro de créditos de The Odds API (sin red)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from predictive_odds.config import get_league
from predictive_odds.data_fetcher import AuthenticationError, DataFetchError, HttpClient, TheOddsApiProvider

EVENT = {"id": "e1", "commence_time": "2026-11-08T15:00:00Z", "home_team": "Arsenal", "away_team": "Chelsea"}
ODDS = [{**EVENT, "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
    {"name": "Arsenal", "price": 2.0}, {"name": "Chelsea", "price": 3.8}, {"name": "Draw", "price": 3.6}]}]}]}]


class _Client(HttpClient):
    def __init__(self, responses: dict[str, Any]) -> None:
        super().__init__("https://x")
        self.responses, self.calls = responses, []

    def get_json(self, path: str, params: Any = None, use_cache: bool = True) -> Any:
        self.calls.append((path, dict(params or {})))
        value = self.responses[path.rsplit("/", 1)[-1]]
        if isinstance(value, Exception):
            raise value
        self.last_headers = {"x-requests-remaining": "480", "x-requests-used": "20"}
        return value


def test_no_upcoming_events_spends_no_credits(tmp_path: Path) -> None:
    client = _Client({"events": [], "odds": ODDS})
    api = TheOddsApiProvider("k", client=client, credits_file=tmp_path / "c.json")
    assert api.fetch_odds(get_league("EPL")).empty
    assert [c[0].rsplit("/", 1)[-1] for c in client.calls] == ["events"]  # /odds nunca se llamó


def test_upcoming_events_fetch_odds_with_window_and_record_credits(tmp_path: Path) -> None:
    client = _Client({"events": [EVENT], "odds": ODDS})
    credits = tmp_path / "c.json"
    api = TheOddsApiProvider("k", client=client, credits_file=credits)
    odds = api.fetch_odds(get_league("EPL"))
    assert len(odds) == 3
    odds_params = client.calls[-1][1]
    assert odds_params["regions"] == "eu" and odds_params["commenceTimeTo"].endswith(":00:00Z")
    assert api.credit_cost == 2  # 2 mercados × 1 región
    assert TheOddsApiProvider.read_credits(credits)["remaining"] == "480"


def test_events_failure_falls_back_to_odds_but_auth_errors_propagate(tmp_path: Path) -> None:
    client = _Client({"events": DataFetchError("caído"), "odds": ODDS})
    assert len(TheOddsApiProvider("k", client=client, credits_file=None).fetch_odds(get_league("EPL"))) == 3
    with pytest.raises(AuthenticationError):
        TheOddsApiProvider("k", client=_Client({"events": AuthenticationError("x"), "odds": ODDS}),
                           credits_file=None).fetch_odds(get_league("EPL"))


def test_horizon_none_skips_events_call() -> None:
    client = _Client({"odds": ODDS})
    TheOddsApiProvider("k", client=client, horizon_hours=None, credits_file=None).fetch_odds(get_league("EPL"))
    assert [c[0].rsplit("/", 1)[-1] for c in client.calls] == ["odds"]
    assert "commenceTimeTo" not in client.calls[0][1]
    assert TheOddsApiProvider.read_credits(Path("no_existe.json")) is None
