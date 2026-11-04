"""Tests de ingesta: cliente HTTP, parsers y validación (sin red)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import requests

from predictive_odds.data_fetcher import (
    ApiFootballProvider,
    AuthenticationError,
    DataFetcher,
    DataFetchError,
    DataValidationError,
    HttpClient,
    HttpClientConfig,
    RateLimitError,
    SimulatedDataProvider,
    TeamNameNormalizer,
    TheOddsApiProvider,
    validate_matches,
    validate_odds,
)
from predictive_odds.config import get_league


class FakeResponse:
    def __init__(self, status: int, payload: Any = None, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = str(payload)
        self.content = str(payload).encode()

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls = 0

    def get(self, *args: Any, **kwargs: Any) -> FakeResponse:
        item = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


def _client(responses: list[Any], retries: int = 2) -> tuple[HttpClient, FakeSession, list[float]]:
    session, sleeps = FakeSession(responses), []
    client = HttpClient("https://x", config=HttpClientConfig(max_retries=retries), session=session,  # type: ignore[arg-type]
                        sleep=sleeps.append)
    return client, session, sleeps


def test_http_retries_then_succeeds() -> None:
    client, session, sleeps = _client([requests.ConnectionError("boom"), FakeResponse(503), FakeResponse(200, {"ok": 1})])
    assert client.get_json("a") == {"ok": 1}
    assert session.calls == 3 and len(sleeps) == 2


def test_http_rate_limit_respects_retry_after() -> None:
    client, _, sleeps = _client([FakeResponse(429, headers={"Retry-After": "7"})], retries=1)
    with pytest.raises(RateLimitError):
        client.get_json("a")
    assert sleeps == [7.0]


def test_http_auth_and_bad_json() -> None:
    client, session, _ = _client([FakeResponse(401)])
    with pytest.raises(AuthenticationError):
        client.get_json("a")
    assert session.calls == 1  # sin reintentos
    client, _, _ = _client([FakeResponse(200, ValueError("no json"))])
    with pytest.raises(DataValidationError):
        client.get_json("a")


def test_http_cache(tmp_path: Any) -> None:
    session = FakeSession([FakeResponse(200, {"v": 1})])
    client = HttpClient("https://x", config=HttpClientConfig(cache_dir=tmp_path), session=session)  # type: ignore[arg-type]
    assert client.get_json("a", {"apiKey": "secret"}) == {"v": 1}
    assert client.get_json("a", {"apiKey": "secret"}) == {"v": 1}
    assert session.calls == 1
    assert all("secret" not in p.read_text() for p in tmp_path.iterdir())


def test_api_football_body_errors() -> None:
    client, _, _ = _client([FakeResponse(200, {"errors": {"token": "Error/Missing application key"}, "response": []})])
    with pytest.raises(AuthenticationError):
        ApiFootballProvider("k", client=client).fetch_matches(get_league("EPL"), 2025)


def test_api_football_parsing() -> None:
    payload = {"errors": [], "response": [
        {"fixture": {"id": 1, "date": "2025-08-16T14:00:00+00:00", "status": {"short": "FT"}},
         "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Chelsea"}},
         "goals": {"home": 3, "away": 2}, "score": {"fulltime": {"home": 2, "away": 2}}},
        {"fixture": {"id": 2, "date": "2025-09-01T14:00:00+00:00", "status": {"short": "NS"}},
         "teams": {"home": {"name": "Chelsea"}, "away": {"name": "Arsenal"}}, "goals": {}, "score": {}},
        {"fixture": {"id": 3, "date": "2025-09-02T14:00:00+00:00", "status": {"short": "PST"}},
         "teams": {"home": {"name": "A"}, "away": {"name": "B"}}},
        {"broken": True},
    ]}
    client, _, _ = _client([FakeResponse(200, payload)])
    df = ApiFootballProvider("k", client=client).fetch_matches(get_league("EPL"), 2025)
    assert list(df["match_id"]) == ["af-1", "af-2"]
    assert df.iloc[0]["home_goals"] == 2  # marcador a 90'
    assert df.iloc[1]["status"] == "NS"


def test_odds_api_parsing() -> None:
    events = [{"id": "e1", "commence_time": "2025-09-01T14:00:00Z", "home_team": "Arsenal", "away_team": "Chelsea",
               "bookmakers": [{"key": "pinnacle", "markets": [
                   {"key": "h2h", "outcomes": [{"name": "Arsenal", "price": 2.1}, {"name": "Chelsea", "price": 3.4},
                                               {"name": "Draw", "price": 3.5}]},
                   {"key": "totals", "outcomes": [{"name": "Over", "price": 1.9, "point": 2.5},
                                                  {"name": "Under", "price": 1.95, "point": 2.5}]}]}]}]
    df = validate_odds(TheOddsApiProvider.parse_events(events, "EPL"))
    assert len(df) == 5
    assert set(df["selection"]) == {"HOME", "AWAY", "DRAW", "OVER", "UNDER"}
    with pytest.raises(DataValidationError):
        TheOddsApiProvider.parse_events("nope", "EPL")


def test_missing_api_keys() -> None:
    with pytest.raises(AuthenticationError):
        TheOddsApiProvider("")
    with pytest.raises(AuthenticationError):
        ApiFootballProvider("")


def test_validation_drops_bad_rows() -> None:
    raw = pd.DataFrame([
        {"match_id": "1", "date": "2025-01-01", "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 0, "status": "FT"},
        {"match_id": "2", "date": "2025-01-01", "home_team": "A", "away_team": "A", "home_goals": 1, "away_goals": 0, "status": "FT"},
        {"match_id": "3", "date": "2025-01-01", "home_team": "C", "away_team": "D", "home_goals": None, "away_goals": 0, "status": "FT"},
        {"match_id": "4", "date": "bad", "home_team": "C", "away_team": "D", "status": "NS"},
    ])
    assert list(validate_matches(raw)["match_id"]) == ["1"]
    with pytest.raises(DataValidationError):
        validate_matches(raw.drop(columns="home_team"))
    odds = pd.DataFrame([
        {"home_team": "A", "away_team": "B", "bookmaker": "x", "market": "1X2", "selection": "HOME", "odds": 2.0},
        {"home_team": "A", "away_team": "B", "bookmaker": "x", "market": "1X2", "selection": "HOME", "odds": 0.9},
        {"home_team": "A", "away_team": "B", "bookmaker": "x", "market": "OU", "selection": "OVER", "line": 2.0, "odds": 1.9},
        {"home_team": "A", "away_team": "B", "bookmaker": "x", "market": "XYZ", "selection": "HOME", "odds": 1.9},
    ])
    assert len(validate_odds(odds)) == 1


def test_team_name_normalizer() -> None:
    n = TeamNameNormalizer()
    known = ["Manchester United", "Manchester City", "Tottenham Hotspur", "Pumas UNAM", "Atletico San Luis"]
    assert n.match("Man Utd", known) == "Manchester United"
    assert n.match("Man City", known) == "Manchester City"
    assert n.match("Tottenham", known) == "Tottenham Hotspur"
    assert n.match("U.N.A.M.", known) == "Pumas UNAM"
    assert n.match("Atlético de San Luis", known) == "Atletico San Luis"
    assert n.match("Real Madrid", known) is None


def test_link_odds_and_fixtures_from_odds() -> None:
    sim = SimulatedDataProvider()
    fetcher = DataFetcher(sim, odds_provider=sim)
    fixtures = fetcher.get_fixtures("EPL", 2025)
    odds = fetcher.get_odds("EPL")
    renamed = odds.assign(match_id="oa-" + odds["match_id"],
                          home_team=odds["home_team"].replace({"Manchester United": "Man Utd"}))
    linked = fetcher.link_odds_to_fixtures(renamed, fixtures)
    assert len(linked) == len(odds)
    assert set(linked["match_id"]) == set(fixtures["match_id"])
    derived = fetcher.fixtures_from_odds(renamed, fixtures["home_team"].tolist() + fixtures["away_team"].tolist())
    assert len(derived) == len(fixtures)
    assert "Manchester United" in set(derived["home_team"]) | set(derived["away_team"])


def test_history_raises_when_all_seasons_fail() -> None:
    with pytest.raises(DataFetchError):
        DataFetcher(SimulatedDataProvider()).get_history("EPL", [1990])
