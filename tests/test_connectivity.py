"""Tests de la comprobación de conexión (sin red)."""

from __future__ import annotations

from typing import Any

from predictive_odds.connectivity import check_api_football, check_football_data, check_odds_api
from predictive_odds.data_fetcher import DataFetchError, FootballDataCoUkProvider, HttpClient, HttpClientConfig


class _Resp:
    def __init__(self, status: int, payload: Any = None, headers: dict[str, str] | None = None) -> None:
        self.status_code, self._payload, self.headers = status, payload, headers or {}
        self.text = str(payload)
        self.content = self.text.encode()

    def json(self) -> Any:
        return self._payload


class _Session:
    def __init__(self, resp: _Resp) -> None:
        self.resp = resp

    def get(self, *a: Any, **k: Any) -> _Resp:
        return self.resp


def _client(resp: _Resp) -> HttpClient:
    return HttpClient("https://x", config=HttpClientConfig(max_retries=0), session=_Session(resp),  # type: ignore[arg-type]
                      sleep=lambda s: None)


def test_odds_api_check() -> None:
    ok = check_odds_api("k", _client(_Resp(200, [{"key": "soccer_epl"}], {"x-requests-remaining": "497",
                                                                       "x-requests-used": "3"})))
    assert ok.ok and "497" in ok.message
    assert not check_odds_api("k", _client(_Resp(401))).ok
    assert "Cuota" in check_odds_api("k", _client(_Resp(429))).message
    missing = check_odds_api("  ")
    assert not missing.ok and "the-odds-api.com" in missing.message


def test_api_football_check_is_optional() -> None:
    payload = {"errors": [], "response": {"subscription": {"plan": "Free"},
                                          "requests": {"current": 3, "limit_day": 100}}}
    ok = check_api_football("k", _client(_Resp(200, payload)))
    assert ok.ok and "3/100" in ok.message and not ok.required
    bad = check_api_football("k", _client(_Resp(200, {"errors": {"token": "invalid"}, "response": []})))
    assert not bad.ok and not bad.required
    assert not check_api_football("").ok


def test_football_data_check() -> None:
    class Down(FootballDataCoUkProvider):
        def fetch_matches(self, league: Any, season: int) -> Any:
            raise DataFetchError("sin red")

    assert not check_football_data("EPL", 2025, Down()).ok
    ucl = check_football_data("UCL", 2025)
    assert not ucl.ok and not ucl.required
