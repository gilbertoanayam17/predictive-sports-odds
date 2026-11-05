"""API-Football (xG y lesiones) y pruebas de humo del CLI (sin red)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from predictive_odds import cli
from predictive_odds.config import get_league
from predictive_odds.data_fetcher import ApiFootballProvider, HttpClient, RateLimitError


class _JsonClient(HttpClient):
    def __init__(self, responses: dict[str, Any]) -> None:
        super().__init__("https://x")
        self.responses, self.calls = responses, []

    def get_json(self, path: str, params: Any = None, use_cache: bool = True) -> Any:
        self.calls.append((path, dict(params or {})))
        value = self.responses[path]
        if isinstance(value, Exception):
            raise value
        return value


def test_api_football_xg_budget_and_injuries() -> None:
    stats = {"errors": [], "response": [
        {"statistics": [{"type": "Shots", "value": 9}, {"type": "expected_goals", "value": "1.73"}]},
        {"statistics": [{"type": "expected_goals", "value": None}]},
    ]}
    client = _JsonClient({"fixtures/statistics": stats})
    af = ApiFootballProvider("k", client=client)
    matches = pd.DataFrame({"match_id": [f"af-{i}" for i in range(5)], "status": "FT",
                            "date": pd.date_range("2025-08-01", periods=5, tz="UTC"),
                            "home_xg": np.nan, "away_xg": np.nan})
    out = af.enrich_with_xg(matches, max_calls=2)
    assert len(client.calls) == 2  # respeta el presupuesto de llamadas
    assert out["home_xg"].notna().sum() == 2 and out.loc[4, "home_xg"] == pytest.approx(1.73)  # los más recientes
    assert out["away_xg"].isna().all()

    quota = ApiFootballProvider("k", client=_JsonClient({"fixtures/statistics": RateLimitError("cuota")}))
    assert quota.enrich_with_xg(matches, max_calls=5)["home_xg"].isna().all()  # se detiene sin romper

    inj = {"errors": [], "response": [
        {"team": {"name": "Arsenal"}, "player": {"name": "X", "type": "Missing Fixture", "reason": "Knee"},
         "fixture": {"id": 7}},
        {"team": {"name": "Arsenal"}},  # mal formada: se omite
    ]}
    df = ApiFootballProvider("k", client=_JsonClient({"injuries": inj})).fetch_injuries(get_league("EPL"), 2025)
    assert list(df.columns) == ["team", "player", "type", "reason", "match_id"] and len(df) == 1


@pytest.mark.parametrize("args", [
    ["--no-ml"],
    ["--no-ml", "--leagues", "EPL", "LIGA_MX", "--max-singles", "5"],
    ["--no-ml", "--risk", "--risk-rounds", "5"],
])
def test_cli_smoke(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "DATOS SIMULADOS" in out


def test_cli_files_and_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    report, bets, snaps = tmp_path / "r.html", tmp_path / "b.sqlite", tmp_path / "o.sqlite"
    assert cli.main(["--no-ml", "--report", str(report), "--track", str(bets), "--snapshot", str(snaps)]) == 0
    assert report.exists() and bets.exists() and snaps.exists()
    assert cli.main(["--track", str(bets), "--settle", "--played-rounds", "29", "--snapshot", str(snaps)]) == 0
    assert "Liquidadas ahora:" in capsys.readouterr().out
    assert cli.main(["--league", "NOPE"]) == 2
    assert cli.main(["--settle"]) == 2
    assert cli.main(["--no-ml", "--report", str(tmp_path / "r.pdf")]) == 7
    assert cli.main(["--live", "--league", "EPL"]) == 3  # sin ODDS_API_KEY: error de autenticación, sin red
