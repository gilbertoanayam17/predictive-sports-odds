"""Tests del histórico de momios."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from predictive_odds.bet_tracker import BetTracker
from predictive_odds.data_fetcher import CLOSING_BOOKMAKER
from predictive_odds.domain import Selection
from predictive_odds.odds_history import OddsHistory
from predictive_odds.parlay_builder import Pick

KICKOFF = datetime(2025, 9, 1, 15, tzinfo=timezone.utc)


def _odds(home: float, draw: float, away: float, book: str = "b1") -> pd.DataFrame:
    base = {"match_id": "m1", "league": "T", "home_team": "H", "away_team": "A", "commence_time": KICKOFF,
            "bookmaker": book, "market": "1X2", "line": None}
    return pd.DataFrame([{**base, "selection": s, "odds": o} for s, o in (("HOME", home), ("DRAW", draw), ("AWAY", away))])


def test_snapshots_movement_and_closing() -> None:
    with OddsHistory(":memory:") as h:
        assert h.save_snapshot(_odds(2.2, 3.4, 3.3), KICKOFF - timedelta(days=2)) == 3
        h.save_snapshot(pd.concat([_odds(2.0, 3.5, 3.8), _odds(1.95, 3.6, 3.9, "b2")]), KICKOFF - timedelta(hours=1))
        h.save_snapshot(_odds(1.5, 4.0, 6.0), KICKOFF + timedelta(minutes=30))  # en juego: no es cierre
        h.save_snapshot(_odds(2.0, 3.5, 3.8, CLOSING_BOOKMAKER), KICKOFF)  # pseudo-casa: se ignora
        mov = h.line_movement().set_index("selection")
        assert mov.loc["HOME", "apertura"] == 2.2 and mov.loc["HOME", "ultimo"] == 1.5
        assert mov.loc["HOME", "capturas"] == 3
        closing = h.closing_odds()
        assert set(closing["bookmaker"]) == {"b1", "b2"}
        assert closing.loc[closing["selection"] == "HOME", "odds"].max() == 2.0
        fair = h.closing_consensus()
        assert sum(fair[("m1", s)] for s in (Selection.home(), Selection.draw(), Selection.away())) == pytest.approx(1)

        with BetTracker(":memory:") as t:
            t.record_pick(Pick("m1", "T", "H", "A", None, Selection.home(), 2.2, "b1", 0.5), 10)
            assert t.update_closing(closing) == 1
            assert t.bets()["clv"].iloc[0] > 0  # apostamos a 2.2 y la línea cerró ~2.0


def test_empty_history() -> None:
    with OddsHistory(":memory:") as h:
        assert h.save_snapshot(pd.DataFrame()) == 0
        assert h.line_movement().empty and h.closing_odds().empty
