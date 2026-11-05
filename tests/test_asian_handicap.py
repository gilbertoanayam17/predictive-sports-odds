"""Tests de Draw No Bet y hándicap asiático (push y medios resultados)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from predictive_odds.backtester import selection_result, settlement_pnl
from predictive_odds.bet_tracker import BetTracker
from predictive_odds.data_fetcher import TheOddsApiProvider, validate_odds
from predictive_odds.domain import MatchPrediction, Selection
from predictive_odds.parlay_builder import ParlayBuilder, ParlayConfig, Pick, ValueBetConfig, ValueBetFinder

AH = Selection.asian_handicap


@pytest.mark.parametrize("sel,score,expected", [
    (Selection.draw_no_bet(True), (1, 1), 0.0), (Selection.draw_no_bet(True), (2, 1), 1.0),
    (Selection.draw_no_bet(False), (2, 1), -1.0),
    (AH(True, -0.25), (1, 1), -0.5), (AH(True, -0.25), (2, 1), 1.0), (AH(True, -0.25), (0, 1), -1.0),
    (AH(True, -0.75), (2, 1), 0.5), (AH(True, -0.75), (3, 1), 1.0), (AH(True, -0.75), (1, 1), -1.0),
    (AH(True, -1.0), (2, 1), 0.0), (AH(True, -1.0), (3, 1), 1.0),
    (AH(False, 0.25), (1, 1), 0.5), (AH(False, 0.25), (2, 1), -1.0),
    (AH(False, 1.5), (2, 1), 1.0), (AH(True, -0.5), (1, 0), 1.0),
])
def test_result_grid(sel: Selection, score: tuple[int, int], expected: float) -> None:
    assert selection_result(sel, *score) == expected


def test_invalid_lines_and_labels() -> None:
    with pytest.raises(ValueError):
        AH(True, 0.3)
    with pytest.raises(ValueError):
        Selection(Selection.draw_no_bet().market, Selection.draw_no_bet().outcome, line=0.5)
    assert AH(True, -0.75).label == "AH Local -0.75"
    assert Selection.from_codes("AH", "AWAY", 0.25) == AH(False, 0.25)


def _pred(lh: float = 1.6, la: float = 1.0) -> MatchPrediction:
    g = np.arange(11)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    return MatchPrediction("m1", "H", "A", lh, la, m / m.sum())


def test_effective_probability_gives_exact_ev() -> None:
    pred = _pred()
    for sel in (Selection.draw_no_bet(True), AH(True, -0.25), AH(True, -0.75), AH(False, 0.25), AH(True, -1.0)):
        odds = 1.95
        grid = sel.result_grid(10)
        payoff = np.where(grid > 0, grid * (odds - 1), grid)
        exact_ev = float((pred.score_matrix * payoff).sum())
        p, s = pred.probability(sel), pred.action_share(sel)
        assert s * (p * odds - 1) == pytest.approx(exact_ev)
    # DNB = AH 0 y su "fracción en juego" excluye el empate
    assert pred.probability(Selection.draw_no_bet(True)) == pytest.approx(pred.probability(AH(True, 0.0)))
    assert pred.action_share(Selection.draw_no_bet(True)) == pytest.approx(1 - pred.probability(Selection.draw()))


def test_settlement_pnl() -> None:
    assert settlement_pnl(1.0, 2.0, 10) == 10
    assert settlement_pnl(0.5, 2.0, 10) == 5
    assert settlement_pnl(0.0, 2.0, 10) == 0
    assert settlement_pnl(-0.5, 2.0, 10) == -5
    assert settlement_pnl(-1.0, 2.0, 10) == -10


def _ah_odds(home_line: float, home_odds: float, away_odds: float) -> pd.DataFrame:
    base = {"match_id": "m1", "league": "T", "home_team": "H", "away_team": "A", "commence_time": None,
            "bookmaker": "b", "market": "AH"}
    return validate_odds(pd.DataFrame([{**base, "selection": "HOME", "line": home_line, "odds": home_odds},
                                       {**base, "selection": "AWAY", "line": -home_line, "odds": away_odds}]))


def test_value_finder_pairs_ah_sides_and_scales_stake() -> None:
    pred = _pred()
    finder = ValueBetFinder(ValueBetConfig(value_model_weight=1.0))
    picks = finder.evaluate({"m1": pred}, _ah_odds(-0.25, 1.95, 1.95))
    consensus = finder.market_consensus(_ah_odds(-0.25, 1.95, 1.95))
    assert consensus[("m1", AH(True, -0.25))] + consensus[("m1", AH(False, 0.25))] == pytest.approx(1.0)
    home = next(p for p in picks if p.selection == AH(True, -0.25))
    assert home.action_share < 1 and home.stake_fraction > 0
    # Los mercados con push no entran en parlays
    pool = ParlayBuilder(ParlayConfig(min_leg_prob=0.0, min_leg_edge=-1)).select_candidates(picks)
    assert pool == []


def test_odds_api_spreads_parsing() -> None:
    events = [{"id": "e1", "commence_time": "2025-09-01T14:00:00Z", "home_team": "H", "away_team": "A",
               "bookmakers": [{"key": "pinnacle", "markets": [{"key": "spreads", "outcomes": [
                   {"name": "H", "price": 1.9, "point": -0.75}, {"name": "A", "price": 2.0, "point": 0.75}]}]}]}]
    df = validate_odds(TheOddsApiProvider.parse_events(events, "EPL"))
    assert set(zip(df["selection"], df["line"])) == {("HOME", -0.75), ("AWAY", 0.75)}


def test_tracker_settles_push_and_half_results() -> None:
    with BetTracker(":memory:") as t:
        mk = lambda sel, mid: Pick(mid, "T", "H", "A", None, sel, 2.0, "b", 0.5)  # noqa: E731
        t.record_pick(mk(Selection.draw_no_bet(True), "m1"), 10)
        t.record_pick(mk(AH(True, -0.25), "m2"), 10)
        t.settle(pd.DataFrame([{"match_id": "m1", "home_goals": 1, "away_goals": 1, "status": "FT"},
                               {"match_id": "m2", "home_goals": 1, "away_goals": 1, "status": "FT"}]))
        bets = t.bets().set_index("id")
        assert bets.loc[1, "status"] == "void" and bets.loc[1, "pnl"] == 0
        assert bets.loc[2, "status"] == "lost" and bets.loc[2, "pnl"] == -5
