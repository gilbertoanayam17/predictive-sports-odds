"""Momio mínimo y recálculo a otra casa (Caliente)."""

from __future__ import annotations

import pytest

from predictive_odds.domain import Selection
from predictive_odds.parlay_builder import Pick, ValueBetConfig, ValueBetFinder


def _pick(**kw: object) -> Pick:
    base = dict(match_id="m1", league="EPL", home_team="H", away_team="A", kickoff=None,
                selection=Selection.home(), odds=2.0, bookmaker="pinnacle", model_prob=0.55,
                market_prob=0.50, decision_prob=0.525)
    base.update(kw)
    return Pick(**base)  # type: ignore[arg-type]


def test_minimum_odds_gives_exactly_the_required_edge() -> None:
    pick = _pick()
    m = pick.minimum_odds(0.04)
    assert m == pytest.approx(1.04 / 0.525)
    assert _pick(odds=m).adjusted_edge == pytest.approx(0.04)
    graded = _pick(selection=Selection.asian_handicap(True, -0.25), action_share=0.8)
    assert graded.minimum_odds(0.04) == pytest.approx((1 + 0.04 / 0.8) / 0.525)
    assert _pick(decision_prob=0.0).minimum_odds() == float("inf")


def test_reprice_to_caliente() -> None:
    finder = ValueBetFinder(ValueBetConfig(min_edge=0.04))
    pick = _pick()
    good = finder.reprice(pick, 2.10, "Caliente")
    assert good.bookmaker == "Caliente" and good.odds == 2.10 and good.stake_fraction > 0
    bad = finder.reprice(pick, 1.90, "Caliente")
    assert bad.stake_fraction == 0 and bad.adjusted_edge < 0.04
    with pytest.raises(ValueError):
        finder.reprice(pick, 1.0, "Caliente")
