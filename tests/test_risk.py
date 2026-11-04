"""Tests del simulador de riesgo Monte Carlo."""

from __future__ import annotations

import pytest

from predictive_odds.domain import Selection
from predictive_odds.parlay_builder import KellyConfig, Pick
from predictive_odds.risk import RiskConfig, risk_table, simulate_bankroll, stake_fractions


def _slate(n: int = 10, edge: float = 0.08) -> list[Pick]:
    # Momio 2.0, mercado justo 0.5, el modelo cree 0.5*(1+edge)
    p = 0.5 * (1 + edge)
    return [Pick(f"m{i}", "T", "H", "A", None, Selection.home(), 2.0, "b", p, 0.5, decision_prob=p)
            for i in range(n)]


def test_stakes_respect_exposure_cap() -> None:
    f = stake_fractions(_slate(40), KellyConfig(), 0.10)
    assert f.sum() == pytest.approx(0.10)


def test_edge_haircut_orders_outcomes() -> None:
    cfg = RiskConfig(n_rounds=30, n_sims=3000, seed=1)
    good = simulate_bankroll(_slate(), edge_haircut=0.0, config=cfg)
    none = simulate_bankroll(_slate(), edge_haircut=1.0, config=cfg)
    bad = simulate_bankroll(_slate(), edge_haircut=1.5, config=cfg)
    assert good["ev_por_jornada"] > 0 and none["ev_por_jornada"] == pytest.approx(0, abs=1e-12)
    assert bad["ev_por_jornada"] < 0
    assert good["mediana_final"] > none["mediana_final"] > bad["mediana_final"]
    assert good["p_perdida"] < bad["p_perdida"]
    with pytest.raises(ValueError):
        simulate_bankroll([], config=cfg)
    with pytest.raises(ValueError):
        simulate_bankroll(_slate(), edge_haircut=3.0, config=cfg)


def test_full_kelly_is_riskier() -> None:
    tab = risk_table(_slate(), fractions=(0.25, 1.0), haircuts=(1.5,), config=RiskConfig(n_rounds=38, n_sims=3000))
    quarter, full = tab.iloc[0], tab.iloc[1]
    assert full["p_dd_30"] > quarter["p_dd_30"]
    assert full["dd_mediano"] > quarter["dd_mediano"]
