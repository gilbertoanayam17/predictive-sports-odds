"""Tests de las explicaciones de picks."""

from __future__ import annotations

from predictive_odds.data_fetcher import DataFetcher, SimulatedDataProvider
from predictive_odds.domain import Selection
from predictive_odds.explain import explain_pick
from predictive_odds.parlay_builder import Pick
from predictive_odds.pipeline import AnalysisPipeline, PipelineConfig
from predictive_odds.statistical_models import TeamAdjustment


def _result():  # type: ignore[no-untyped-def]
    sim = SimulatedDataProvider()
    return AnalysisPipeline(DataFetcher(sim, odds_provider=sim), PipelineConfig(use_ml=False)).run(evaluate=False)


def test_explanations_use_relevant_football_data() -> None:
    r = _result()
    for p in r.value_bets:
        lines = explain_pick(p, r.predictions[p.match_id], r.reports[p.match_id], r.adjustments)
        assert 1 <= len(lines) <= 3 and lines[0].startswith("Modelo ")
        warned = any(("Ojo" in line or "Bajas" in line) for line in lines)
        if p.selection.market.value in ("OU", "BTTS"):  # los avisos pueden desplazar el dato de goles
            assert any("Goles esperados" in line for line in lines) or warned
        if p.selection.market.value == "1X2":
            assert any("Forma:" in line or "Ojo" in line or "Bajas" in line for line in lines)


def test_warnings_take_priority() -> None:
    pick = Pick("m", "EPL", "Coventry City", "Arsenal", None, Selection.home(), 3.0, "b", 0.50, 0.36,
                decision_prob=0.43)
    lines = explain_pick(pick, None, None, {"Coventry City": TeamAdjustment(0.9, 1.05, "2 baja(s): X, Y")},
                         unknown_teams={"Coventry City"})
    assert lines[0].startswith("Modelo 50% frente a mercado 36% (+14 pp)")
    assert any("sin historial de Coventry City" in line for line in lines)
    assert any("diferencia grande" in line for line in lines)
    assert len(lines) == 3  # la línea de bajas queda fuera por el máximo, pero los avisos van primero
    no_market = explain_pick(Pick("m", "EPL", "A", "B", None, Selection.home(), 2.0, "b", 0.6))
    assert "implícito en el momio" in no_market[0]
