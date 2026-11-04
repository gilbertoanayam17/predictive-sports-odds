"""Tests de dominio, modelos y motor de apuestas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictive_odds.data_fetcher import DataFetcher, SimulatedDataProvider, SimulationConfig
from predictive_odds.config import get_league
from predictive_odds.domain import MatchPrediction, Selection, selections_compatible
from predictive_odds.parlay_builder import (
    ConflictChecker,
    KellyConfig,
    ParlayBuilder,
    ParlayConfig,
    Pick,
    ValueBetConfig,
    ValueBetFinder,
    apply_exposure_cap,
    kelly_fraction,
    modified_kelly,
    remove_margin,
)
from predictive_odds.pipeline import AnalysisPipeline, PipelineConfig
from predictive_odds.statistical_models import (
    FeatureEngineer,
    HeuristicAnalyzer,
    PoissonModel,
    evaluate_predictions,
)


@pytest.fixture(scope="module")
def sim() -> SimulatedDataProvider:
    return SimulatedDataProvider(SimulationConfig(seed=3))


@pytest.fixture(scope="module")
def history(sim: SimulatedDataProvider) -> pd.DataFrame:
    return DataFetcher(sim).get_history("EPL", [2024, 2025])


@pytest.fixture(scope="module")
def poisson_model(history: pd.DataFrame) -> PoissonModel:
    return PoissonModel().fit(history)


# ----------------------------------------------------------------------------- domain
def test_selection_validation() -> None:
    with pytest.raises(ValueError):
        Selection.over(2.0)  # solo líneas .5
    with pytest.raises(ValueError):
        Selection.from_codes("1X2", "OVER")


def test_contradictions_detected_by_masks() -> None:
    assert not selections_compatible([Selection.home(), Selection.away()])
    assert not selections_compatible([Selection.under(1.5), Selection.btts(True)])
    assert not selections_compatible([Selection.over(3.5), Selection.under(2.5)])
    assert selections_compatible([Selection.home(), Selection.over(2.5), Selection.btts(True)])


def test_prediction_markets_consistent(poisson_model: PoissonModel) -> None:
    pred = poisson_model.predict("Arsenal", "Chelsea")
    assert pred.score_matrix.sum() == pytest.approx(1.0)
    p1x2 = sum(pred.probability(s) for s in (Selection.home(), Selection.draw(), Selection.away()))
    assert p1x2 == pytest.approx(1.0)
    assert pred.probability(Selection.over(2.5)) + pred.probability(Selection.under(2.5)) == pytest.approx(1.0)
    assert pred.joint_probability([Selection.home(), Selection.away()]) == 0.0
    # Local + Over 2.5 están positivamente correlacionadas si el local es ofensivo
    joint = pred.joint_probability([Selection.over(2.5), Selection.btts(True)])
    assert joint > pred.probability(Selection.over(2.5)) * pred.probability(Selection.btts(True))


# ----------------------------------------------------------------------------- models
def test_poisson_recovers_true_strengths(sim: SimulatedDataProvider, poisson_model: PoissonModel) -> None:
    league = get_league("EPL")
    teams = list(league.sample_teams)
    true_lh = [sim.true_lambdas(league, 2025, "Arsenal", t)[0] for t in teams if t != "Arsenal"]
    est_lh = [poisson_model.expected_goals("Arsenal", t)[0] for t in teams if t != "Arsenal"]
    assert np.corrcoef(true_lh, est_lh)[0, 1] > 0.7
    assert 0.0 < poisson_model.home_adv < 0.5


def test_unknown_team_defaults_to_average(poisson_model: PoissonModel) -> None:
    lh, la = poisson_model.expected_goals("Equipo Nuevo FC", "Otro Nuevo")
    assert lh > la > 0


def test_features_have_no_leakage(history: pd.DataFrame) -> None:
    feats = FeatureEngineer().fit_transform(history)
    first = feats.iloc[0]
    assert np.isnan(first["home_form_ppg"]) and first["home_games"] == 0
    # Cambiar el resultado de un partido no altera SUS propias features
    altered = history.copy()
    idx = len(altered) // 2
    altered.loc[idx, ["home_goals", "away_goals"]] = [9, 0]
    feats2 = FeatureEngineer().fit_transform(altered)
    cols = ["home_form_ppg", "away_form_ppg", "elo_diff"]
    pd.testing.assert_series_equal(feats.loc[idx, cols], feats2.loc[idx, cols])


def test_heuristics(history: pd.DataFrame) -> None:
    an = HeuristicAnalyzer(history)
    table = an.standings()
    assert table["pos"].tolist() == list(range(1, len(table) + 1))
    assert (table["W"] + table["D"] + table["L"] == table["P"]).all()
    form = an.form("Arsenal", 5, "home")
    assert form.matches == 5 and len(form.results) == 5
    rep = an.match_report("Arsenal", "Chelsea")
    assert -1.0 <= rep.score <= 1.0


def test_evaluate_predictions() -> None:
    m = evaluate_predictions(np.array([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]), np.array([0, 2]))
    assert m["accuracy"] == 1.0
    assert m["log_loss"] == pytest.approx(-(np.log(0.7) + np.log(0.5)) / 2)


# ----------------------------------------------------------------------------- betting
def test_odds_math() -> None:
    fair = remove_margin([1.90, 1.90])
    assert fair == pytest.approx([0.5, 0.5])
    power = remove_margin([1.5, 4.0, 7.0], "power")
    assert sum(power) == pytest.approx(1.0)
    assert kelly_fraction(0.5, 2.0) == 0.0
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    cfg = KellyConfig(fraction=0.5, model_trust=1.0, max_stake=1.0)
    assert modified_kelly(0.6, 2.0, config=cfg) == pytest.approx(0.1)
    assert modified_kelly(0.6, 2.0, config=cfg, legs=2) == pytest.approx(0.1 * cfg.leg_penalty)
    assert modified_kelly(0.9, 2.0) <= KellyConfig().max_stake


def _pred(mid: str, lh: float = 1.6, la: float = 1.1) -> MatchPrediction:
    from scipy.stats import poisson
    g = np.arange(11)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    return MatchPrediction(mid, f"H{mid}", f"A{mid}", lh, la, m / m.sum())


def _pick(mid: str, sel: Selection, odds: float, p: float) -> Pick:
    return Pick(mid, "T", f"H{mid}", f"A{mid}", None, sel, odds, "book", p, p * 0.95)


def test_value_finder_flags_only_value() -> None:
    preds = {"m1": _pred("m1")}
    p_home = preds["m1"].probability(Selection.home())
    odds = pd.DataFrame([
        {"match_id": "m1", "league": "T", "home_team": "Hm1", "away_team": "Am1", "commence_time": None,
         "bookmaker": "b", "market": "1X2", "selection": s, "line": None, "odds": o}
        for s, o in (("HOME", 1.15 / p_home), ("DRAW", 3.0), ("AWAY", 3.0))
    ])
    picks = ValueBetFinder(ValueBetConfig(value_model_weight=1.0)).find(preds, odds)
    assert [p.selection for p in picks] == [Selection.home()]
    assert picks[0].stake_fraction > 0
    # Encogido hacia el mercado, el EV ajustado es menor que el del modelo puro
    shrunk = ValueBetFinder(ValueBetConfig(value_model_weight=0.5)).evaluate(preds, odds)
    home = next(p for p in shrunk if p.selection == Selection.home())
    assert home.adjusted_edge < home.edge


def test_conflict_checker() -> None:
    checker = ConflictChecker()
    contradictory = [_pick("m1", Selection.home(), 2.0, 0.5), _pick("m1", Selection.away(), 3.0, 0.3)]
    redundant = [_pick("m1", Selection.over(2.5), 2.0, 0.5), _pick("m1", Selection.over(1.5), 1.3, 0.8)]
    ok = [_pick("m1", Selection.home(), 2.0, 0.5), _pick("m2", Selection.away(), 3.0, 0.3)]
    assert any("contradictoria" in i for i in checker.check(contradictory))
    assert any("redundante" in i for i in checker.check(redundant))
    assert checker.check(ok) == []


def test_parlay_builder_independent_and_same_match() -> None:
    preds = {m: _pred(m) for m in ("m1", "m2", "m3")}
    picks = [
        _pick("m1", Selection.over(1.5), 1.45, preds["m1"].probability(Selection.over(1.5))),
        _pick("m2", Selection.home(), 2.1, preds["m2"].probability(Selection.home())),
        _pick("m3", Selection.over(1.5), 1.45, preds["m3"].probability(Selection.over(1.5))),
        _pick("m1", Selection.btts(True), 2.0, preds["m1"].probability(Selection.btts(True))),
    ]
    parlays = ParlayBuilder(ParlayConfig(min_leg_prob=0.45)).build(picks, preds)
    assert parlays
    for par in parlays:
        assert len({leg.match_id for leg in par.legs}) == len(par.legs)  # sin mismo partido
        assert par.joint_prob == pytest.approx(np.prod([leg.model_prob for leg in par.legs]))
        assert 0 < par.stake_fraction <= 0.015

    sgp = ParlayBuilder(ParlayConfig(min_leg_prob=0.45, allow_same_match=True))
    # BTTS Sí implica Más de 1.5 → redundante, se rechaza
    assert sgp.evaluate_combination([picks[0], picks[3]], preds) is None
    p = preds["m1"]
    legs = [_pick("m1", Selection.home(), 2.3, p.probability(Selection.home())),
            _pick("m1", Selection.over(2.5), 2.3, p.probability(Selection.over(2.5)))]
    same = sgp.evaluate_combination(legs, preds)
    assert same is not None and same.warnings
    assert same.joint_prob == pytest.approx(p.joint_probability([Selection.home(), Selection.over(2.5)]))
    assert same.joint_prob > same.independent_prob  # correlación positiva local/goles


def test_exposure_cap() -> None:
    singles = [Pick("m", "T", "H", "A", None, Selection.home(), 2.0, "b", 0.6, stake_fraction=0.08)] * 2
    s, _, factor = apply_exposure_cap(singles, [], 0.10)
    assert factor == pytest.approx(0.625)
    assert sum(p.stake_fraction for p in s) == pytest.approx(0.10)


# ----------------------------------------------------------------------------- pipeline
def test_pipeline_end_to_end(sim: SimulatedDataProvider) -> None:
    result = AnalysisPipeline(DataFetcher(sim, odds_provider=sim), PipelineConfig()).run()
    assert len(result.predictions) == 10
    assert result.all_picks
    assert result.total_stake <= 0.10 * result.bankroll + 1e-6
    assert {"poisson_dixon_coles", "baseline_frecuencias"} <= set(result.evaluation)
    assert result.evaluation["poisson_dixon_coles"]["log_loss"] < result.evaluation["baseline_frecuencias"]["log_loss"]
    assert not result.value_bets_frame().empty
