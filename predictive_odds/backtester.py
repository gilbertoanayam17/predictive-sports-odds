"""Backtesting walk-forward del sistema completo.

Para cada fecha de partidos, en orden cronológico:

1. se entrenan los modelos SOLO con partidos anteriores (Poisson se reentrena
   cada ``refit_every`` fechas; las features del ML se calculan en orden
   cronológico, sin fuga de información);
2. se predicen los partidos del día y se buscan apuestas de valor con los
   momios pre-partido (la línea de cierre nunca se usa para decidir);
3. se calculan los stakes con el bankroll vigente (Kelly modificado + tope de
   exposición) y se liquidan con el resultado real.

Métricas: ROI/yield, crecimiento del bankroll, drawdown máximo, tasa de
acierto, CLV (Closing Line Value: si se apuesta mejor que el cierre de forma
sistemática, el edge es real) y una tabla de calibración por tramos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .data_fetcher import CLOSING_BOOKMAKER
from .domain import STATUS_FINISHED, MatchPrediction, Selection
from .parlay_builder import (
    ParlayBuilder,
    ParlayConfig,
    Pick,
    ValueBetConfig,
    ValueBetFinder,
    apply_exposure_cap,
    apply_match_cap,
)
from .statistical_models import (
    EnsemblePredictor,
    FeatureEngineer,
    InsufficientDataError,
    MLOutcomeModel,
    PoissonModel,
    PoissonModelConfig,
)

logger = logging.getLogger(__name__)


def selection_wins(selection: Selection, home_goals: int, away_goals: int) -> bool:
    """Indica si una selección gana con un marcador dado.

    Args:
        selection: Selección.
        home_goals: Goles del local (90').
        away_goals: Goles del visitante (90').

    Returns:
        ``True`` si la selección resulta ganadora.
    """
    return selection_result(selection, home_goals, away_goals) > 0


def selection_result(selection: Selection, home_goals: int, away_goals: int) -> float:
    """Resultado graduado: +1 gana, +0.5 medio gana, 0 push, −0.5 medio pierde, −1 pierde.

    Args:
        selection: Selección.
        home_goals: Goles del local (90').
        away_goals: Goles del visitante (90').

    Returns:
        Resultado en {−1, −0.5, 0, 0.5, 1}.
    """
    size = max(10, int(home_goals), int(away_goals))
    return float(selection.result_grid(size)[int(home_goals), int(away_goals)])


def settlement_pnl(result: float, odds: float, stake: float) -> float:
    """Beneficio de una apuesta dado su resultado graduado.

    Args:
        result: Salida de :func:`selection_result`.
        odds: Momio decimal.
        stake: Importe apostado.

    Returns:
        Beneficio (negativo si pierde; 0 si es push).
    """
    return stake * (odds - 1) * result if result > 0 else stake * result


@dataclass
class BacktestConfig:
    """Configuración del backtest.

    Attributes:
        initial_bankroll: Bankroll inicial.
        warmup_matches: Partidos mínimos antes de empezar a apostar.
        refit_every: Reentrenar los modelos cada N fechas de partidos.
        use_ml: Si se usa el ensamble Poisson + ML.
        ml_weight: Peso del ML.
        include_parlays: Si se apuestan también los parlays sugeridos.
        max_total_exposure: Exposición máxima por fecha.
        max_match_exposure: Exposición máxima de las sencillas de un mismo partido.
        closing_bookmaker: Casa que representa la línea de cierre.
        poisson: Configuración de Poisson.
        value: Configuración de valor.
        parlay: Configuración de parlays.
    """

    initial_bankroll: float = 1000.0
    warmup_matches: int = 150
    refit_every: int = 1
    use_ml: bool = True
    ml_weight: float = 0.35
    include_parlays: bool = False
    max_total_exposure: float = 0.10
    max_match_exposure: float | None = None
    closing_bookmaker: str = CLOSING_BOOKMAKER
    poisson: PoissonModelConfig = field(default_factory=PoissonModelConfig)
    value: ValueBetConfig = field(default_factory=ValueBetConfig)
    parlay: ParlayConfig = field(default_factory=ParlayConfig)


@dataclass
class BacktestResult:
    """Resultado del backtest.

    Attributes:
        bets: Una fila por apuesta liquidada.
        bankroll: Evolución del bankroll por fecha.
        evaluated: Todas las selecciones evaluadas (para calibración).
        metrics: Métricas agregadas.
    """

    bets: pd.DataFrame
    bankroll: pd.DataFrame
    evaluated: pd.DataFrame
    metrics: dict[str, float]

    def by_market(self) -> pd.DataFrame:
        """Rendimiento desglosado por mercado/tipo de apuesta."""
        if self.bets.empty:
            return pd.DataFrame()
        g = self.bets.groupby("market")
        out = pd.DataFrame({
            "apuestas": g.size(), "stake": g["stake"].sum(), "beneficio": g["pnl"].sum(),
            "acierto": g["won"].mean(), "momio_medio": g["odds"].mean(), "clv_medio": g["clv"].mean(),
        })
        out["roi"] = out["beneficio"] / out["stake"]
        return out.sort_values("apuestas", ascending=False)

    def calibration(self, bins: Sequence[float] = (0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)) -> pd.DataFrame:
        """Tabla de calibración: probabilidad media predicha vs frecuencia real.

        Args:
            bins: Límites de los tramos de probabilidad.

        Returns:
            DataFrame con ``tramo, n, p_media, frecuencia_real, diferencia``.
        """
        if self.evaluated.empty:
            return pd.DataFrame()
        ev = self.evaluated.assign(tramo=pd.cut(self.evaluated["model_prob"], list(bins), include_lowest=True))
        g = ev.groupby("tramo", observed=True)
        out = pd.DataFrame({"n": g.size(), "p_media": g["model_prob"].mean(), "frecuencia_real": g["won"].mean()})
        out["diferencia"] = out["frecuencia_real"] - out["p_media"]
        return out.reset_index()


class Backtester:
    """Simula el sistema completo fecha a fecha sobre datos históricos."""

    def __init__(self, matches: pd.DataFrame, odds: pd.DataFrame, config: BacktestConfig | None = None) -> None:
        """Crea el backtester.

        Args:
            matches: Partidos finalizados (esquema estándar).
            odds: Momios históricos enlazados por ``match_id`` (puede incluir la línea de cierre).
            config: Configuración.

        Raises:
            InsufficientDataError: Si no hay partidos suficientes tras el calentamiento.
        """
        self.config = config or BacktestConfig()
        self.matches = matches[matches["status"] == STATUS_FINISHED].sort_values("date", kind="stable").reset_index(drop=True)
        if len(self.matches) <= self.config.warmup_matches:
            raise InsufficientDataError(
                f"Se requieren más de {self.config.warmup_matches} partidos; hay {len(self.matches)}")
        is_closing = odds["bookmaker"] == self.config.closing_bookmaker
        self.pre_odds = odds[~is_closing]
        self.closing_odds = odds[is_closing]
        self._closing_fair = ValueBetFinder(self.config.value).market_consensus(self.closing_odds)

    def _fit(self, train: pd.DataFrame, features: pd.DataFrame) -> EnsemblePredictor:
        """Entrena los modelos con los partidos de entrenamiento."""
        c = self.config
        poisson_model = PoissonModel(c.poisson).fit(train)
        ml: MLOutcomeModel | None = None
        if c.use_ml:
            try:
                ml = MLOutcomeModel().fit(features[features["match_id"].isin(train["match_id"])])
            except InsufficientDataError as exc:
                logger.debug("ML aún sin datos suficientes: %s", exc)
        return EnsemblePredictor(poisson_model, ml, None, c.ml_weight)

    def _clv(self, pick: Pick) -> float:
        """CLV = momio tomado × probabilidad justa de cierre − 1 (NaN si no hay cierre)."""
        p_close = self._closing_fair.get((pick.match_id, pick.selection))
        return pick.odds * p_close - 1.0 if p_close is not None else float("nan")

    def run(self) -> BacktestResult:
        """Ejecuta el backtest.

        Returns:
            :class:`BacktestResult` con apuestas, curva de bankroll y métricas.
        """
        c = self.config
        features = FeatureEngineer().fit_transform(self.matches)
        feat_by_id = features.set_index("match_id")
        finder = ValueBetFinder(c.value)
        builder = ParlayBuilder(c.parlay)
        bankroll = c.initial_bankroll
        bets: list[dict[str, object]] = []
        evaluated: list[dict[str, object]] = []
        curve: list[dict[str, object]] = []
        ensemble: EnsemblePredictor | None = None
        dates = self.matches["date"].drop_duplicates().tolist()
        n_seen = 0
        for i, day in enumerate(dates):
            today = self.matches[self.matches["date"] == day]
            train = self.matches[self.matches["date"] < day]
            if len(train) < c.warmup_matches:
                continue
            if ensemble is None or n_seen % c.refit_every == 0:
                ensemble = self._fit(train, features)
            n_seen += 1

            preds: dict[str, MatchPrediction] = {}
            for m in today.itertuples(index=False):
                feats = feat_by_id.loc[m.match_id].to_dict() if m.match_id in feat_by_id.index else None
                preds[str(m.match_id)] = ensemble.predict(str(m.home_team), str(m.away_team), str(m.match_id),
                                                          str(m.league), m.date, features=feats)
            day_odds = self.pre_odds[self.pre_odds["match_id"].isin(list(preds))]
            picks = finder.evaluate(preds, day_odds)
            scores = {str(m.match_id): (int(m.home_goals), int(m.away_goals)) for m in today.itertuples(index=False)}
            for p in picks:
                if p.selection.is_graded:  # la calibración binaria no aplica con push
                    continue
                evaluated.append({"date": day, "match_id": p.match_id, "market": p.selection.market.value,
                                  "model_prob": p.model_prob, "odds": p.odds,
                                  "won": selection_wins(p.selection, *scores[p.match_id])})
            singles = [p for p in picks if finder.is_value(p)]
            parlays = builder.build(picks, preds) if c.include_parlays else []
            singles = apply_match_cap(singles, c.max_match_exposure)
            singles, parlays, _ = apply_exposure_cap(singles, parlays, c.max_total_exposure)

            start_bankroll = bankroll
            for p in singles:
                stake = p.stake_fraction * start_bankroll
                if stake <= 0:
                    continue
                outcome = selection_result(p.selection, *scores[p.match_id])
                won = outcome > 0
                pnl = settlement_pnl(outcome, p.odds, stake)
                bankroll += pnl
                bets.append({"date": day, "match_id": p.match_id, "fixture": p.fixture, "type": "single",
                             "market": p.selection.market.value, "selection": p.selection.label, "odds": p.odds,
                             "model_prob": p.model_prob, "edge": p.adjusted_edge, "stake": stake, "won": won, "pnl": pnl,
                             "clv": self._clv(p)})
            for par in parlays:
                stake = par.stake_fraction * start_bankroll
                if stake <= 0:
                    continue
                won = all(selection_wins(leg.selection, *scores[leg.match_id]) for leg in par.legs)
                pnl = stake * (par.combined_odds - 1) if won else -stake
                bankroll += pnl
                clvs = [self._clv(leg) for leg in par.legs]
                bets.append({"date": day, "match_id": "+".join(leg.match_id for leg in par.legs),
                             "fixture": " + ".join(leg.fixture for leg in par.legs), "type": "parlay",
                             "market": f"PARLAY{len(par.legs)}", "selection": " + ".join(l.selection.label for l in par.legs),
                             "odds": par.combined_odds, "model_prob": par.joint_prob, "edge": par.expected_value,
                             "stake": stake, "won": won, "pnl": pnl,
                             "clv": float(np.prod([1 + x for x in clvs]) - 1) if not np.isnan(clvs).any() else float("nan")})
            curve.append({"date": day, "bankroll": bankroll, "bets": sum(1 for b in bets if b["date"] == day)})
            if bankroll <= 0:
                logger.warning("Bankroll agotado en %s", day)
                break
            logger.debug("Fecha %d/%d: bankroll %.2f", i + 1, len(dates), bankroll)

        bets_df = pd.DataFrame(bets, columns=["date", "match_id", "fixture", "type", "market", "selection", "odds",
                                              "model_prob", "edge", "stake", "won", "pnl", "clv"])
        curve_df = pd.DataFrame(curve, columns=["date", "bankroll", "bets"])
        evaluated_df = pd.DataFrame(evaluated, columns=["date", "match_id", "market", "model_prob", "odds", "won"])
        return BacktestResult(bets_df, curve_df, evaluated_df, self._metrics(bets_df, curve_df, evaluated_df))

    def _metrics(self, bets: pd.DataFrame, curve: pd.DataFrame, evaluated: pd.DataFrame) -> dict[str, float]:
        """Calcula las métricas agregadas del backtest."""
        c = self.config
        m: dict[str, float] = {"apuestas": float(len(bets)), "bankroll_inicial": c.initial_bankroll}
        final = float(curve["bankroll"].iloc[-1]) if not curve.empty else c.initial_bankroll
        m["bankroll_final"] = final
        m["crecimiento"] = final / c.initial_bankroll - 1
        if bets.empty:
            return m
        staked = float(bets["stake"].sum())
        m["total_apostado"] = staked
        m["beneficio"] = float(bets["pnl"].sum())
        m["roi"] = m["beneficio"] / staked if staked else 0.0
        m["acierto"] = float(bets["won"].mean())
        m["momio_medio"] = float(bets["odds"].mean())
        m["edge_esperado_medio"] = float(bets["edge"].mean())
        clv = bets["clv"].dropna()
        m["clv_medio"] = float(clv.mean()) if len(clv) else float("nan")
        m["clv_positivo_%"] = float((clv > 0).mean()) if len(clv) else float("nan")
        equity = np.r_[c.initial_bankroll, curve["bankroll"].to_numpy()]
        peaks = np.maximum.accumulate(equity)
        m["max_drawdown"] = float(np.max(1 - equity / peaks))
        if not evaluated.empty:
            p = evaluated["model_prob"].to_numpy(float)
            y = evaluated["won"].to_numpy(float)
            m["brier_selecciones"] = float(np.mean((p - y) ** 2))
        return m


__all__ = ["BacktestConfig", "BacktestResult", "Backtester", "selection_result", "selection_wins",
           "settlement_pnl"]
