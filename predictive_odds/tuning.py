"""Ajuste de hiperparámetros con validación cronológica (*rolling origin*).

Para cada corte temporal se entrena con el pasado y se evalúa en las
siguientes semanas: nunca se usa información del futuro. La métrica es el
log-loss medio de 1X2, Over/Under 2.5 y BTTS.

.. note::
   Ajustar sobre datos SIMULADOS solo valida la herramienta: favorecería lo
   que la simulación hace por construcción (p. ej. su xG es muy informativo).
   Los valores por defecto deben cambiarse solo con resultados sobre datos reales.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .domain import STATUS_FINISHED, Selection
from .statistical_models import (
    FeatureEngineer,
    InsufficientDataError,
    MLOutcomeModel,
    PoissonModel,
    PoissonModelConfig,
)

logger = logging.getLogger(__name__)

#: Grupos de selecciones evaluados (cada grupo es un mercado completo).
EVAL_MARKETS: dict[str, list[Selection]] = {
    "1x2": [Selection.home(), Selection.draw(), Selection.away()],
    "ou25": [Selection.over(2.5), Selection.under(2.5)],
    "btts": [Selection.btts(True), Selection.btts(False)],
}


@dataclass(frozen=True)
class Split:
    """Partición temporal: entrenamiento antes de ``cutoff`` y prueba en ``[cutoff, end)``."""

    cutoff: pd.Timestamp
    end: pd.Timestamp


@dataclass
class TuningResult:
    """Resultado de un ajuste.

    Attributes:
        table: Una fila por configuración con sus métricas (ordenada, mejor primero).
        best: Parámetros de la mejor configuración.
        baseline_score: Puntuación de la configuración por defecto (referencia).
    """

    table: pd.DataFrame
    best: dict[str, float]
    baseline_score: float

    @property
    def improvement(self) -> float:
        """Mejora relativa del log-loss frente a la configuración por defecto (>0 = mejor)."""
        best_score = float(self.table["score"].to_numpy(float)[0])
        return (self.baseline_score - best_score) / self.baseline_score if self.baseline_score else 0.0


def rolling_origin_splits(matches: pd.DataFrame, n_folds: int = 6, horizon_days: int = 28,
                          min_train: int = 150) -> list[Split]:
    """Genera cortes temporales repartidos en la parte final del histórico.

    Args:
        matches: Partidos (se usan los finalizados).
        n_folds: Número de cortes.
        horizon_days: Días evaluados tras cada corte.
        min_train: Partidos mínimos antes del primer corte.

    Returns:
        Lista de :class:`Split`.

    Raises:
        InsufficientDataError: Si no hay datos suficientes.
    """
    m = matches[matches["status"] == STATUS_FINISHED].sort_values("date")
    if len(m) <= min_train:
        raise InsufficientDataError(f"Se requieren más de {min_train} partidos para validar")
    first = m["date"].iloc[min_train]
    last = m["date"].iloc[-1] - pd.Timedelta(days=horizon_days)
    if last <= first:
        last = first
    cutoffs = pd.date_range(first, last, periods=max(1, n_folds))
    return [Split(c, c + pd.Timedelta(days=horizon_days)) for c in cutoffs]


def _log_losses(probs: dict[str, np.ndarray], outcomes: dict[str, np.ndarray]) -> dict[str, float]:
    """Log-loss por mercado."""
    out = {}
    for k, p in probs.items():
        idx = outcomes[k]
        out[f"ll_{k}"] = float(-np.mean(np.log(np.clip(p[np.arange(len(idx)), idx], 1e-12, 1))))
    out["score"] = float(np.mean([out[f"ll_{k}"] for k in probs]))
    return out


def _outcome_index(hg: np.ndarray, ag: np.ndarray) -> dict[str, np.ndarray]:
    """Índice del resultado real en cada grupo de ``EVAL_MARKETS``."""
    return {
        "1x2": np.where(hg > ag, 0, np.where(hg == ag, 1, 2)),
        "ou25": np.where(hg + ag > 2.5, 0, 1),
        "btts": np.where((hg > 0) & (ag > 0), 0, 1),
    }


def evaluate_poisson_config(matches: pd.DataFrame, config: PoissonModelConfig,
                            splits: Sequence[Split]) -> dict[str, float]:
    """Log-loss fuera de muestra de una configuración de Poisson.

    Args:
        matches: Partidos finalizados.
        config: Configuración a evaluar.
        splits: Cortes temporales.

    Returns:
        ``ll_1x2``, ``ll_ou25``, ``ll_btts``, ``score`` (media) y ``n``.
    """
    m = matches[matches["status"] == STATUS_FINISHED].sort_values("date")
    probs: dict[str, list[list[float]]] = {k: [] for k in EVAL_MARKETS}
    hg_all: list[float] = []
    ag_all: list[float] = []
    for sp in splits:
        train = m[m["date"] < sp.cutoff]
        test = m[(m["date"] >= sp.cutoff) & (m["date"] < sp.end)]
        if test.empty:
            continue
        model = PoissonModel(config).fit(train, reference_date=sp.cutoff)
        for r in test.itertuples(index=False):
            pred = model.predict(str(r.home_team), str(r.away_team))
            for k, sels in EVAL_MARKETS.items():
                probs[k].append([pred.matrix_probability(s) for s in sels])
        hg_all.extend(test["home_goals"].to_numpy(float).tolist())
        ag_all.extend(test["away_goals"].to_numpy(float).tolist())
    if not hg_all:
        raise InsufficientDataError("Ningún partido en los periodos de prueba")
    res = _log_losses({k: np.array(v) for k, v in probs.items()}, _outcome_index(np.array(hg_all), np.array(ag_all)))
    res["n"] = float(len(hg_all))
    return res


def tune_poisson(matches: pd.DataFrame, grid: Mapping[str, Sequence[float]],
                 base: PoissonModelConfig | None = None, splits: Sequence[Split] | None = None) -> TuningResult:
    """Búsqueda en rejilla de hiperparámetros de :class:`PoissonModelConfig`.

    Args:
        matches: Partidos finalizados.
        grid: ``parámetro -> valores`` (p. ej. ``{"half_life_days": [120, 240, 480]}``).
        base: Configuración base (por defecto la actual).
        splits: Cortes temporales (por defecto :func:`rolling_origin_splits`).

    Returns:
        :class:`TuningResult` ordenado por ``score`` (menor es mejor).

    Raises:
        ValueError: Si algún parámetro no existe en la configuración.
    """
    base = base or PoissonModelConfig()
    unknown = set(grid) - set(base.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Parámetros desconocidos: {sorted(unknown)}")
    splits = list(splits) if splits is not None else rolling_origin_splits(matches)
    baseline = evaluate_poisson_config(matches, base, splits)["score"]
    rows = []
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params: dict[str, Any] = dict(zip(keys, values))
        res = evaluate_poisson_config(matches, replace(base, **params), splits)
        rows.append({**params, **res})
        logger.info("%s -> %.5f", params, res["score"])
    table = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    best = {k: float(table[k].to_numpy(float)[0]) for k in keys}
    return TuningResult(table, best, baseline)


def tune_ml_weight(matches: pd.DataFrame, weights: Sequence[float], config: PoissonModelConfig | None = None,
                   splits: Sequence[Split] | None = None) -> TuningResult:
    """Ajusta el peso del ML en el ensamble (1X2, OU 2.5 y BTTS).

    Args:
        matches: Partidos finalizados.
        weights: Pesos a probar (0 = solo Poisson).
        config: Configuración de Poisson.
        splits: Cortes temporales.

    Returns:
        :class:`TuningResult` (la referencia es el peso 0.35 por defecto).
    """
    config = config or PoissonModelConfig()
    m = matches[matches["status"] == STATUS_FINISHED].sort_values("date").reset_index(drop=True)
    splits = list(splits) if splits is not None else rolling_origin_splits(m, min_train=300)
    feats = FeatureEngineer().fit_transform(m).set_index("match_id")
    p_pois: dict[str, list[list[float]]] = {k: [] for k in EVAL_MARKETS}
    p_ml: dict[str, list[list[float]]] = {k: [] for k in EVAL_MARKETS}
    hg_all: list[float] = []
    ag_all: list[float] = []
    for sp in splits:
        train = m[m["date"] < sp.cutoff]
        test = m[(m["date"] >= sp.cutoff) & (m["date"] < sp.end)]
        if test.empty:
            continue
        try:
            ml = MLOutcomeModel().fit(feats.loc[train["match_id"]].reset_index())
        except InsufficientDataError as exc:
            logger.info("Corte %s omitido: %s", sp.cutoff.date(), exc)
            continue
        pois = PoissonModel(config).fit(train, reference_date=sp.cutoff)
        for r in test.itertuples(index=False):
            pred = pois.predict(str(r.home_team), str(r.away_team))
            mlp = ml.predict_proba(feats.loc[[r.match_id]])
            for k, sels in EVAL_MARKETS.items():
                p_pois[k].append([pred.matrix_probability(s) for s in sels])
                p_ml[k].append([mlp[s] for s in sels])
        hg_all.extend(test["home_goals"].to_numpy(float).tolist())
        ag_all.extend(test["away_goals"].to_numpy(float).tolist())
    if not hg_all:
        raise InsufficientDataError("Ningún corte con datos suficientes para el ML")
    outcomes = _outcome_index(np.array(hg_all), np.array(ag_all))
    rows = []
    for w in weights:
        blended = {}
        for k in EVAL_MARKETS:
            b = (1 - w) * np.array(p_pois[k]) + w * np.array(p_ml[k])
            blended[k] = b / b.sum(axis=1, keepdims=True)
        rows.append({"ml_weight": float(w), **_log_losses(blended, outcomes), "n": float(len(hg_all))})
    table = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    w_arr, sc_arr = table["ml_weight"].to_numpy(float), table["score"].to_numpy(float)
    ref = float(sc_arr[int(np.argmin(np.abs(w_arr - 0.35)))])
    return TuningResult(table, {"ml_weight": float(w_arr[0])}, ref)


__all__ = ["EVAL_MARKETS", "Split", "TuningResult", "evaluate_poisson_config", "rolling_origin_splits",
           "tune_ml_weight", "tune_poisson"]
