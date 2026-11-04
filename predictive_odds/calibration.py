"""Calibración de probabilidades y métricas de fiabilidad.

Un modelo con buena discriminación puede estar mal calibrado (p. ej. decir
60% en eventos que ocurren el 52% de las veces), lo que infla el EV y los
stakes. :class:`ProbabilityCalibrator` aprende un mapeo monótono
``p_modelo → p_calibrada`` por mercado a partir de predicciones fuera de muestra
(por ejemplo, las selecciones evaluadas en un backtest) y luego renormaliza
cada mercado para que sus resultados sumen 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .domain import MatchPrediction
from .statistical_models import ML_SELECTIONS

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float64]


def expected_calibration_error(probs: npt.ArrayLike, outcomes: npt.ArrayLike, n_bins: int = 10) -> float:
    """Error de calibración esperado (ECE) con tramos de igual ancho.

    Args:
        probs: Probabilidades predichas.
        outcomes: Resultados binarios (1 = ocurrió).
        n_bins: Número de tramos.

    Returns:
        ECE en [0, 1] (0 = calibración perfecta).

    Raises:
        ValueError: Si los tamaños no coinciden o no hay datos.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probs y outcomes deben tener el mismo tamaño no nulo")
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def reliability_table(probs: npt.ArrayLike, outcomes: npt.ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Datos del diagrama de fiabilidad (listos para graficar en la UI).

    Args:
        probs: Probabilidades predichas.
        outcomes: Resultados binarios.
        n_bins: Número de tramos.

    Returns:
        DataFrame con ``tramo_inf, tramo_sup, n, p_media, frecuencia_real``.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            rows.append({"tramo_inf": b / n_bins, "tramo_sup": (b + 1) / n_bins, "n": int(mask.sum()),
                         "p_media": float(p[mask].mean()), "frecuencia_real": float(y[mask].mean())})
    return pd.DataFrame(rows, columns=["tramo_inf", "tramo_sup", "n", "p_media", "frecuencia_real"])


@dataclass
class ProbabilityCalibrator:
    """Calibrador por mercado (isotónico o Platt).

    Attributes:
        method: ``isotonic`` (no paramétrico, necesita más datos) o ``platt``
            (logístico sobre el logit, robusto con pocos datos).
        min_samples: Muestras mínimas por mercado para calibrarlo.
    """

    method: Literal["isotonic", "platt"] = "isotonic"
    min_samples: int = 200
    _models: dict[str, IsotonicRegression | LogisticRegression] = field(default_factory=dict, repr=False)

    @property
    def is_fitted(self) -> bool:
        """Indica si hay al menos un mercado calibrado."""
        return bool(self._models)

    @property
    def markets(self) -> list[str]:
        """Mercados calibrados."""
        return sorted(self._models)

    @staticmethod
    def _logit(p: FloatArray) -> FloatArray:
        """Logit acotado numéricamente."""
        q = np.clip(p, 1e-6, 1 - 1e-6)
        return np.asarray(np.log(q / (1 - q)), dtype=float).reshape(-1, 1)

    def fit(self, probs: npt.ArrayLike, outcomes: npt.ArrayLike, markets: npt.ArrayLike) -> ProbabilityCalibrator:
        """Ajusta un mapeo por mercado.

        Args:
            probs: Probabilidades del modelo (fuera de muestra).
            outcomes: Resultados binarios.
            markets: Código de mercado de cada fila (``"1X2"``, ``"OU"``, ``"BTTS"``).

        Returns:
            ``self``.

        Raises:
            ValueError: Si los tamaños no coinciden.
        """
        p = np.asarray(probs, dtype=float)
        y = np.asarray(outcomes, dtype=float)
        mk = np.asarray(markets).astype(str)
        if not (p.shape == y.shape == mk.shape):
            raise ValueError("probs, outcomes y markets deben tener el mismo tamaño")
        self._models = {}
        for market in np.unique(mk):
            mask = mk == market
            if mask.sum() < self.min_samples or len(np.unique(y[mask])) < 2:
                logger.info("Mercado %s sin datos suficientes para calibrar (%d)", market, mask.sum())
                continue
            if self.method == "platt":
                model: IsotonicRegression | LogisticRegression = LogisticRegression().fit(self._logit(p[mask]), y[mask])
            else:
                model = IsotonicRegression(y_min=0.005, y_max=0.995, out_of_bounds="clip").fit(p[mask], y[mask])
            self._models[str(market)] = model
        return self

    def transform(self, probs: npt.ArrayLike, market: str) -> FloatArray:
        """Calibra probabilidades de un mercado (sin cambios si no está calibrado).

        Args:
            probs: Probabilidades.
            market: Código de mercado.

        Returns:
            Probabilidades calibradas.
        """
        p = np.asarray(probs, dtype=float)
        model = self._models.get(market)
        if model is None:
            return p
        if isinstance(model, LogisticRegression):
            return np.asarray(model.predict_proba(self._logit(p))[:, 1], dtype=float)
        return np.asarray(model.predict(p), dtype=float)

    def calibrate_prediction(self, pred: MatchPrediction) -> MatchPrediction:
        """Calibra in situ los mercados principales de una predicción.

        Cada grupo (1X2, OU 2.5, BTTS) se calibra y se renormaliza para sumar 1;
        el resultado se guarda en ``overrides`` (la matriz se conserva para las
        correlaciones).

        Args:
            pred: Predicción a calibrar.

        Returns:
            La misma predicción (modificada).
        """
        if not self.is_fitted:
            return pred
        for sels in ML_SELECTIONS.values():
            market = sels[0].market.value
            if market not in self._models:
                continue
            raw = np.array([pred.probability(s) for s in sels])
            cal = self.transform(raw, market)
            total = cal.sum()
            if total <= 0:
                continue
            pred.overrides.update({s: float(v / total) for s, v in zip(sels, cal)})
        pred.sources["calibrated"] = {s: pred.overrides[s] for sels in ML_SELECTIONS.values()
                                      for s in sels if s in pred.overrides}
        return pred


def calibration_report(evaluated: pd.DataFrame, calibrator: ProbabilityCalibrator | None = None,
                       train_fraction: float = 0.5) -> dict[str, float]:
    """Mide el efecto de la calibración con una partición cronológica.

    Args:
        evaluated: Selecciones evaluadas (``date``, ``market``, ``model_prob``, ``won``).
        calibrator: Calibrador a usar (por defecto isotónico).
        train_fraction: Fracción inicial usada para ajustar el calibrador.

    Returns:
        Brier y ECE antes/después sobre la parte de prueba.

    Raises:
        ValueError: Si no hay datos suficientes.
    """
    ev = evaluated.sort_values("date").reset_index(drop=True)
    cut = int(len(ev) * train_fraction)
    train, test = ev.iloc[:cut], ev.iloc[cut:]
    if len(test) < 50:
        raise ValueError("Datos insuficientes para evaluar la calibración")
    cal = (calibrator or ProbabilityCalibrator()).fit(train["model_prob"], train["won"].astype(float), train["market"])
    p_raw = test["model_prob"].to_numpy(float)
    y = test["won"].to_numpy(float)
    p_cal = np.concatenate([cal.transform(g["model_prob"].to_numpy(float), str(m)) for m, g in test.groupby("market")])
    y_cal = np.concatenate([g["won"].to_numpy(float) for _, g in test.groupby("market")])
    return {
        "brier_antes": float(np.mean((p_raw - y) ** 2)), "brier_despues": float(np.mean((p_cal - y_cal) ** 2)),
        "ece_antes": expected_calibration_error(p_raw, y), "ece_despues": expected_calibration_error(p_cal, y_cal),
        "n_prueba": float(len(test)),
    }


__all__ = ["ProbabilityCalibrator", "calibration_report", "expected_calibration_error", "reliability_table"]
