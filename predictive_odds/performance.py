"""Análisis del rendimiento real del registro de apuestas: habilidad frente a suerte.

Para cada apuesta liquidada se compara el beneficio real con el **esperado**:

* si se conoce la probabilidad justa de cierre (CLV), el esperado usa esa
  probabilidad, que es la mejor estimación disponible de la probabilidad
  real (``fuente = "cierre"``);
* si no, usa la probabilidad del modelo (``fuente = "modelo"``, optimista).

La diferencia real − esperado es la **suerte**; se expresa en desviaciones
típicas (z). Un |z| < 2 es variación normal.

Aproximación: en DNB/hándicap asiático se ignora la probabilidad de push al
calcular el esperado y la varianza (efecto pequeño y conservador).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

SETTLED = ("won", "lost")


@dataclass(frozen=True)
class PerformanceSummary:
    """Resumen de habilidad frente a suerte.

    Attributes:
        bets: Apuestas liquidadas (ganadas o perdidas).
        staked: Total apostado.
        profit: Beneficio real.
        expected_profit: Beneficio esperado (cierre si existe, si no modelo).
        expected_from_closing: Parte del esperado calculada con el cierre.
        closing_coverage: Fracción de apuestas con probabilidad de cierre.
        luck: ``profit − expected_profit``.
        luck_z: Suerte en desviaciones típicas.
        longest_losing_streak: Racha perdedora más larga.
        current_streak: Racha actual (positiva = ganando, negativa = perdiendo).
    """

    bets: int
    staked: float
    profit: float
    expected_profit: float
    expected_from_closing: float
    closing_coverage: float
    luck: float
    luck_z: float
    longest_losing_streak: int
    current_streak: int

    @property
    def roi(self) -> float:
        """ROI real."""
        return self.profit / self.staked if self.staked else 0.0

    @property
    def expected_roi(self) -> float:
        """ROI esperado."""
        return self.expected_profit / self.staked if self.staked else 0.0

    def verdict(self) -> str:
        """Interpretación en una frase."""
        if self.bets < 30:
            return "Muestra pequeña: aún no se puede distinguir habilidad de suerte."
        skill = "con ventaja" if self.expected_roi > 0 else "sin ventaja"
        if abs(self.luck_z) < 2:
            luck = "y la suerte está dentro de lo normal"
        elif self.luck_z > 0:
            luck = "y con buena suerte por encima de lo normal (no durará)"
        else:
            luck = "y con mala suerte por encima de lo normal (debería corregirse)"
        return f"Según el cierre apuestas {skill} (ROI esperado {self.expected_roi:+.1%}) {luck}."


def _prepare(bets: pd.DataFrame) -> pd.DataFrame:
    """Apuestas liquidadas con mercado, probabilidad usada, esperado y varianza por apuesta."""
    df = bets[bets["status"].isin(SETTLED)].copy()
    if df.empty:
        return df.assign(market=pd.Series(dtype=str), source=pd.Series(dtype=str), p=pd.Series(dtype=float),
                         expected=pd.Series(dtype=float), variance=pd.Series(dtype=float))
    df["market"] = ["PARLAY" if t == "parlay" else json.loads(l)[0]["selection"].split("|")[0]
                    for t, l in zip(df["bet_type"], df["legs"])]
    closing = pd.to_numeric(df["closing_prob"], errors="coerce")
    has_close = closing.notna() & (df["bet_type"] == "single")
    df["source"] = np.where(has_close, "cierre", "modelo")
    df["p"] = np.where(has_close, closing, df["model_prob"]).astype(float).clip(1e-6, 1 - 1e-6)
    odds, stake = df["odds"].astype(float), df["stake"].astype(float)
    df["expected"] = stake * (df["p"] * odds - 1)
    df["variance"] = (stake * odds) ** 2 * df["p"] * (1 - df["p"])
    return df.sort_values(["settled_at", "id"], kind="stable")


def _streaks(won: list[bool]) -> tuple[int, int]:
    """(racha perdedora más larga, racha actual con signo)."""
    longest = run = 0
    for w in won:
        run = 0 if w else run + 1
        longest = max(longest, run)
    current = 0
    for w in reversed(won):
        if current == 0:
            current = 1 if w else -1
        elif (current > 0) == w:
            current += 1 if w else -1
        else:
            break
    return longest, current


def summarize(bets: pd.DataFrame) -> PerformanceSummary:
    """Resumen global de habilidad frente a suerte.

    Args:
        bets: Salida de :meth:`BetTracker.bets`.

    Returns:
        :class:`PerformanceSummary` (ceros si no hay apuestas liquidadas).
    """
    df = _prepare(bets)
    if df.empty:
        return PerformanceSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    profit, expected = float(df["pnl"].sum()), float(df["expected"].sum())
    sd = float(np.sqrt(df["variance"].sum()))
    longest, current = _streaks((df["status"] == "won").tolist())
    return PerformanceSummary(
        bets=len(df), staked=float(df["stake"].sum()), profit=profit, expected_profit=expected,
        expected_from_closing=float(df.loc[df["source"] == "cierre", "expected"].sum()),
        closing_coverage=float((df["source"] == "cierre").mean()), luck=profit - expected,
        luck_z=(profit - expected) / sd if sd > 0 else 0.0,
        longest_losing_streak=longest, current_streak=current,
    )


def breakdown(bets: pd.DataFrame, by: str) -> pd.DataFrame:
    """Rendimiento agrupado por ``"league"`` o ``"market"``.

    Returns:
        DataFrame con apuestas, apostado, beneficio, ROI real, ROI esperado,
        CLV medio, acierto y z de suerte por grupo.
    """
    df = _prepare(bets)
    cols = [by, "apuestas", "apostado", "beneficio", "roi", "roi_esperado", "clv_medio", "acierto", "z_suerte"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for key, g in df.groupby(by):
        staked, sd = float(g["stake"].sum()), float(np.sqrt(g["variance"].sum()))
        clv = pd.to_numeric(g["clv"], errors="coerce").dropna()
        rows.append({by: key, "apuestas": len(g), "apostado": staked, "beneficio": float(g["pnl"].sum()),
                     "roi": float(g["pnl"].sum()) / staked if staked else 0.0,
                     "roi_esperado": float(g["expected"].sum()) / staked if staked else 0.0,
                     "clv_medio": float(clv.mean()) if len(clv) else float("nan"),
                     "acierto": float((g["status"] == "won").mean()),
                     "z_suerte": float((g["pnl"].sum() - g["expected"].sum()) / sd) if sd > 0 else 0.0})
    return pd.DataFrame(rows, columns=cols).sort_values("apuestas", ascending=False).reset_index(drop=True)


def cumulative(bets: pd.DataFrame) -> pd.DataFrame:
    """Beneficio acumulado real frente al esperado, en orden de liquidación (para graficar)."""
    df = _prepare(bets)
    if df.empty:
        return pd.DataFrame(columns=["n", "real", "esperado"])
    return pd.DataFrame({"n": np.arange(1, len(df) + 1), "real": df["pnl"].cumsum().to_numpy(),
                         "esperado": df["expected"].cumsum().to_numpy()})


__all__ = ["PerformanceSummary", "breakdown", "cumulative", "summarize"]
