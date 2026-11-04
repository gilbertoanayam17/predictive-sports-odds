"""Simulación Monte Carlo del bankroll: riesgo de ruina, drawdown y crecimiento.

Responde a "¿qué pasa si repito esta estrategia N jornadas?" bajo distintos
supuestos de calidad del modelo. El parámetro clave es ``edge_haircut``:

* 0.0 → el modelo acierta exactamente (``p_real = p_decisión``);
* 0.5 → el edge real es la mitad del estimado;
* 1.0 → el modelo no aporta: ``p_real`` = probabilidad justa de consenso del
  mercado (puede quedar algo de EV por apostar al MEJOR precio entre casas);
* 1.5 → escenario pesimista: el modelo se equivoca en contra (``p_real`` por
  debajo del consenso en la mitad de lo que el modelo creía estar por encima).

Supuesto: las apuestas de una jornada se tratan como independientes (las de un
mismo partido están correlacionadas en la realidad, lo que aumenta la varianza).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd

from .parlay_builder import KellyConfig, Pick, modified_kelly


@dataclass(frozen=True)
class RiskConfig:
    """Parámetros de la simulación.

    Attributes:
        n_rounds: Jornadas simuladas (p. ej. una temporada).
        n_sims: Número de trayectorias.
        initial_bankroll: Bankroll inicial.
        max_total_exposure: Tope de exposición por jornada.
        ruin_level: Fracción del bankroll inicial considerada "ruina".
        seed: Semilla.
    """

    n_rounds: int = 38
    n_sims: int = 5000
    initial_bankroll: float = 1000.0
    max_total_exposure: float = 0.10
    ruin_level: float = 0.5
    seed: int = 0


def _true_prob(pick: Pick, haircut: float) -> float:
    """Probabilidad "real" supuesta para un pick según el recorte del edge."""
    p_dec = pick.decision_prob if pick.decision_prob is not None else pick.model_prob
    anchor = pick.market_prob if pick.market_prob is not None else 1.0 / pick.odds
    return float(np.clip(anchor + (1.0 - haircut) * (p_dec - anchor), 1e-6, 1 - 1e-6))


def stake_fractions(picks: Sequence[Pick], kelly: KellyConfig, max_total_exposure: float) -> np.ndarray:
    """Stakes (fracción del bankroll) con Kelly modificado y tope de exposición.

    Args:
        picks: Apuestas de una jornada.
        kelly: Parámetros de Kelly.
        max_total_exposure: Tope total.

    Returns:
        Vector de fracciones.
    """
    f = np.array([modified_kelly(p.model_prob, p.odds, p.market_prob, kelly) for p in picks], dtype=float)
    total = f.sum()
    return f * (max_total_exposure / total) if total > max_total_exposure else f


def simulate_bankroll(picks: Sequence[Pick], kelly: KellyConfig = KellyConfig(), edge_haircut: float = 0.0,
                      config: RiskConfig = RiskConfig()) -> dict[str, float]:
    """Simula repetir una jornada tipo durante ``n_rounds`` jornadas.

    Args:
        picks: Jornada tipo (p. ej. los value bets de un análisis).
        kelly: Parámetros de staking.
        edge_haircut: Fracción del edge estimado que NO es real (0-2; >1 = modelo en contra).
        config: Parámetros de la simulación.

    Returns:
        Métricas: percentiles del bankroll final, crecimiento mediano,
        P(pérdida), P(ruina), drawdown máximo mediano y P(drawdown > 30%).

    Raises:
        ValueError: Si no hay picks o el recorte está fuera de [0, 1].
    """
    if not picks:
        raise ValueError("Se requiere al menos un pick")
    if not 0.0 <= edge_haircut <= 2.0:
        raise ValueError("edge_haircut debe estar en [0, 2]")
    rng = np.random.default_rng(config.seed)
    p = np.array([_true_prob(pk, edge_haircut) for pk in picks])
    odds = np.array([pk.odds for pk in picks])
    f = stake_fractions(picks, kelly, config.max_total_exposure)
    bank = np.full(config.n_sims, config.initial_bankroll)
    peak = bank.copy()
    max_dd = np.zeros(config.n_sims)
    ruined = np.zeros(config.n_sims, dtype=bool)
    for _ in range(config.n_rounds):
        wins = rng.uniform(size=(config.n_sims, len(p))) < p
        returns = np.where(wins, f * (odds - 1), -f).sum(axis=1)
        bank = bank * (1 + returns)
        peak = np.maximum(peak, bank)
        max_dd = np.maximum(max_dd, 1 - bank / peak)
        ruined |= bank < config.ruin_level * config.initial_bankroll
    final = bank / config.initial_bankroll
    exp_round = float(np.sum(f * (p * odds - 1)))
    return {
        "ev_por_jornada": exp_round,
        "p05_final": float(np.percentile(final, 5)),
        "mediana_final": float(np.median(final)),
        "p95_final": float(np.percentile(final, 95)),
        "p_perdida": float(np.mean(final < 1)),
        "p_ruina": float(np.mean(ruined)),
        "dd_mediano": float(np.median(max_dd)),
        "p_dd_30": float(np.mean(max_dd > 0.30)),
    }


def risk_table(picks: Sequence[Pick], fractions: Sequence[float] = (0.1, 0.25, 0.5, 1.0),
               haircuts: Sequence[float] = (0.0, 0.5, 1.0, 1.5), config: RiskConfig = RiskConfig(),
               base: KellyConfig = KellyConfig()) -> pd.DataFrame:
    """Matriz de escenarios: fracción de Kelly × recorte del edge.

    Para comparar el efecto puro de la fracción de Kelly, el tope por apuesta
    (``max_stake``) y el tope de exposición por jornada escalan en proporción
    a la fracción (si no, el tope del 10% igualaría todos los escenarios).

    Args:
        picks: Jornada tipo.
        fractions: Fracciones de Kelly a comparar.
        haircuts: Recortes del edge a comparar.
        config: Parámetros de la simulación.
        base: Configuración de Kelly de referencia.

    Returns:
        DataFrame con una fila por escenario.
    """
    rows = []
    for frac in fractions:
        scale = frac / base.fraction
        kelly = replace(base, fraction=frac, max_stake=min(1.0, base.max_stake * scale))
        cfg = replace(config, max_total_exposure=min(1.0, config.max_total_exposure * scale))
        for h in haircuts:
            rows.append({"kelly": frac, "exposicion_max": cfg.max_total_exposure, "recorte_edge": h,
                         **simulate_bankroll(picks, kelly, h, cfg)})
    return pd.DataFrame(rows)


__all__ = ["RiskConfig", "risk_table", "simulate_bankroll", "stake_fractions"]
