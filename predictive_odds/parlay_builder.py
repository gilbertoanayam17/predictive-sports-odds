"""Motor de apuestas: detección de valor, parlays inteligentes y staking.

Flujo:

1. :class:`ValueBetFinder` compara la probabilidad del modelo contra la
   probabilidad implícita de cada momio (mejor precio entre casas). Si
   ``p_modelo > 1/momio`` (con un margen mínimo) es una apuesta de valor.
2. :class:`ParlayBuilder` toma los picks más probables con EV no negativo y
   genera combinadas:

   * descarta combinaciones contradictorias (intersección de marcadores vacía)
     y redundantes (una selección implica a la otra);
   * calcula la probabilidad conjunta exacta para selecciones del mismo
     partido (matriz de Poisson) y asume independencia entre partidos;
   * sugiere el stake con Kelly modificado (fraccional + encogimiento hacia el
     mercado + penalización por número de patas + tope).
3. :func:`apply_exposure_cap` limita la exposición total del bankroll.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field, replace
from typing import Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import brentq

from .domain import MARKET_SIZE, Market, MatchPrediction, Selection, selections_compatible

logger = logging.getLogger(__name__)


# =============================================================================
# Matemática de momios
# =============================================================================
def implied_probability(odds: float) -> float:
    """Probabilidad implícita de un momio decimal.

    Args:
        odds: Momio decimal (> 1).

    Returns:
        ``1 / odds``.

    Raises:
        ValueError: Si el momio no es mayor que 1.
    """
    if not odds > 1.0 or not math.isfinite(odds):
        raise ValueError(f"Momio decimal inválido: {odds}")
    return 1.0 / odds


def remove_margin(odds: Sequence[float], method: Literal["multiplicative", "power"] = "multiplicative") -> list[float]:
    """Quita el margen de la casa a un mercado completo.

    Args:
        odds: Momios de todos los resultados del mercado.
        method: ``multiplicative`` (normaliza) o ``power`` (corrige mejor el
            sesgo favorito-longshot: busca ``k`` con ``Σ (1/o)^k = 1``).

    Returns:
        Probabilidades "justas" que suman 1.

    Raises:
        ValueError: Si hay menos de dos momios o alguno es inválido.
    """
    if len(odds) < 2:
        raise ValueError("Se requieren al menos dos resultados para quitar el margen")
    implied = np.array([implied_probability(o) for o in odds])
    if method == "power":
        overround = implied.sum()
        if abs(overround - 1.0) < 1e-9:
            return implied.tolist()
        try:
            k = brentq(lambda kk: np.sum(implied ** kk) - 1.0, 0.2, 5.0)
            fair = implied ** k
            return (fair / fair.sum()).tolist()
        except ValueError:
            logger.warning("Método 'power' no convergió; se usa el multiplicativo")
    return (implied / implied.sum()).tolist()


def expected_value(probability: float, odds: float) -> float:
    """Valor esperado por unidad apostada: ``p·momio − 1``."""
    return probability * odds - 1.0


def kelly_fraction(probability: float, odds: float) -> float:
    """Fracción de Kelly completa ``(b·p − q) / b`` (0 si no hay ventaja).

    Args:
        probability: Probabilidad estimada de ganar.
        odds: Momio decimal.

    Returns:
        Fracción del bankroll en [0, 1).
    """
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (b * probability - (1.0 - probability)) / b)


@dataclass(frozen=True)
class KellyConfig:
    """Parámetros del criterio de Kelly modificado.

    Attributes:
        fraction: Fracción de Kelly (0.25 = "cuarto de Kelly").
        model_trust: Peso del modelo frente a la probabilidad justa del
            mercado (encogimiento bayesiano contra el error de estimación).
        max_stake: Tope por apuesta como fracción del bankroll.
        leg_penalty: Factor multiplicativo por cada pata adicional de un parlay.
    """

    fraction: float = 0.25
    model_trust: float = 0.6
    max_stake: float = 0.03
    leg_penalty: float = 0.8


def modified_kelly(probability: float, odds: float, market_probability: float | None = None,
                   config: KellyConfig = KellyConfig(), legs: int = 1) -> float:
    """Stake sugerido (fracción del bankroll) con Kelly modificado.

    1. Encoge la probabilidad del modelo hacia la del mercado
       (``p = t·p_modelo + (1−t)·p_mercado``).
    2. Aplica Kelly fraccional.
    3. Penaliza cada pata extra de un parlay (``leg_penalty^(legs−1)``),
       porque el error del modelo se compone multiplicativamente.
    4. Limita el resultado a ``max_stake``.

    Args:
        probability: Probabilidad del modelo.
        odds: Momio decimal.
        market_probability: Probabilidad justa del mercado (sin margen).
        config: Parámetros.
        legs: Número de patas (1 para sencillas).

    Returns:
        Fracción del bankroll en [0, max_stake].
    """
    p = probability
    if market_probability is not None and 0 < market_probability < 1:
        p = config.model_trust * probability + (1 - config.model_trust) * market_probability
    f = kelly_fraction(p, odds) * config.fraction * (config.leg_penalty ** max(0, legs - 1))
    return float(min(max(f, 0.0), config.max_stake))


# =============================================================================
# Picks
# =============================================================================
@dataclass(frozen=True)
class Pick:
    """Una selección apostable con su precio y la opinión del modelo.

    Attributes:
        match_id: ID del partido.
        league: Liga.
        home_team: Local.
        away_team: Visitante.
        kickoff: Fecha/hora.
        selection: Selección.
        odds: Mejor momio disponible.
        bookmaker: Casa que ofrece ese momio.
        model_prob: Probabilidad del modelo.
        market_prob: Probabilidad justa de consenso (sin margen), si se conoce.
        stake_fraction: Stake sugerido como fracción del bankroll.
        decision_prob: Probabilidad usada para decidir el valor (modelo + mercado).
        action_share: Fracción del stake en juego (1 − prob. de push); 1 en
            mercados binarios. En DNB/hándicap asiático, ``model_prob`` es la
            probabilidad efectiva ``W/(W+L)`` y el EV es ``s·(p·momio − 1)``.
    """

    match_id: str
    league: str
    home_team: str
    away_team: str
    kickoff: pd.Timestamp | None
    selection: Selection
    odds: float
    bookmaker: str
    model_prob: float
    market_prob: float | None = None
    stake_fraction: float = 0.0
    decision_prob: float | None = None
    action_share: float = 1.0

    @property
    def adjusted_edge(self) -> float:
        """EV por unidad con la probabilidad de decisión (modelo encogido hacia el mercado)."""
        p = self.decision_prob if self.decision_prob is not None else self.model_prob
        return self.action_share * expected_value(p, self.odds)

    @property
    def implied_prob(self) -> float:
        """Probabilidad implícita del momio (con margen)."""
        return 1.0 / self.odds

    @property
    def edge(self) -> float:
        """Valor esperado por unidad (``s·(p·momio − 1)``; ``s = 1`` en mercados binarios)."""
        return self.action_share * expected_value(self.model_prob, self.odds)

    @property
    def fixture(self) -> str:
        """Texto ``Local vs Visitante``."""
        return f"{self.home_team} vs {self.away_team}"

    def describe(self) -> str:
        """Descripción breve del pick."""
        return f"{self.fixture} — {self.selection.label} @ {self.odds:.2f} ({self.bookmaker})"

    def to_dict(self, bankroll: float | None = None) -> dict[str, object]:
        """Representación plana (para tablas/UI)."""
        d: dict[str, object] = {
            "partido": self.fixture, "selección": self.selection.label, "momio": self.odds,
            "casa": self.bookmaker, "p_modelo": round(self.model_prob, 4),
            "p_implícita": round(self.implied_prob, 4),
            "p_mercado": round(self.market_prob, 4) if self.market_prob is not None else None,
            "EV_modelo": round(self.edge, 4), "EV_ajustado": round(self.adjusted_edge, 4),
            "stake_%": round(100 * self.stake_fraction, 2),
        }
        if bankroll is not None:
            d["stake"] = round(self.stake_fraction * bankroll, 2)
        return d


@dataclass(frozen=True)
class ValueBetConfig:
    """Filtros de la detección de valor.

    Attributes:
        min_edge: EV mínimo para considerar valor (0.04 = 4%).
        min_odds: Momio mínimo.
        max_odds: Momio máximo (evita longshots con alta varianza de modelo).
        min_model_prob: Probabilidad mínima del modelo.
        max_market_gap: Diferencia máxima tolerada entre modelo y mercado; si
            se supera suele indicar información que el modelo no tiene
            (lesiones, rotaciones) y el pick se descarta.
        devig_method: Método para quitar el margen.
        allowed_markets: Mercados permitidos (``"1X2"``, ``"OU"``, ``"AH"``...); ``None`` = todos.
            Lo fija el estudio por liga (:mod:`predictive_odds.league_study`).
        value_model_weight: Peso del modelo en la probabilidad usada para
            DECIDIR si hay valor (``w·p_modelo + (1−w)·p_mercado``). Con 1.0 se usa
            el modelo puro; valores < 1 corrigen la "maldición del ganador"
            (el filtro selecciona justo donde el modelo más se equivoca).
            Backtest simulado (4 semillas): 0.5 sube el EV real de 3.8% a 5.3%
            y el CLV de 2.6% a 4.0% frente a 1.0.
        kelly: Parámetros de staking.
    """

    min_edge: float = 0.04
    min_odds: float = 1.20
    max_odds: float = 8.0
    min_model_prob: float = 0.10
    max_market_gap: float = 0.15
    devig_method: Literal["multiplicative", "power"] = "power"
    value_model_weight: float = 0.5
    allowed_markets: tuple[str, ...] | None = None
    kelly: KellyConfig = field(default_factory=KellyConfig)


class ValueBetFinder:
    """Compara probabilidades del modelo contra momios del mercado."""

    def __init__(self, config: ValueBetConfig | None = None) -> None:
        """Crea el buscador.

        Args:
            config: Filtros y parámetros de staking.
        """
        self.config = config or ValueBetConfig()

    @staticmethod
    def _selection(row: pd.Series) -> Selection | None:
        """Convierte una fila de momios en ``Selection`` (None si es inválida)."""
        try:
            return Selection.from_codes(str(row["market"]), str(row["selection"]), row["line"])
        except (ValueError, TypeError):
            return None

    def market_consensus(self, odds: pd.DataFrame) -> dict[tuple[str, Selection], float]:
        """Probabilidad justa de consenso por (partido, selección).

        Para cada casa con el mercado completo se quita el margen y se
        promedian las probabilidades entre casas.

        Args:
            odds: Momios validados.

        Returns:
            Mapa ``(match_id, selection) -> probabilidad justa``.
        """
        if odds.empty:
            return {}
        # Implementación vectorizada (numpy): es la parte más costosa del backtest.
        market = odds["market"].astype(str).to_numpy()
        outcome = odds["selection"].astype(str).to_numpy()
        line = pd.to_numeric(odds["line"], errors="coerce").to_numpy(float)
        price = pd.to_numeric(odds["odds"], errors="coerce").to_numpy(float)
        # En el hándicap asiático las dos caras tienen líneas opuestas (local −0.5 / visitante +0.5):
        # se agrupan por la línea vista desde el local.
        away_ah = (market == Market.ASIAN_HANDICAP.value) & (outcome == "AWAY")
        line_key = np.where(np.isnan(line), -1.0, np.where(away_ah, -line, line))
        frame = pd.DataFrame({"mid": odds["match_id"].astype(str).to_numpy(), "book": odds["bookmaker"].astype(str).to_numpy(),
                              "mkt": market, "lk": line_key, "sel": outcome, "line": line, "odds": price})
        frame = frame[np.isfinite(frame["odds"].to_numpy()) & (frame["odds"].to_numpy() > 1.0)]
        frame = frame.drop_duplicates(subset=["mid", "book", "mkt", "lk", "sel"])
        if frame.empty:
            return {}
        codes, _ = pd.factorize(pd.MultiIndex.from_frame(frame[["mid", "book", "mkt", "lk"]]))
        n_groups = int(codes.max()) + 1
        sizes = np.bincount(codes, minlength=n_groups)
        size_by_code = {mk.value: n for mk, n in MARKET_SIZE.items()}
        expected = frame["mkt"].map(size_by_code).fillna(-1).to_numpy(int)
        keep = sizes[codes] == expected
        frame, codes = frame[keep], codes[keep]
        if frame.empty:
            return {}
        implied = 1.0 / frame["odds"].to_numpy(float)
        total = np.bincount(codes, implied, minlength=n_groups)
        if self.config.devig_method == "power":
            lo: npt.NDArray[np.float64] = np.full(n_groups, 0.2)
            hi: npt.NDArray[np.float64] = np.full(n_groups, 5.0)
            for _ in range(60):  # bisección vectorizada de k: Σ imp^k = 1 (Σ decrece con k)
                mid = (lo + hi) / 2
                s_mid = np.bincount(codes, implied ** mid[codes], minlength=n_groups)
                too_big = s_mid > 1.0
                lo = np.where(too_big, mid, lo)
                hi = np.where(too_big, hi, mid)
            k = (lo + hi) / 2
            powered = implied ** k[codes]
            fair = powered / np.bincount(codes, powered, minlength=n_groups)[codes]
        else:
            fair = implied / total[codes]
        frame = frame.assign(fair=fair)
        cache: dict[tuple[str, str, float], Selection | None] = {}
        out: dict[tuple[str, Selection], float] = {}
        for (mid, mkt, sel, ln), p in frame.groupby(["mid", "mkt", "sel", "line"], dropna=False)["fair"].mean().items():
            key = (str(mkt), str(sel), float(ln))
            if key not in cache:
                try:
                    cache[key] = Selection.from_codes(key[0], key[1], None if np.isnan(key[2]) else key[2])
                except (ValueError, TypeError):
                    cache[key] = None
            selection = cache[key]
            if selection is not None:
                out[(str(mid), selection)] = float(p)
        return out

    def evaluate(self, predictions: Mapping[str, MatchPrediction], odds: pd.DataFrame) -> list[Pick]:
        """Evalúa TODAS las selecciones con momio (mejor precio), con o sin valor.

        Args:
            predictions: Predicciones por ``match_id``.
            odds: Momios validados y enlazados a los ``match_id``.

        Returns:
            Lista de picks (uno por partido/selección) con stake calculado.
        """
        if odds.empty:
            return []
        consensus = self.market_consensus(odds)
        df = odds[odds["match_id"].isin(list(predictions))].copy()
        if df.empty:
            logger.warning("Ningún momio corresponde a partidos con predicción")
            return []
        # Una Selection por combinación única (mercado, resultado, línea), no por fila.
        line = pd.to_numeric(df["line"], errors="coerce")
        combos: dict[tuple[str, str, float], Selection | None] = {}
        keys = list(zip(df["market"].astype(str), df["selection"].astype(str), line.fillna(np.nan).to_numpy(float)))
        for k in set(keys):
            try:
                combos[k] = Selection.from_codes(k[0], k[1], None if np.isnan(k[2]) else k[2])
            except (ValueError, TypeError):
                combos[k] = None
        df["_sel"] = pd.Series([combos[k] for k in keys], index=df.index, dtype=object)
        df = df[df["_sel"].notna()]
        if self.config.allowed_markets is not None:  # perfil por liga: solo mercados con ventaja medida
            allowed = set(self.config.allowed_markets)
            df = df[[sel.market.value in allowed for sel in df["_sel"]]]
            if df.empty:
                return []
        best = df.loc[df.groupby(["match_id", "_sel"], sort=False)["odds"].idxmax()]
        picks: list[Pick] = []
        for mid, league, kickoff, bookmaker, price, sel in zip(
                best["match_id"], best["league"], best["commence_time"], best["bookmaker"], best["odds"], best["_sel"]):
            pred = predictions[mid]
            p_model = pred.probability(sel)
            p_market = consensus.get((str(mid), sel))
            pick = Pick(str(mid), str(league), pred.home_team, pred.away_team,
                        pred.kickoff if pred.kickoff is not None else kickoff, sel,
                        float(price), str(bookmaker), p_model, p_market,
                        action_share=pred.action_share(sel))
            pick = replace(pick, decision_prob=self.decision_probability(pick))
            # Kelly con push: f* = s · f*_binario(p_eff) (exacto para ganar/push/perder)
            stake = (modified_kelly(p_model, pick.odds, p_market, self.config.kelly) * pick.action_share
                     if self.is_value(pick) else 0.0)
            picks.append(replace(pick, stake_fraction=stake))
        return picks

    def decision_probability(self, pick: Pick) -> float:
        """Probabilidad usada para decidir el valor (modelo encogido hacia el mercado).

        Args:
            pick: Pick evaluado.

        Returns:
            ``w·p_modelo + (1−w)·p_mercado`` (o ``p_modelo`` si no hay mercado).
        """
        w = self.config.value_model_weight
        if pick.market_prob is None or w >= 1.0:
            return pick.model_prob
        return w * pick.model_prob + (1 - w) * pick.market_prob

    def is_value(self, pick: Pick) -> bool:
        """Indica si un pick cumple los criterios de valor configurados."""
        c = self.config
        p = self.decision_probability(pick)
        if p <= pick.implied_prob or pick.action_share * (p * pick.odds - 1) < c.min_edge:
            return False
        if not c.min_odds <= pick.odds <= c.max_odds or pick.model_prob < c.min_model_prob:
            return False
        if pick.market_prob is not None and abs(pick.model_prob - pick.market_prob) > c.max_market_gap:
            logger.info("Descartado por discrepancia con el mercado (>%.0f pp): %s",
                        100 * c.max_market_gap, pick.describe())
            return False
        return True

    def find(self, predictions: Mapping[str, MatchPrediction], odds: pd.DataFrame) -> list[Pick]:
        """Devuelve solo las apuestas de valor, ordenadas por EV descendente.

        Args:
            predictions: Predicciones por ``match_id``.
            odds: Momios validados.

        Returns:
            Picks de valor con stake sugerido.
        """
        value = [p for p in self.evaluate(predictions, odds) if self.is_value(p)]
        return sorted(value, key=lambda p: p.adjusted_edge, reverse=True)


# =============================================================================
# Parlays
# =============================================================================
@dataclass(frozen=True)
class Parlay:
    """Combinada evaluada.

    Attributes:
        legs: Patas del parlay.
        combined_odds: Momio combinado (producto).
        joint_prob: Probabilidad conjunta según el modelo.
        independent_prob: Producto simple de marginales (referencia).
        market_prob: Probabilidad justa del mercado para la combinada.
        stake_fraction: Stake sugerido (fracción del bankroll).
        warnings: Avisos (correlaciones, mismo partido...).
    """

    legs: tuple[Pick, ...]
    combined_odds: float
    joint_prob: float
    independent_prob: float
    market_prob: float | None
    stake_fraction: float
    warnings: tuple[str, ...] = ()

    @property
    def expected_value(self) -> float:
        """EV por unidad apostada."""
        return expected_value(self.joint_prob, self.combined_odds)

    @property
    def correlation_factor(self) -> float:
        """``joint_prob / independent_prob`` (1 = independientes)."""
        return self.joint_prob / self.independent_prob if self.independent_prob > 0 else 0.0

    @property
    def expected_log_growth(self) -> float:
        """Crecimiento logarítmico esperado del bankroll con el stake sugerido."""
        f, b, p = self.stake_fraction, self.combined_odds - 1.0, self.joint_prob
        if f <= 0:
            return 0.0
        return float(p * math.log1p(f * b) + (1 - p) * math.log1p(-f))

    @property
    def risk_level(self) -> str:
        """Nivel de riesgo cualitativo según la probabilidad conjunta."""
        if self.joint_prob >= 0.40:
            return "BAJO"
        if self.joint_prob >= 0.22:
            return "MEDIO"
        return "ALTO"

    def to_dict(self, bankroll: float | None = None) -> dict[str, object]:
        """Representación plana (para tablas/UI)."""
        d: dict[str, object] = {
            "patas": " + ".join(f"{leg.fixture}: {leg.selection.label}" for leg in self.legs),
            "n": len(self.legs), "momio": round(self.combined_odds, 2),
            "p_conjunta": round(self.joint_prob, 4), "p_implícita": round(1 / self.combined_odds, 4),
            "EV": round(self.expected_value, 4), "riesgo": self.risk_level,
            "stake_%": round(100 * self.stake_fraction, 2),
        }
        if bankroll is not None:
            d["stake"] = round(self.stake_fraction * bankroll, 2)
        return d


@dataclass(frozen=True)
class ParlayConfig:
    """Parámetros del generador de parlays.

    Attributes:
        min_legs: Patas mínimas.
        max_legs: Patas máximas.
        min_leg_prob: Probabilidad mínima de cada pata ("banker").
        min_leg_edge: EV mínimo de cada pata (0 = no aceptar patas con EV negativo).
        min_leg_odds: Momio mínimo por pata (patas a 1.05 no aportan).
        pool_size: Nº máximo de candidatos (controla la explosión combinatoria).
        min_joint_prob: Probabilidad conjunta mínima del parlay.
        max_combined_odds: Momio combinado máximo.
        min_parlay_edge: EV mínimo del parlay.
        allow_same_match: Permite varias patas del mismo partido (Bet Builder).
        max_leg_exposure: Veces máximas que una pata puede repetirse entre los
            parlays sugeridos (diversificación).
        top_n: Número de parlays a devolver.
        rank_by: Criterio de orden: ``growth`` (Kelly), ``ev`` o ``probability``.
        kelly: Parámetros de staking.
    """

    min_legs: int = 2
    max_legs: int = 3
    min_leg_prob: float = 0.55
    min_leg_edge: float = 0.0
    min_leg_odds: float = 1.15
    pool_size: int = 14
    min_joint_prob: float = 0.15
    max_combined_odds: float = 12.0
    min_parlay_edge: float = 0.02
    allow_same_match: bool = False
    max_leg_exposure: int = 2
    top_n: int = 5
    rank_by: Literal["growth", "ev", "probability"] = "growth"
    kelly: KellyConfig = field(default_factory=lambda: KellyConfig(max_stake=0.015))


class ConflictChecker:
    """Detecta combinaciones contradictorias o redundantes."""

    def __init__(self, max_goals: int = 10) -> None:
        """Crea el verificador.

        Args:
            max_goals: Tamaño de la rejilla de marcadores.
        """
        self.max_goals = max_goals

    def check(self, legs: Sequence[Pick], predictions: Mapping[str, MatchPrediction] | None = None) -> list[str]:
        """Devuelve los problemas de una combinación (lista vacía = válida).

        * Contradicción: dos patas del mismo partido sin marcador común
          (Local + Visitante, Menos de 1.5 + Ambos marcan...).
        * Redundancia: una pata implica a otra (Más de 2.5 + Más de 1.5).
        * Duplicado: la misma selección dos veces.

        Args:
            legs: Patas.
            predictions: Predicciones (para la verificación con probabilidades).

        Returns:
            Lista de descripciones de conflictos.
        """
        issues: list[str] = []
        by_match: dict[str, list[Pick]] = {}
        for leg in legs:
            by_match.setdefault(leg.match_id, []).append(leg)
        for mid, group in by_match.items():
            if len(group) < 2:
                continue
            sels = [g.selection for g in group]
            if len(set(sels)) < len(sels):
                issues.append(f"{group[0].fixture}: selección duplicada")
                continue
            if not selections_compatible(sels, self.max_goals):
                issues.append(f"{group[0].fixture}: contradictoria ({' + '.join(s.label for s in sels)})")
                continue
            for a, b in itertools.permutations(sels, 2):
                ma, mb = a.mask(self.max_goals), b.mask(self.max_goals)
                if np.array_equal(ma & mb, ma):
                    issues.append(f"{group[0].fixture}: redundante ({a.label} implica {b.label})")
                    break
            pred = (predictions or {}).get(mid)
            if pred is not None and pred.joint_probability(sels) <= 1e-9:
                issues.append(f"{group[0].fixture}: probabilidad conjunta nula")
        return issues


class ParlayBuilder:
    """Genera parlays a partir de picks de alta probabilidad y EV positivo."""

    def __init__(self, config: ParlayConfig | None = None, checker: ConflictChecker | None = None) -> None:
        """Crea el generador.

        Args:
            config: Parámetros.
            checker: Verificador de conflictos.

        Raises:
            ValueError: Si la configuración de patas es incoherente.
        """
        self.config = config or ParlayConfig()
        if not 2 <= self.config.min_legs <= self.config.max_legs:
            raise ValueError("Se requiere 2 <= min_legs <= max_legs")
        self.checker = checker or ConflictChecker()

    def select_candidates(self, picks: Sequence[Pick]) -> list[Pick]:
        """Filtra y ordena las patas candidatas ("bankers" con EV >= mínimo).

        Args:
            picks: Picks evaluados (no necesariamente de valor).

        Returns:
            Hasta ``pool_size`` candidatos, ordenados por probabilidad y EV.
        """
        c = self.config
        best: dict[tuple[str, Selection], Pick] = {}
        for p in picks:
            if p.selection.is_graded:  # DNB/hándicap asiático: el push en parlays varía por casa
                continue
            if p.model_prob < c.min_leg_prob or p.edge < c.min_leg_edge or p.odds < c.min_leg_odds:
                continue
            key = (p.match_id, p.selection)
            if key not in best or p.odds > best[key].odds:
                best[key] = p
        ranked = sorted(best.values(), key=lambda p: (p.model_prob * (1 + p.edge), p.edge), reverse=True)
        return ranked[: c.pool_size]

    def evaluate_combination(self, legs: Sequence[Pick],
                             predictions: Mapping[str, MatchPrediction] | None = None) -> Parlay | None:
        """Evalúa una combinación concreta.

        Args:
            legs: Patas.
            predictions: Predicciones por ``match_id`` (necesarias para patas
                del mismo partido).

        Returns:
            El :class:`Parlay` o ``None`` si es inválido o no cumple filtros.
        """
        c = self.config
        issues = self.checker.check(legs, predictions)
        if issues:
            logger.debug("Combinación descartada: %s", issues)
            return None
        by_match: dict[str, list[Pick]] = {}
        for leg in legs:
            by_match.setdefault(leg.match_id, []).append(leg)
        warnings: list[str] = []
        joint, indep, market = 1.0, 1.0, 1.0
        market_known = True
        for mid, group in by_match.items():
            indep *= float(np.prod([g.model_prob for g in group]))
            if len(group) == 1:
                joint *= group[0].model_prob
            else:
                if not c.allow_same_match:
                    return None
                pred = (predictions or {}).get(mid)
                if pred is None:
                    return None
                sels = [g.selection for g in group]
                joint *= pred.joint_probability(sels)
                factor = pred.correlation_factor(sels)
                warnings.append(f"{group[0].fixture}: patas correlacionadas (factor {factor:.2f}); "
                                "la casa puede ajustar el momio en un Bet Builder")
            for g in group:
                if g.market_prob is None:
                    market_known = False
                else:
                    market *= g.market_prob
        odds = float(np.prod([leg.odds for leg in legs]))
        if joint < c.min_joint_prob or odds > c.max_combined_odds:
            return None
        if expected_value(joint, odds) < c.min_parlay_edge:
            return None
        market_prob = market if market_known else None
        stake = modified_kelly(joint, odds, market_prob, c.kelly, legs=len(legs))
        return Parlay(tuple(legs), odds, joint, indep, market_prob, stake, tuple(warnings))

    def _rank_key(self, parlay: Parlay) -> float:
        """Clave de ordenación según ``rank_by``."""
        if self.config.rank_by == "ev":
            return parlay.expected_value
        if self.config.rank_by == "probability":
            return parlay.joint_prob
        return parlay.expected_log_growth

    def build(self, picks: Sequence[Pick], predictions: Mapping[str, MatchPrediction] | None = None) -> list[Parlay]:
        """Genera y ordena los mejores parlays.

        Args:
            picks: Picks evaluados (p. ej. salida de :meth:`ValueBetFinder.evaluate`).
            predictions: Predicciones por ``match_id``.

        Returns:
            Hasta ``top_n`` parlays diversificados.
        """
        c = self.config
        pool = self.select_candidates(picks)
        if len(pool) < c.min_legs:
            logger.info("Candidatos insuficientes para parlays (%d)", len(pool))
            return []
        evaluated: list[Parlay] = []
        for n_legs in range(c.min_legs, min(c.max_legs, len(pool)) + 1):
            for combo in itertools.combinations(pool, n_legs):
                parlay = self.evaluate_combination(combo, predictions)
                if parlay is not None:
                    evaluated.append(parlay)
        evaluated.sort(key=self._rank_key, reverse=True)
        selected: list[Parlay] = []
        usage: dict[tuple[str, Selection], int] = {}
        for parlay in evaluated:
            keys = [(leg.match_id, leg.selection) for leg in parlay.legs]
            if any(usage.get(k, 0) >= c.max_leg_exposure for k in keys):
                continue
            selected.append(parlay)
            for k in keys:
                usage[k] = usage.get(k, 0) + 1
            if len(selected) >= c.top_n:
                break
        return selected


def apply_exposure_cap(singles: Sequence[Pick], parlays: Sequence[Parlay],
                       max_total_exposure: float = 0.10) -> tuple[list[Pick], list[Parlay], float]:
    """Escala los stakes si la exposición total supera el tope.

    Args:
        singles: Apuestas sencillas con stake.
        parlays: Parlays con stake.
        max_total_exposure: Fracción máxima del bankroll en juego simultáneo.

    Returns:
        ``(singles_escaladas, parlays_escalados, factor_aplicado)``.

    Raises:
        ValueError: Si el tope no está en (0, 1].
    """
    if not 0 < max_total_exposure <= 1:
        raise ValueError("max_total_exposure debe estar en (0, 1]")
    total = sum(p.stake_fraction for p in singles) + sum(p.stake_fraction for p in parlays)
    factor = 1.0 if total <= max_total_exposure or total == 0 else max_total_exposure / total
    if factor < 1.0:
        logger.info("Exposición %.1f%% > tope %.1f%%: stakes escalados x%.2f",
                    100 * total, 100 * max_total_exposure, factor)
    return ([replace(p, stake_fraction=p.stake_fraction * factor) for p in singles],
            [replace(p, stake_fraction=p.stake_fraction * factor) for p in parlays], factor)


def apply_match_cap(singles: Sequence[Pick], max_per_match: float | None) -> list[Pick]:
    """Limita la exposición total de las sencillas de un MISMO partido.

    Varias sencillas de un partido suelen estar correlacionadas (p. ej. DNB
    visitante y AH visitante +0.25); sin tope, el riesgo se concentra.

    Args:
        singles: Sencillas con stake.
        max_per_match: Fracción máxima del bankroll por partido (``None`` = sin tope).

    Returns:
        Sencillas con los stakes escalados donde se superaba el tope.

    Raises:
        ValueError: Si el tope no está en (0, 1].
    """
    if max_per_match is None:
        return list(singles)
    if not 0 < max_per_match <= 1:
        raise ValueError("max_per_match debe estar en (0, 1]")
    totals: dict[str, float] = {}
    for p in singles:
        totals[p.match_id] = totals.get(p.match_id, 0.0) + p.stake_fraction
    return [replace(p, stake_fraction=p.stake_fraction * max_per_match / totals[p.match_id])
            if totals[p.match_id] > max_per_match else p for p in singles]


__all__ = [
    "apply_match_cap", "ConflictChecker", "KellyConfig", "Parlay", "ParlayBuilder", "ParlayConfig", "Pick", "ValueBetConfig",
    "ValueBetFinder", "apply_exposure_cap", "expected_value", "implied_probability", "kelly_fraction",
    "modified_kelly", "remove_margin",
]
