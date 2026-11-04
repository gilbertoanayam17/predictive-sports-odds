"""Modelos de dominio compartidos por todos los módulos.

La pieza central es :class:`Selection`, que representa una selección de apuesta
(p. ej. "Más de 2.5 goles") y sabe expresarse como una *máscara* sobre la matriz
de marcadores ``[goles_local, goles_visitante]``. Gracias a ello:

* cualquier mercado se deriva de la misma matriz de probabilidades de Poisson;
* la probabilidad conjunta de varias selecciones del mismo partido es exacta;
* una combinación es contradictoria si y solo si la intersección de sus
  máscaras es vacía (p. ej. "Menos de 1.5" + "Ambos marcan: Sí").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

BoolMatrix = npt.NDArray[np.bool_]
FloatMatrix = npt.NDArray[np.float64]

#: Columnas estándar de un DataFrame de partidos (históricos o próximos).
MATCH_COLUMNS: list[str] = [
    "match_id", "league", "season", "date", "home_team", "away_team",
    "home_goals", "away_goals", "home_xg", "away_xg", "status",
]

#: Columnas estándar de un DataFrame de momios (formato decimal).
ODDS_COLUMNS: list[str] = [
    "match_id", "league", "home_team", "away_team", "commence_time",
    "bookmaker", "market", "selection", "line", "odds",
]

STATUS_FINISHED = "FT"
STATUS_SCHEDULED = "NS"


class Market(str, Enum):
    """Mercados de apuesta soportados."""

    MATCH_RESULT = "1X2"
    DOUBLE_CHANCE = "DC"
    OVER_UNDER = "OU"
    BTTS = "BTTS"
    CORRECT_SCORE = "CS"


class Outcome(str, Enum):
    """Resultados posibles dentro de un mercado."""

    HOME = "HOME"
    DRAW = "DRAW"
    AWAY = "AWAY"
    HOME_OR_DRAW = "1X"
    HOME_OR_AWAY = "12"
    DRAW_OR_AWAY = "X2"
    OVER = "OVER"
    UNDER = "UNDER"
    YES = "YES"
    NO = "NO"
    SCORE = "SCORE"


VALID_OUTCOMES: dict[Market, frozenset[Outcome]] = {
    Market.MATCH_RESULT: frozenset({Outcome.HOME, Outcome.DRAW, Outcome.AWAY}),
    Market.DOUBLE_CHANCE: frozenset(
        {Outcome.HOME_OR_DRAW, Outcome.HOME_OR_AWAY, Outcome.DRAW_OR_AWAY}
    ),
    Market.OVER_UNDER: frozenset({Outcome.OVER, Outcome.UNDER}),
    Market.BTTS: frozenset({Outcome.YES, Outcome.NO}),
    Market.CORRECT_SCORE: frozenset({Outcome.SCORE}),
}

#: Número de resultados que forman un mercado completo (para quitar el margen).
MARKET_SIZE: dict[Market, int] = {
    Market.MATCH_RESULT: 3,
    Market.OVER_UNDER: 2,
    Market.BTTS: 2,
}


@dataclass(frozen=True)
class Selection:
    """Una selección concreta de apuesta.

    Attributes:
        market: Mercado al que pertenece.
        outcome: Resultado elegido dentro del mercado.
        line: Línea de goles (solo ``OVER_UNDER``; se exigen líneas ``x.5``
            para evitar devoluciones/push).
        score: Marcador exacto ``(local, visitante)`` (solo ``CORRECT_SCORE``).
    """

    market: Market
    outcome: Outcome
    line: float | None = None
    score: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Valida la coherencia entre mercado, resultado, línea y marcador.

        Raises:
            ValueError: Si la combinación de atributos no es válida.
        """
        if self.outcome not in VALID_OUTCOMES[self.market]:
            raise ValueError(f"Resultado {self.outcome} inválido para el mercado {self.market}")
        if self.market is Market.OVER_UNDER:
            if self.line is None or self.line < 0 or abs((self.line * 2) % 2 - 1) > 1e-9:
                raise ValueError(f"Línea Over/Under inválida: {self.line} (solo se admiten x.5)")
        elif self.line is not None:
            raise ValueError("Solo el mercado Over/Under admite línea")
        if self.market is Market.CORRECT_SCORE:
            if self.score is None or min(self.score) < 0:
                raise ValueError("El mercado de marcador exacto requiere un marcador válido")
        elif self.score is not None:
            raise ValueError("Solo el mercado de marcador exacto admite 'score'")

    # ------------------------------------------------------------------ factories
    @classmethod
    def home(cls) -> Selection:
        """Crea la selección 'gana local' (1)."""
        return cls(Market.MATCH_RESULT, Outcome.HOME)

    @classmethod
    def draw(cls) -> Selection:
        """Crea la selección 'empate' (X)."""
        return cls(Market.MATCH_RESULT, Outcome.DRAW)

    @classmethod
    def away(cls) -> Selection:
        """Crea la selección 'gana visitante' (2)."""
        return cls(Market.MATCH_RESULT, Outcome.AWAY)

    @classmethod
    def over(cls, line: float) -> Selection:
        """Crea la selección 'Más de ``line`` goles'."""
        return cls(Market.OVER_UNDER, Outcome.OVER, line=float(line))

    @classmethod
    def under(cls, line: float) -> Selection:
        """Crea la selección 'Menos de ``line`` goles'."""
        return cls(Market.OVER_UNDER, Outcome.UNDER, line=float(line))

    @classmethod
    def btts(cls, yes: bool = True) -> Selection:
        """Crea la selección 'Ambos marcan' (Sí/No)."""
        return cls(Market.BTTS, Outcome.YES if yes else Outcome.NO)

    @classmethod
    def correct_score(cls, home_goals: int, away_goals: int) -> Selection:
        """Crea la selección de marcador exacto."""
        return cls(Market.CORRECT_SCORE, Outcome.SCORE, score=(home_goals, away_goals))

    @classmethod
    def from_codes(cls, market: str, outcome: str, line: float | None = None) -> Selection:
        """Construye una selección a partir de códigos de texto (p. ej. de un CSV/API).

        Args:
            market: Código del mercado (``"1X2"``, ``"OU"``, ``"BTTS"``...).
            outcome: Código del resultado (``"HOME"``, ``"OVER"``...).
            line: Línea de goles, si aplica.

        Returns:
            La selección correspondiente.

        Raises:
            ValueError: Si los códigos no son válidos.
        """
        mkt = Market(market)
        clean_line = None if line is None or (isinstance(line, float) and np.isnan(line)) else float(line)
        return cls(mkt, Outcome(outcome), line=clean_line if mkt is Market.OVER_UNDER else None)

    # --------------------------------------------------------------- presentation
    @property
    def label(self) -> str:
        """Etiqueta legible en español."""
        labels = {
            Outcome.HOME: "Local (1)",
            Outcome.DRAW: "Empate (X)",
            Outcome.AWAY: "Visitante (2)",
            Outcome.HOME_OR_DRAW: "Doble oportunidad 1X",
            Outcome.HOME_OR_AWAY: "Doble oportunidad 12",
            Outcome.DRAW_OR_AWAY: "Doble oportunidad X2",
            Outcome.YES: "Ambos marcan: Sí",
            Outcome.NO: "Ambos marcan: No",
        }
        if self.outcome is Outcome.OVER:
            return f"Más de {self.line}"
        if self.outcome is Outcome.UNDER:
            return f"Menos de {self.line}"
        if self.outcome is Outcome.SCORE and self.score is not None:
            return f"Marcador {self.score[0]}-{self.score[1]}"
        return labels[self.outcome]

    def mask(self, max_goals: int) -> BoolMatrix:
        """Devuelve la máscara booleana de marcadores donde la selección gana.

        Args:
            max_goals: Goles máximos por equipo representados en la matriz.

        Returns:
            Matriz booleana de tamaño ``(max_goals+1, max_goals+1)`` (solo lectura).
        """
        return _selection_mask(self, max_goals)

    def __str__(self) -> str:  # pragma: no cover - cosmético
        """Representación legible."""
        return self.label


@lru_cache(maxsize=512)
def _selection_mask(selection: Selection, max_goals: int) -> BoolMatrix:
    """Calcula (y memoiza) la máscara de una selección.

    Args:
        selection: Selección de apuesta.
        max_goals: Goles máximos por equipo.

    Returns:
        Máscara booleana inmutable.
    """
    h = np.arange(max_goals + 1)[:, None]
    a = np.arange(max_goals + 1)[None, :]
    total = h + a
    o = selection.outcome
    if o is Outcome.HOME:
        m = h > a
    elif o is Outcome.DRAW:
        m = h == a
    elif o is Outcome.AWAY:
        m = h < a
    elif o is Outcome.HOME_OR_DRAW:
        m = h >= a
    elif o is Outcome.HOME_OR_AWAY:
        m = h != a
    elif o is Outcome.DRAW_OR_AWAY:
        m = h <= a
    elif o is Outcome.OVER:
        m = total > (selection.line or 0.0)
    elif o is Outcome.UNDER:
        m = total < (selection.line or 0.0)
    elif o is Outcome.YES:
        m = (h > 0) & (a > 0)
    elif o is Outcome.NO:
        m = (h == 0) | (a == 0)
    elif o is Outcome.SCORE and selection.score is not None:
        m = (h == selection.score[0]) & (a == selection.score[1])
    else:  # pragma: no cover - protegido por __post_init__
        raise ValueError(f"Selección no soportada: {selection}")
    result = np.broadcast_to(m, (max_goals + 1, max_goals + 1)).copy()
    result.setflags(write=False)
    return result


def selections_compatible(selections: Sequence[Selection], max_goals: int = 10) -> bool:
    """Indica si un conjunto de selecciones del MISMO partido puede ganar a la vez.

    Args:
        selections: Selecciones a combinar.
        max_goals: Tamaño de la rejilla de marcadores a considerar.

    Returns:
        ``True`` si existe al menos un marcador que satisface todas.
    """
    if len(selections) < 2:
        return True
    mask = np.ones((max_goals + 1, max_goals + 1), dtype=bool)
    for sel in selections:
        mask &= sel.mask(max_goals)
    return bool(mask.any())


@dataclass
class MatchPrediction:
    """Predicción probabilística de un partido.

    La matriz ``score_matrix[i, j]`` contiene ``P(local = i, visitante = j)``.
    ``overrides`` permite que un ensamble (Poisson + ML) sustituya la
    probabilidad marginal de algunos mercados; las probabilidades conjuntas se
    reescalan para mantener la coherencia.

    Attributes:
        match_id: Identificador del partido.
        home_team: Equipo local.
        away_team: Equipo visitante.
        lambda_home: Goles esperados del local.
        lambda_away: Goles esperados del visitante.
        score_matrix: Matriz de probabilidades de marcador (suma 1).
        league: Clave de la liga.
        kickoff: Fecha/hora del partido.
        overrides: Probabilidades marginales sustituidas por el ensamble.
        sources: Desglose de probabilidades por modelo (para auditoría/UI).
    """

    match_id: str
    home_team: str
    away_team: str
    lambda_home: float
    lambda_away: float
    score_matrix: FloatMatrix
    league: str = ""
    kickoff: pd.Timestamp | None = None
    overrides: dict[Selection, float] = field(default_factory=dict)
    sources: dict[str, dict[Selection, float]] = field(default_factory=dict)

    @property
    def max_goals(self) -> int:
        """Goles máximos por equipo representados en la matriz."""
        return int(self.score_matrix.shape[0] - 1)

    def matrix_probability(self, selection: Selection) -> float:
        """Probabilidad de la selección según la matriz de marcadores pura.

        Args:
            selection: Selección a evaluar.

        Returns:
            Probabilidad en [0, 1].
        """
        return float(self.score_matrix[selection.mask(self.max_goals)].sum())

    def probability(self, selection: Selection) -> float:
        """Probabilidad final (ensamble si existe, si no la matriz).

        Args:
            selection: Selección a evaluar.

        Returns:
            Probabilidad en [0, 1].
        """
        if selection in self.overrides:
            return float(self.overrides[selection])
        return self.matrix_probability(selection)

    def correlation_factor(self, selections: Sequence[Selection]) -> float:
        """Razón ``P(A∩B∩...) / (P(A)·P(B)·...)`` calculada sobre la matriz.

        Un valor > 1 indica correlación positiva (p. ej. Local + Más de 2.5),
        < 1 negativa y 0 que las selecciones son contradictorias.

        Args:
            selections: Selecciones del mismo partido.

        Returns:
            Factor de correlación (>= 0).
        """
        if len(selections) < 2:
            return 1.0
        mask = np.ones_like(self.score_matrix, dtype=bool)
        marg = 1.0
        for sel in selections:
            mask &= sel.mask(self.max_goals)
            marg *= self.matrix_probability(sel)
        if marg <= 0:
            return 0.0
        return float(self.score_matrix[mask].sum() / marg)

    def joint_probability(self, selections: Sequence[Selection]) -> float:
        """Probabilidad de que todas las selecciones (mismo partido) ganen.

        Se usa el factor de correlación de la matriz aplicado a las marginales
        finales (que pueden venir del ensamble), acotado por la menor marginal.

        Args:
            selections: Selecciones del mismo partido.

        Returns:
            Probabilidad conjunta en [0, 1].
        """
        if not selections:
            return 1.0
        if len(selections) == 1:
            return self.probability(selections[0])
        factor = self.correlation_factor(selections)
        if factor <= 1e-12:
            return 0.0
        marginals = [self.probability(s) for s in selections]
        joint = factor * float(np.prod(marginals))
        return float(min(joint, min(marginals)))

    def most_likely_scores(self, n: int = 5) -> list[tuple[int, int, float]]:
        """Devuelve los ``n`` marcadores más probables.

        Args:
            n: Número de marcadores.

        Returns:
            Lista de tuplas ``(goles_local, goles_visitante, probabilidad)``.
        """
        flat = np.argsort(self.score_matrix, axis=None)[::-1][:n]
        rows, cols = np.unravel_index(flat, self.score_matrix.shape)
        return [(int(r), int(c), float(self.score_matrix[r, c])) for r, c in zip(rows, cols)]

    def summary(self) -> dict[str, float | str]:
        """Resumen plano de los mercados principales (útil para tablas/UI).

        Returns:
            Diccionario con goles esperados y probabilidades clave.
        """
        top = self.most_likely_scores(1)[0]
        return {
            "match_id": self.match_id,
            "partido": f"{self.home_team} vs {self.away_team}",
            "xG_local": round(self.lambda_home, 2),
            "xG_visit": round(self.lambda_away, 2),
            "P(1)": self.probability(Selection.home()),
            "P(X)": self.probability(Selection.draw()),
            "P(2)": self.probability(Selection.away()),
            "P(O2.5)": self.probability(Selection.over(2.5)),
            "P(BTTS)": self.probability(Selection.btts(True)),
            "marcador_top": f"{top[0]}-{top[1]} ({top[2]:.1%})",
        }
