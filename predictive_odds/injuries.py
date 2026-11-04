"""Traducción de lesiones y bajas a ajustes de fuerza por equipo.

Sin datos de minutos o importancia de cada jugador, se usa una heurística
conservadora: cada baja confirmada resta un pequeño porcentaje al ataque y
suma otro a los goles recibidos; las dudas cuentan a medias y el efecto total
se acota. Si se dispone de la importancia de los jugadores (p. ej. su cuota de
goles+asistencias o de minutos), puede pasarse en ``player_impacts`` y
sustituye a la heurística para esos jugadores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import pandas as pd

from .data_fetcher import TeamNameNormalizer
from .statistical_models import TeamAdjustment

logger = logging.getLogger(__name__)

INJURY_COLUMNS: list[str] = ["team", "player", "type", "reason", "match_id"]


@dataclass(frozen=True)
class PlayerImpact:
    """Impacto conocido de la ausencia de un jugador.

    Attributes:
        attack: Fracción de goles del equipo que se pierden (0.08 = −8%).
        defense: Fracción extra de goles recibidos (0.05 = +5%).
    """

    attack: float = 0.0
    defense: float = 0.0


@dataclass(frozen=True)
class InjuryImpactConfig:
    """Parámetros de la heurística de lesiones.

    Attributes:
        attack_per_player: Pérdida de ataque por baja confirmada.
        defense_per_player: Aumento de goles recibidos por baja confirmada.
        doubtful_weight: Peso de un jugador en duda (``Questionable``).
        max_attack_loss: Pérdida máxima total de ataque.
        max_defense_loss: Aumento máximo total de goles recibidos.
        player_impacts: Impactos específicos por jugador (nombre exacto); sustituyen
            a la heurística, pero los topes totales siguen aplicando (protección
            ante datos erróneos).
    """

    attack_per_player: float = 0.03
    defense_per_player: float = 0.02
    doubtful_weight: float = 0.5
    max_attack_loss: float = 0.15
    max_defense_loss: float = 0.12
    player_impacts: Mapping[str, PlayerImpact] = field(default_factory=dict)


class InjuryAdjuster:
    """Convierte un listado de bajas en :class:`TeamAdjustment` por equipo."""

    def __init__(self, config: InjuryImpactConfig | None = None,
                 normalizer: TeamNameNormalizer | None = None) -> None:
        """Crea el ajustador.

        Args:
            config: Parámetros de la heurística.
            normalizer: Normalizador para emparejar nombres de equipos.
        """
        self.config = config or InjuryImpactConfig()
        self.normalizer = normalizer or TeamNameNormalizer()

    def _weight(self, injury_type: object) -> float:
        """Peso de una baja según su tipo (confirmada = 1, duda = ``doubtful_weight``)."""
        text = str(injury_type or "").lower()
        if "question" in text or "doubt" in text or "duda" in text:
            return self.config.doubtful_weight
        return 1.0

    def adjustments(self, injuries: pd.DataFrame, teams: Iterable[str] | None = None) -> dict[str, TeamAdjustment]:
        """Calcula el ajuste de cada equipo con bajas.

        Args:
            injuries: DataFrame con ``team``, ``player`` y ``type``.
            teams: Nombres de referencia (los del histórico) para alinear nombres.

        Returns:
            Mapa ``equipo -> TeamAdjustment`` (solo equipos con bajas).
        """
        if injuries is None or injuries.empty:
            return {}
        missing = {"team", "player"} - set(injuries.columns)
        if missing:
            logger.warning("Datos de lesiones sin columnas %s; se ignoran", sorted(missing))
            return {}
        known = list(teams) if teams is not None else None
        c = self.config
        acc: dict[str, tuple[float, float, list[str]]] = {}
        for row in injuries.drop_duplicates(subset=["team", "player"]).itertuples(index=False):
            team = str(row.team)
            if known is not None:
                matched = self.normalizer.match(team, known)
                if matched is None:
                    logger.info("Lesión de un equipo desconocido ignorada: %s", team)
                    continue
                team = matched
            w = self._weight(getattr(row, "type", None))
            impact = c.player_impacts.get(str(row.player))
            att = (impact.attack if impact else c.attack_per_player) * w
            dfn = (impact.defense if impact else c.defense_per_player) * w
            a, d, players = acc.get(team, (0.0, 0.0, []))
            acc[team] = (a + att, d + dfn, players + [str(row.player)])
        out: dict[str, TeamAdjustment] = {}
        for team, (a, d, players) in acc.items():
            a, d = min(a, c.max_attack_loss), min(d, c.max_defense_loss)
            out[team] = TeamAdjustment(attack=1 - a, defense=1 + d,
                                       reason=f"{len(players)} baja(s): {', '.join(players[:4])}"
                                              + ("…" if len(players) > 4 else ""))
        return out


__all__ = ["INJURY_COLUMNS", "InjuryAdjuster", "InjuryImpactConfig", "PlayerImpact"]
