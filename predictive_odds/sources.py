"""Elección de fuentes de datos, compartida por la app y el CLI.

Una sola función decide qué proveedores usar para que la app y la línea de
comandos se comporten igual (antes cada una tenía su propia copia y divergían).
"""

from __future__ import annotations

from typing import Sequence

from .config import get_league
from .data_fetcher import (
    ApiFootballProvider,
    AuthenticationError,
    DataFetcher,
    FootballDataCoUkProvider,
    ResultsProvider,
    SimulatedDataProvider,
    SimulationConfig,
    TheOddsApiProvider,
)
from .xg_sources import UnderstatProvider, XgEnrichedProvider


def make_fetcher(live: bool, league_key: str, seasons: Sequence[int], odds_key: str = "", af_key: str = "",
                 seed: int = 7, played_rounds: int = 28, include_btts: bool = False,
                 understat: bool = False) -> DataFetcher:
    """Construye la fachada de datos.

    Modo simulado: liga sintética (sin red ni claves).

    Modo real:

    * histórico: football-data.co.uk si la liga está ahí (opcionalmente con xG
      de Understat); si no, API-Football (requiere clave);
    * calendario: API-Football si hay clave; si no, se deduce de los eventos de
      momios (el histórico de football-data solo trae partidos jugados);
    * momios: The Odds API.

    Args:
        live: Datos reales (``True``) o simulados.
        league_key: Liga principal (decide la fuente del histórico).
        seasons: Temporadas a usar (la última es la actual).
        odds_key: Clave de The Odds API.
        af_key: Clave de API-Football (opcional).
        seed: Semilla de la simulación.
        played_rounds: Jornadas jugadas de la temporada actual en la simulación.
        include_btts: Pedir también momios BTTS (más créditos).
        understat: Añadir xG de Understat al histórico (experimental).

    Returns:
        ``DataFetcher`` configurado.

    Raises:
        AuthenticationError: Si falta una clave necesaria.
    """
    if not live:
        sim = SimulatedDataProvider(SimulationConfig(seed=seed, seasons=tuple(seasons),
                                                     played_rounds_current=played_rounds))
        return DataFetcher(sim, odds_provider=sim)
    odds = TheOddsApiProvider(odds_key.strip(), include_btts=include_btts)
    api_football = ApiFootballProvider(af_key.strip()) if af_key.strip() else None
    history = make_history_provider(True, league_key, seasons, af_key, understat=understat)
    return DataFetcher(history, fixtures_provider=api_football or history, odds_provider=odds)


def make_history_provider(live: bool, league_key: str, seasons: Sequence[int], af_key: str = "", seed: int = 7,
                          played_rounds: int = 28, understat: bool = False) -> ResultsProvider:
    """Solo la fuente de resultados (para liquidar apuestas): no necesita la clave de momios ni gasta créditos.

    Args:
        live: Datos reales (``True``) o simulados.
        league_key: Liga.
        seasons: Temporadas (solo para la simulación).
        af_key: Clave de API-Football, necesaria solo si la liga no está en football-data.co.uk.
        seed: Semilla de la simulación.
        played_rounds: Jornadas jugadas de la temporada actual en la simulación.
        understat: Añadir xG de Understat (experimental).

    Returns:
        Proveedor con ``fetch_matches``.

    Raises:
        AuthenticationError: Si la liga necesita API-Football y no hay clave.
    """
    if not live:
        return SimulatedDataProvider(SimulationConfig(seed=seed, seasons=tuple(seasons),
                                                      played_rounds_current=played_rounds))
    league = get_league(league_key)
    if league.football_data_code:
        history: ResultsProvider = FootballDataCoUkProvider()
        return XgEnrichedProvider(history, UnderstatProvider()) if understat else history
    if af_key.strip():
        return ApiFootballProvider(af_key.strip())
    raise AuthenticationError(f"{league.name} requiere una clave de API-Football para el histórico")


__all__ = ["make_fetcher", "make_history_provider"]
