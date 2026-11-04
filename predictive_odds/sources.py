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
    league = get_league(league_key)
    odds = TheOddsApiProvider(odds_key.strip(), include_btts=include_btts)
    api_football = ApiFootballProvider(af_key.strip()) if af_key.strip() else None
    history: ResultsProvider
    if league.football_data_code:
        history = FootballDataCoUkProvider()
        if understat:
            history = XgEnrichedProvider(history, UnderstatProvider())
    elif api_football is not None:
        history = api_football
    else:
        raise AuthenticationError(f"{league.name} requiere una clave de API-Football para el histórico")
    return DataFetcher(history, fixtures_provider=api_football or history, odds_provider=odds)


__all__ = ["make_fetcher"]
