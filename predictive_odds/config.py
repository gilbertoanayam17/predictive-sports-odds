"""Registro de ligas y configuración agnóstica al deporte.

Agregar una liga nueva consiste únicamente en registrar un ``LeagueConfig`` con
los identificadores que usa cada proveedor de datos. El resto del sistema
(modelos, motor de apuestas) no conoce detalles de ninguna liga concreta.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal


class ConfigurationError(Exception):
    """Error de configuración (liga desconocida, parámetros inválidos, etc.)."""


@dataclass(frozen=True)
class LeagueConfig:
    """Describe una competición y cómo se identifica en cada proveedor.

    Attributes:
        key: Identificador interno estable (p. ej. ``"EPL"``).
        name: Nombre legible.
        sport: Deporte (``"soccer"`` por ahora; la arquitectura admite otros).
        country: País o región.
        api_football_id: ID de la liga en API-Football (api-sports.io).
        odds_api_sport_key: Clave del deporte/liga en The Odds API.
        football_data_code: Código de división en football-data.co.uk.
        football_data_format: ``"main"`` (un CSV por temporada) o ``"extra"``
            (un CSV con todas las temporadas, usado para ligas como Liga MX).
        sample_teams: Equipos usados por el proveedor simulado.
        is_international: Competición entre clubes de varias ligas (p. ej. UCL);
            el modelo multiliga usa las ligas domésticas para comparar equipos.
    """

    key: str
    name: str
    sport: str = "soccer"
    country: str = ""
    api_football_id: int | None = None
    odds_api_sport_key: str | None = None
    football_data_code: str | None = None
    football_data_format: Literal["main", "extra"] | None = None
    sample_teams: tuple[str, ...] = ()
    is_international: bool = False


LEAGUES: dict[str, LeagueConfig] = {
    "EPL": LeagueConfig(
        key="EPL",
        name="Premier League",
        country="England",
        api_football_id=39,
        odds_api_sport_key="soccer_epl",
        football_data_code="E0",
        football_data_format="main",
        sample_teams=(
            "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
            "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
            "Leeds United", "Liverpool", "Manchester City", "Manchester United",
            "Newcastle United", "Nottingham Forest", "Sunderland", "Tottenham Hotspur",
            "West Ham United", "Wolverhampton Wanderers",
        ),
    ),
    "UCL": LeagueConfig(
        key="UCL",
        name="UEFA Champions League",
        country="Europe",
        api_football_id=2,
        odds_api_sport_key="soccer_uefa_champs_league",
        is_international=True,
        sample_teams=(
            "Real Madrid", "Barcelona", "Bayern Munich", "Paris Saint-Germain",
            "Inter", "Manchester City", "Arsenal", "Liverpool", "Borussia Dortmund",
            "Atletico Madrid", "Juventus", "Benfica", "Napoli", "Bayer Leverkusen",
            "Chelsea", "Sporting CP", "PSV Eindhoven", "Club Brugge",
        ),
    ),
    "LIGA_MX": LeagueConfig(
        key="LIGA_MX",
        name="Liga MX",
        country="Mexico",
        api_football_id=262,
        odds_api_sport_key="soccer_mexico_ligamx",
        football_data_code="MEX",
        football_data_format="extra",
        sample_teams=(
            "America", "Guadalajara", "Cruz Azul", "Pumas UNAM", "Tigres UANL",
            "Monterrey", "Toluca", "Santos Laguna", "Pachuca", "Leon",
            "Atlas", "Puebla", "Queretaro", "Necaxa", "Tijuana",
            "Mazatlan", "Juarez", "Atletico San Luis",
        ),
    ),
    "LA_LIGA": LeagueConfig(
        key="LA_LIGA",
        name="La Liga",
        country="Spain",
        api_football_id=140,
        odds_api_sport_key="soccer_spain_la_liga",
        football_data_code="SP1",
        football_data_format="main",
        sample_teams=(
            "Alaves", "Athletic Club", "Atletico Madrid", "Barcelona", "Celta Vigo",
            "Elche", "Espanyol", "Getafe", "Girona", "Levante",
            "Mallorca", "Osasuna", "Rayo Vallecano", "Real Betis", "Real Madrid",
            "Real Oviedo", "Real Sociedad", "Sevilla", "Valencia", "Villarreal",
        ),
    ),
    "SERIE_A": LeagueConfig(
        key="SERIE_A",
        name="Serie A",
        country="Italy",
        api_football_id=135,
        odds_api_sport_key="soccer_italy_serie_a",
        football_data_code="I1",
        football_data_format="main",
        sample_teams=(
            "Atalanta", "Bologna", "Cagliari", "Como", "Cremonese",
            "Fiorentina", "Genoa", "Hellas Verona", "Inter", "Juventus",
            "Lazio", "Lecce", "Milan", "Napoli", "Parma",
            "Pisa", "Roma", "Sassuolo", "Torino", "Udinese",
        ),
    ),
    "BUNDESLIGA": LeagueConfig(
        key="BUNDESLIGA",
        name="Bundesliga",
        country="Germany",
        api_football_id=78,
        odds_api_sport_key="soccer_germany_bundesliga",
        football_data_code="D1",
        football_data_format="main",
        sample_teams=(
            "Augsburg", "Bayer Leverkusen", "Bayern Munich", "Borussia Dortmund",
            "Borussia Monchengladbach", "Eintracht Frankfurt", "FC Koln", "Freiburg",
            "Hamburger SV", "Heidenheim", "Hoffenheim", "Mainz", "RB Leipzig",
            "St Pauli", "Union Berlin", "VfB Stuttgart", "Werder Bremen", "Wolfsburg",
        ),
    ),
}


def selectable_leagues() -> list[str]:
    """Ligas que se ofrecen en la app/CLI: las que funcionan con fuentes gratuitas.

    Las competiciones internacionales (UCL) quedan fuera: requieren API-Football
    y su plan gratuito no cubre la temporada en curso. El código multiliga se
    conserva para quien tenga un plan de pago.

    Returns:
        Claves de liga en el orden del registro.
    """
    return [k for k, cfg in LEAGUES.items() if not cfg.is_international]


def get_league(key: str) -> LeagueConfig:
    """Obtiene la configuración de una liga por su clave.

    Args:
        key: Clave interna de la liga (insensible a mayúsculas).

    Returns:
        La ``LeagueConfig`` registrada.

    Raises:
        ConfigurationError: Si la liga no está registrada.
    """
    try:
        return LEAGUES[key.upper()]
    except KeyError as exc:
        available = ", ".join(sorted(LEAGUES))
        raise ConfigurationError(f"Liga desconocida '{key}'. Disponibles: {available}") from exc


def register_league(config: LeagueConfig, *, overwrite: bool = False) -> None:
    """Registra una liga nueva en tiempo de ejecución.

    Args:
        config: Configuración de la liga.
        overwrite: Si es ``True`` permite reemplazar una liga existente.

    Raises:
        ConfigurationError: Si la clave ya existe y ``overwrite`` es ``False``.
    """
    key = config.key.upper()
    if key in LEAGUES and not overwrite:
        raise ConfigurationError(f"La liga '{key}' ya está registrada")
    LEAGUES[key] = config


def current_season(today: date | None = None) -> int:
    """Año de inicio de la temporada en curso (julio en adelante = temporada nueva).

    Las ligas europeas arrancan en agosto y la Liga MX (Apertura) en julio, así
    que en noviembre de 2026 la temporada actual es la 2026/27 → ``2026``.

    Args:
        today: Fecha de referencia (por defecto, hoy).

    Returns:
        Año de inicio de la temporada.
    """
    d = today or date.today()
    return d.year if d.month >= 7 else d.year - 1
