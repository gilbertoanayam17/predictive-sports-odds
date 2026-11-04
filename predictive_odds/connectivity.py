"""Comprobación de conexión con las fuentes de datos reales (sin consumir créditos).

* The Odds API: ``/v4/sports`` es gratuito (no descuenta créditos) y devuelve en
  las cabeceras los créditos usados y restantes.
* API-Football: ``/status`` informa del plan y del uso del día (no cuenta como petición).
* football-data.co.uk: descarga el CSV de la temporada indicada (sin clave).
"""

from __future__ import annotations

from dataclasses import dataclass

from .data_fetcher import (
    AuthenticationError,
    DataFetchError,
    FootballDataCoUkProvider,
    HttpClient,
    HttpClientConfig,
    RateLimitError,
)
from .config import get_league

ODDS_API_SIGNUP = "https://the-odds-api.com/#get-access"
API_FOOTBALL_SIGNUP = "https://dashboard.api-football.com/register"


@dataclass(frozen=True)
class ConnectionCheck:
    """Resultado de comprobar una fuente.

    Attributes:
        service: Nombre de la fuente.
        ok: Si la conexión y la clave funcionan.
        message: Detalle legible (créditos restantes, error...).
        required: Si la fuente es imprescindible para el modo real.
    """

    service: str
    ok: bool
    message: str
    required: bool = True


def _client(base_url: str, headers: dict[str, str] | None = None) -> HttpClient:
    """Cliente sin caché ni reintentos largos (respuesta rápida en la UI)."""
    return HttpClient(base_url, headers=headers, config=HttpClientConfig(timeout=10.0, max_retries=1,
                                                                         backoff_factor=0.5, cache_dir=None))


def check_odds_api(api_key: str, client: HttpClient | None = None) -> ConnectionCheck:
    """Comprueba la clave de The Odds API sin gastar créditos.

    Args:
        api_key: Clave de the-odds-api.com.
        client: Cliente HTTP (inyectable en tests).

    Returns:
        Resultado con los créditos restantes si la clave es válida.
    """
    name = "The Odds API (momios)"
    if not api_key.strip():
        return ConnectionCheck(name, False, f"Falta la clave. Consíguela gratis en {ODDS_API_SIGNUP}")
    http = client or _client("https://api.the-odds-api.com/v4")
    try:
        sports = http.get_json("sports", {"apiKey": api_key.strip()}, use_cache=False)
    except AuthenticationError:
        return ConnectionCheck(name, False, "La clave no es válida o está desactivada")
    except RateLimitError:
        return ConnectionCheck(name, False, "Cuota agotada este mes (se renueva al inicio del ciclo)")
    except DataFetchError as exc:
        return ConnectionCheck(name, False, f"Sin conexión: {exc}")
    remaining = http.last_headers.get("x-requests-remaining", "?")
    used = http.last_headers.get("x-requests-used", "?")
    n = len(sports) if isinstance(sports, list) else 0
    return ConnectionCheck(name, True, f"Clave válida · créditos restantes: {remaining} (usados: {used}) · "
                                       f"{n} competiciones disponibles")


def check_api_football(api_key: str, client: HttpClient | None = None) -> ConnectionCheck:
    """Comprueba la clave de API-Football (opcional) sin gastar peticiones.

    Args:
        api_key: Clave de api-sports.io.
        client: Cliente HTTP (inyectable en tests).

    Returns:
        Resultado con el uso del día si la clave es válida.
    """
    name = "API-Football (calendario, lesiones, UCL)"
    if not api_key.strip():
        return ConnectionCheck(name, False, f"Opcional: sin ella no hay ajuste por lesiones ni UCL. "
                                            f"Registro gratis en {API_FOOTBALL_SIGNUP}", required=False)
    http = client or _client("https://v3.football.api-sports.io", {"x-apisports-key": api_key.strip()})
    try:
        data = http.get_json("status", use_cache=False)
    except AuthenticationError:
        return ConnectionCheck(name, False, "La clave no es válida", required=False)
    except DataFetchError as exc:
        return ConnectionCheck(name, False, f"Sin conexión: {exc}", required=False)
    errors = data.get("errors") if isinstance(data, dict) else None
    if errors:
        return ConnectionCheck(name, False, f"La API rechazó la clave: {errors}", required=False)
    resp = data.get("response", {}) if isinstance(data, dict) else {}
    req = resp.get("requests", {}) if isinstance(resp, dict) else {}
    plan = (resp.get("subscription") or {}).get("plan", "?") if isinstance(resp, dict) else "?"
    usage = f"peticiones hoy: {req.get('current', '?')}/{req.get('limit_day', '?')}"
    if str(plan).lower() == "free":
        return ConnectionCheck(name, False, f"Clave válida, pero el plan gratuito solo cubre temporadas antiguas: "
                                            f"no sirve para la temporada actual (lesiones, calendario). Puedes "
                                            f"dejarla vacía · {usage}", required=False)
    return ConnectionCheck(name, True, f"Clave válida · plan {plan} · {usage}", required=False)


def check_football_data(league_key: str, season: int, provider: FootballDataCoUkProvider | None = None
                        ) -> ConnectionCheck:
    """Comprueba que se puede descargar el histórico de football-data.co.uk.

    Args:
        league_key: Liga.
        season: Temporada.
        provider: Proveedor (inyectable en tests).

    Returns:
        Resultado con el número de partidos descargados.
    """
    name = "football-data.co.uk (histórico, sin clave)"
    league = get_league(league_key)
    if not league.football_data_code:
        return ConnectionCheck(name, False, f"{league.name} no está en football-data; requiere API-Football",
                               required=False)
    try:
        matches = (provider or FootballDataCoUkProvider()).fetch_matches(league, season)
    except DataFetchError as exc:
        return ConnectionCheck(name, False, f"No se pudo descargar: {exc}")
    return ConnectionCheck(name, True, f"{len(matches)} partidos de {league.name} {season}/{season + 1}")


def check_all(odds_key: str, api_football_key: str, league_key: str, season: int) -> list[ConnectionCheck]:
    """Comprueba todas las fuentes relevantes para la liga elegida."""
    return [check_odds_api(odds_key), check_api_football(api_football_key), check_football_data(league_key, season)]


__all__ = ["API_FOOTBALL_SIGNUP", "ODDS_API_SIGNUP", "ConnectionCheck", "check_all", "check_api_football",
           "check_football_data", "check_odds_api"]
