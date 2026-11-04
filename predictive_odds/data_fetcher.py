"""Ingesta de datos: resultados históricos, calendario y momios.

Arquitectura (ver ``docs/fase1_reflexion.md`` para la justificación):

* :class:`HttpClient` — cliente HTTP robusto: timeouts, reintentos con backoff
  exponencial, respeto de ``Retry-After`` (429), *throttling* y caché en disco.
* Proveedores intercambiables detrás de dos interfaces abstractas:

  - :class:`ResultsProvider` (resultados/calendario):
    :class:`FootballDataCoUkProvider` (CSV gratuito, histórico + momios de
    cierre), :class:`ApiFootballProvider` (freemium, calendario, xG, lesiones).
  - :class:`OddsProvider` (momios en vivo): :class:`TheOddsApiProvider`.
  - :class:`SimulatedDataProvider` implementa ambas para pruebas/demos offline.

* :class:`DataFetcher` — fachada que combina proveedores, valida y normaliza
  los datos al esquema estándar (``MATCH_COLUMNS`` / ``ODDS_COLUMNS``) y
  empareja nombres de equipos entre fuentes distintas.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import json
import logging
import re
import time
import unicodedata
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson

from .config import LeagueConfig, get_league
from .domain import (
    MARKET_SIZE,
    MATCH_COLUMNS,
    ODDS_COLUMNS,
    STATUS_FINISHED,
    STATUS_SCHEDULED,
    Market,
    Selection,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Excepciones
# =============================================================================
class DataFetchError(Exception):
    """Error genérico al obtener datos de una fuente externa."""


class AuthenticationError(DataFetchError):
    """Credenciales inválidas, ausentes o sin permisos (HTTP 401/403)."""


class RateLimitError(DataFetchError):
    """Se excedió la cuota o el límite de peticiones del proveedor."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        """Inicializa el error.

        Args:
            message: Descripción del error.
            retry_after: Segundos sugeridos por el servidor antes de reintentar.
        """
        super().__init__(message)
        self.retry_after = retry_after


class DataValidationError(DataFetchError):
    """Los datos recibidos no cumplen el esquema esperado."""


class ProviderNotSupportedError(DataFetchError):
    """El proveedor no soporta la liga u operación solicitada."""


# =============================================================================
# Cliente HTTP
# =============================================================================
@dataclass
class HttpClientConfig:
    """Parámetros del cliente HTTP.

    Attributes:
        timeout: Timeout por petición en segundos.
        max_retries: Reintentos ante errores transitorios (red, 429, 5xx).
        backoff_factor: Base del backoff exponencial (segundos).
        min_interval: Intervalo mínimo entre peticiones (throttling).
        cache_dir: Directorio de caché en disco (``None`` la desactiva).
        cache_ttl: Vigencia de la caché en segundos.
        user_agent: Cabecera User-Agent.
    """

    timeout: float = 15.0
    max_retries: int = 3
    backoff_factor: float = 1.5
    min_interval: float = 0.0
    cache_dir: Path | None = None
    cache_ttl: float = 3600.0
    user_agent: str = "predictive-sports-odds/0.1"


class HttpClient:
    """Cliente HTTP con reintentos, backoff, throttling y caché opcional."""

    def __init__(
        self,
        base_url: str = "",
        headers: Mapping[str, str] | None = None,
        config: HttpClientConfig | None = None,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Crea el cliente.

        Args:
            base_url: URL base que se antepone a rutas relativas.
            headers: Cabeceras fijas (p. ej. API keys).
            config: Configuración de reintentos/caché.
            session: Sesión ``requests`` reutilizable (inyectable en tests).
            sleep: Función de espera (inyectable en tests).
        """
        self.base_url = base_url.rstrip("/")
        self.config = config or HttpClientConfig()
        self.session = session or requests.Session()
        self.headers: dict[str, str] = {"User-Agent": self.config.user_agent, **(headers or {})}
        self._sleep = sleep
        self._last_request_ts = 0.0
        self.last_headers: Mapping[str, str] = {}

    # ----------------------------------------------------------------- helpers
    def _url(self, path: str) -> str:
        """Construye la URL absoluta a partir de una ruta."""
        if path.startswith(("http://", "https://")):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def _throttle(self) -> None:
        """Respeta el intervalo mínimo entre peticiones."""
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self.config.min_interval:
            self._sleep(self.config.min_interval - elapsed)
        self._last_request_ts = time.monotonic()

    def _cache_path(self, url: str, params: Mapping[str, Any] | None) -> Path | None:
        """Calcula la ruta de caché de una petición (las API keys no se incluyen en el hash)."""
        if self.config.cache_dir is None:
            return None
        safe_params = {k: v for k, v in sorted((params or {}).items()) if "key" not in k.lower()}
        digest = hashlib.sha256(f"{url}|{json.dumps(safe_params, default=str)}".encode()).hexdigest()
        return Path(self.config.cache_dir) / f"{digest[:32]}.json"

    def _read_cache(self, path: Path | None) -> Any | None:
        """Lee una entrada de caché vigente, o ``None`` si no existe/caducó/está corrupta."""
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - float(payload["ts"]) <= self.config.cache_ttl:
                return payload["data"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Entrada de caché ilegible %s: %s", path, exc)
        return None

    def _write_cache(self, path: Path | None, data: Any) -> None:
        """Escribe una entrada en caché; los fallos solo se registran."""
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"ts": time.time(), "data": data}), encoding="utf-8")
        except (OSError, TypeError) as exc:
            logger.warning("No se pudo escribir la caché %s: %s", path, exc)

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        """Interpreta la cabecera ``Retry-After`` (segundos)."""
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None

    # --------------------------------------------------------------- requests
    def request(self, path: str, params: Mapping[str, Any] | None = None) -> requests.Response:
        """Ejecuta un GET con reintentos y manejo exhaustivo de errores.

        Args:
            path: Ruta relativa o URL absoluta.
            params: Parámetros de query.

        Returns:
            Respuesta HTTP exitosa (2xx/3xx).

        Raises:
            AuthenticationError: HTTP 401/403.
            RateLimitError: HTTP 429 tras agotar reintentos.
            DataFetchError: Errores de red, 5xx persistentes u otros 4xx.
        """
        url = self._url(path)
        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            backoff = self.config.backoff_factor * (2 ** (attempt - 1))
            self._throttle()
            try:
                resp = self.session.get(
                    url, params=dict(params or {}), headers=self.headers, timeout=self.config.timeout
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt >= attempts:
                    raise DataFetchError(f"Error de red persistente en {url}: {exc}") from exc
                logger.warning("Error de red (%s), reintento %d/%d en %.1fs", exc, attempt, attempts - 1, backoff)
                self._sleep(backoff)
                continue
            except requests.RequestException as exc:
                raise DataFetchError(f"Petición inválida a {url}: {exc}") from exc

            self.last_headers = resp.headers
            status = resp.status_code
            if status in (401, 403):
                raise AuthenticationError(f"HTTP {status} en {url}: credenciales inválidas o sin acceso")
            if status == 429:
                retry_after = self._parse_retry_after(resp.headers.get("Retry-After"))
                if attempt >= attempts:
                    raise RateLimitError(f"Límite de peticiones excedido en {url}", retry_after)
                wait = min(retry_after if retry_after is not None else backoff, 60.0)
                logger.warning("HTTP 429, esperando %.1fs (reintento %d/%d)", wait, attempt, attempts - 1)
                self._sleep(wait)
                continue
            if status >= 500:
                if attempt >= attempts:
                    raise DataFetchError(f"HTTP {status} persistente en {url}")
                logger.warning("HTTP %d, reintento %d/%d en %.1fs", status, attempt, attempts - 1, backoff)
                self._sleep(backoff)
                continue
            if status >= 400:
                raise DataFetchError(f"HTTP {status} en {url}: {resp.text[:200]}")
            return resp
        raise DataFetchError(f"No se pudo completar la petición a {url}")  # pragma: no cover

    def get_json(self, path: str, params: Mapping[str, Any] | None = None, use_cache: bool = True) -> Any:
        """GET que devuelve JSON decodificado (con caché opcional).

        Args:
            path: Ruta relativa o URL absoluta.
            params: Parámetros de query.
            use_cache: Si se debe consultar/escribir la caché.

        Returns:
            El JSON decodificado.

        Raises:
            DataValidationError: Si la respuesta no es JSON válido.
            DataFetchError: Ver :meth:`request`.
        """
        cache_path = self._cache_path(self._url(path), params) if use_cache else None
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached
        resp = self.request(path, params)
        try:
            data = resp.json()
        except ValueError as exc:
            raise DataValidationError(f"Respuesta no-JSON de {self._url(path)}: {resp.text[:200]}") from exc
        self._write_cache(cache_path, data)
        return data

    def get_text(self, path: str, params: Mapping[str, Any] | None = None, use_cache: bool = True) -> str:
        """GET que devuelve texto (p. ej. CSV), detectando la codificación.

        Args:
            path: Ruta relativa o URL absoluta.
            params: Parámetros de query.
            use_cache: Si se debe consultar/escribir la caché.

        Returns:
            Contenido de texto.
        """
        cache_path = self._cache_path(self._url(path), params) if use_cache else None
        cached = self._read_cache(cache_path)
        if isinstance(cached, str):
            return cached
        content = self.request(path, params).content
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:  # pragma: no cover - latin-1 decodifica cualquier byte
            raise DataValidationError("No se pudo decodificar la respuesta")
        self._write_cache(cache_path, text)
        return text


# =============================================================================
# Normalización de nombres de equipos
# =============================================================================
_STOPWORDS = re.compile(r"\b(fc|cf|afc|sc|cd|club|de|the|ac|ssc|as|calcio|cfc)\b")

DEFAULT_TEAM_ALIASES: dict[str, str] = {
    "man utd": "manchester united",
    "man united": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "wolverhampton": "wolverhampton wanderers",
    "nottm forest": "nottingham forest",
    "newcastle": "newcastle united",
    "west ham": "west ham united",
    "leeds": "leeds united",
    "brighton and hove albion": "brighton",
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "inter milan": "inter",
    "internazionale": "inter",
    "bayern munchen": "bayern munich",
    "atl madrid": "atletico madrid",
    "unam": "pumas unam",
    "pumas": "pumas unam",
    "u n a m": "pumas unam",
    "tigres": "tigres uanl",
    "uanl": "tigres uanl",
    "chivas": "guadalajara",
    "chivas guadalajara": "guadalajara",
    "atl san luis": "atletico san luis",
    "san luis": "atletico san luis",
    "santos": "santos laguna",
    "queretaro fc": "queretaro",
    "fc juarez": "juarez",
}


class TeamNameNormalizer:
    """Empareja nombres de equipos entre fuentes (alias + coincidencia difusa)."""

    def __init__(self, aliases: Mapping[str, str] | None = None, fuzzy_cutoff: float = 0.82) -> None:
        """Crea el normalizador.

        Args:
            aliases: Mapa ``clave_canónica -> clave_canónica`` adicional.
            fuzzy_cutoff: Similitud mínima (0-1) para aceptar un emparejamiento difuso.
        """
        self.aliases = {**DEFAULT_TEAM_ALIASES, **(aliases or {})}
        self.fuzzy_cutoff = fuzzy_cutoff

    @staticmethod
    def canonical(name: str) -> str:
        """Clave canónica: sin acentos, minúsculas, sin sufijos tipo 'FC'.

        Args:
            name: Nombre original.

        Returns:
            Clave normalizada.
        """
        text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
        text = text.lower().replace("&", " and ")
        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        text = _STOPWORDS.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()

    def key(self, name: str) -> str:
        """Clave canónica tras aplicar alias."""
        canon = self.canonical(name)
        return self.aliases.get(canon, canon)

    def match(self, name: str, candidates: Iterable[str]) -> str | None:
        """Busca el candidato que corresponde a ``name``.

        Args:
            name: Nombre a emparejar.
            candidates: Nombres de referencia.

        Returns:
            El candidato emparejado o ``None`` si no hay coincidencia fiable.
        """
        by_key = {self.key(c): c for c in candidates}
        target = self.key(name)
        if target in by_key:
            return by_key[target]
        close = difflib.get_close_matches(target, list(by_key), n=1, cutoff=self.fuzzy_cutoff)
        return by_key[close[0]] if close else None


# =============================================================================
# Validación de esquemas
# =============================================================================
def empty_matches() -> pd.DataFrame:
    """DataFrame vacío con el esquema de partidos."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in MATCH_COLUMNS})


def empty_odds() -> pd.DataFrame:
    """DataFrame vacío con el esquema de momios."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in ODDS_COLUMNS})


def validate_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y normaliza un DataFrame de partidos al esquema estándar.

    Elimina (registrando cuántas) filas inválidas: sin equipos, equipos
    iguales, partidos finalizados sin goles, goles negativos o duplicados.

    Args:
        df: Datos crudos.

    Returns:
        DataFrame limpio con exactamente ``MATCH_COLUMNS`` y tipos correctos.

    Raises:
        DataValidationError: Si faltan columnas obligatorias.
    """
    required = {"match_id", "date", "home_team", "away_team", "status"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Faltan columnas obligatorias en partidos: {sorted(missing)}")
    out = df.copy()
    for col in MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    for col in ("home_goals", "away_goals", "home_xg", "away_xg"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["match_id"] = out["match_id"].astype(str)
    out["home_team"] = out["home_team"].astype("string").str.strip()
    out["away_team"] = out["away_team"].astype("string").str.strip()

    n0 = len(out)
    finished = out["status"] == STATUS_FINISHED
    bad = (
        out["date"].isna()
        | out["home_team"].isna()
        | out["away_team"].isna()
        | (out["home_team"] == out["away_team"])
        | (finished & (out["home_goals"].isna() | out["away_goals"].isna()))
        | (out["home_goals"] < 0)
        | (out["away_goals"] < 0)
        | ~out["status"].isin([STATUS_FINISHED, STATUS_SCHEDULED])
    )
    out = out.loc[~bad.fillna(True)].drop_duplicates(subset="match_id", keep="last")
    if len(out) < n0:
        logger.warning("Validación de partidos: %d filas descartadas de %d", n0 - len(out), n0)
    out["home_team"] = out["home_team"].astype(str)
    out["away_team"] = out["away_team"].astype(str)
    return out[MATCH_COLUMNS].sort_values("date", kind="stable").reset_index(drop=True)


def validate_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y normaliza un DataFrame de momios al esquema estándar.

    Args:
        df: Datos crudos.

    Returns:
        DataFrame limpio con ``ODDS_COLUMNS``; descarta momios <= 1.0, mercados
        o selecciones desconocidas y líneas inválidas.

    Raises:
        DataValidationError: Si faltan columnas obligatorias.
    """
    required = {"home_team", "away_team", "bookmaker", "market", "selection", "odds"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Faltan columnas obligatorias en momios: {sorted(missing)}")
    out = df.copy()
    for col in ODDS_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out["odds"] = pd.to_numeric(out["odds"], errors="coerce")
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    out["commence_time"] = pd.to_datetime(out["commence_time"], utc=True, errors="coerce")

    def _valid_selection(row: pd.Series) -> bool:
        try:
            Selection.from_codes(str(row["market"]), str(row["selection"]), row["line"])
            return True
        except (ValueError, TypeError):
            return False

    n0 = len(out)
    ok = out["odds"].between(1.0, 1000.0, inclusive="neither") & out.apply(_valid_selection, axis=1).astype(bool)
    out = out.loc[ok]
    if len(out) < n0:
        logger.warning("Validación de momios: %d filas descartadas de %d", n0 - len(out), n0)
    return out[ODDS_COLUMNS].reset_index(drop=True)


# =============================================================================
# Interfaces de proveedores
# =============================================================================
class ResultsProvider(ABC):
    """Fuente de resultados históricos y calendario."""

    name: str = "results"

    @abstractmethod
    def fetch_matches(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        """Descarga los partidos (jugados y programados) de una temporada.

        Args:
            league: Liga a consultar.
            season: Año de inicio de la temporada (2025 = 2025/26).

        Returns:
            DataFrame con columnas ``MATCH_COLUMNS``.
        """

    def fetch_injuries(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        """Descarga lesiones/bajas. Por defecto no soportado.

        Raises:
            ProviderNotSupportedError: Siempre, salvo que el proveedor lo implemente.
        """
        raise ProviderNotSupportedError(f"{self.name} no ofrece datos de lesiones")


class OddsProvider(ABC):
    """Fuente de momios actuales."""

    name: str = "odds"

    @abstractmethod
    def fetch_odds(self, league: LeagueConfig) -> pd.DataFrame:
        """Descarga los momios actuales de los próximos partidos.

        Args:
            league: Liga a consultar.

        Returns:
            DataFrame con columnas ``ODDS_COLUMNS``.
        """


def _make_match_id(prefix: str, date: pd.Timestamp, home: str, away: str) -> str:
    """Genera un ID de partido determinista cuando la fuente no lo provee."""
    norm = TeamNameNormalizer.canonical
    return f"{prefix}-{date:%Y%m%d}-{norm(home).replace(' ', '_')}-{norm(away).replace(' ', '_')}"


# =============================================================================
# football-data.co.uk (CSV gratuito)
# =============================================================================
class FootballDataCoUkProvider(ResultsProvider):
    """Resultados históricos y momios de cierre desde football-data.co.uk.

    Gratuito, sin API key y estable desde hace más de 20 años. Ideal para
    entrenar y hacer *backtesting* (incluye momios de varias casas). No trae
    xG ni calendario futuro.
    """

    name = "football-data.co.uk"
    BASE_URL = "https://www.football-data.co.uk"
    #: Prefijos de columnas de momios por casa (formato "main").
    BOOKMAKER_PREFIXES: dict[str, str] = {"B365": "bet365", "PS": "pinnacle", "Avg": "average", "Max": "max"}

    def __init__(self, client: HttpClient | None = None) -> None:
        """Crea el proveedor.

        Args:
            client: Cliente HTTP (por defecto con caché de 12 h en ``.cache/``).
        """
        self.client = client or HttpClient(
            self.BASE_URL, config=HttpClientConfig(cache_dir=Path(".cache/football_data"), cache_ttl=12 * 3600)
        )

    def _download(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        """Descarga y parsea el CSV crudo de una temporada.

        Raises:
            ProviderNotSupportedError: Si la liga no está en football-data.
            DataValidationError: Si el CSV no se puede parsear.
        """
        if not league.football_data_code:
            raise ProviderNotSupportedError(f"{league.name} no está disponible en football-data.co.uk")
        if league.football_data_format == "extra":
            path = f"new/{league.football_data_code}.csv"
        else:
            code = f"{season % 100:02d}{(season + 1) % 100:02d}"
            path = f"mmz4281/{code}/{league.football_data_code}.csv"
        text = self.client.get_text(path)
        try:
            raw = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise DataValidationError(f"CSV inválido de football-data ({path}): {exc}") from exc
        raw.columns = [str(c).strip() for c in raw.columns]
        if league.football_data_format == "extra":
            raw = raw.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam", "HG": "FTHG", "AG": "FTAG"})
            if "Season" in raw.columns:
                wanted = {f"{season}/{season + 1}", f"{season}-{season + 1}", str(season), str(season + 1)}
                raw = raw[raw["Season"].astype(str).isin(wanted)]
        if not {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"} <= set(raw.columns):
            raise DataValidationError(f"Columnas inesperadas en {path}: {list(raw.columns)[:12]}")
        raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
        raw["date_parsed"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", utc=True)
        return raw.dropna(subset=["date_parsed"])

    def fetch_matches(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        """Descarga los resultados de una temporada (ver :class:`ResultsProvider`)."""
        raw = self._download(league, season)
        df = pd.DataFrame(
            {
                "match_id": [
                    _make_match_id(f"fd-{league.key}", d, h, a)
                    for d, h, a in zip(raw["date_parsed"], raw["HomeTeam"], raw["AwayTeam"])
                ],
                "league": league.key,
                "season": season,
                "date": raw["date_parsed"],
                "home_team": raw["HomeTeam"],
                "away_team": raw["AwayTeam"],
                "home_goals": raw["FTHG"],
                "away_goals": raw["FTAG"],
                "home_xg": np.nan,
                "away_xg": np.nan,
                "status": STATUS_FINISHED,
            }
        )
        return validate_matches(df)

    def fetch_historical_odds(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        """Extrae los momios pre-partido incluidos en el CSV (para backtesting).

        Args:
            league: Liga.
            season: Temporada.

        Returns:
            DataFrame con ``ODDS_COLUMNS`` (1X2 y Over/Under 2.5).
        """
        raw = self._download(league, season)
        rows: list[dict[str, Any]] = []
        for _, r in raw.iterrows():
            mid = _make_match_id(f"fd-{league.key}", r["date_parsed"], r["HomeTeam"], r["AwayTeam"])
            base = {"match_id": mid, "league": league.key, "home_team": r["HomeTeam"],
                    "away_team": r["AwayTeam"], "commence_time": r["date_parsed"]}
            for prefix, book in self.BOOKMAKER_PREFIXES.items():
                for col, market, sel, line in (
                    (f"{prefix}H", "1X2", "HOME", None), (f"{prefix}D", "1X2", "DRAW", None),
                    (f"{prefix}A", "1X2", "AWAY", None), (f"{prefix}>2.5", "OU", "OVER", 2.5),
                    (f"{prefix}<2.5", "OU", "UNDER", 2.5),
                ):
                    if col in raw.columns and pd.notna(r[col]):
                        rows.append({**base, "bookmaker": book, "market": market,
                                     "selection": sel, "line": line, "odds": r[col]})
        return validate_odds(pd.DataFrame(rows)) if rows else empty_odds()


# =============================================================================
# API-Football (api-sports.io, freemium)
# =============================================================================
class ApiFootballProvider(ResultsProvider):
    """Calendario, resultados, xG por partido y lesiones desde API-Football v3.

    Plan gratuito: ~100 peticiones/día. Una petición trae una temporada
    completa; el xG requiere una petición por partido (usar con moderación).
    """

    name = "api-football"
    BASE_URL = "https://v3.football.api-sports.io"
    FINISHED = {"FT", "AET", "PEN"}
    SCHEDULED = {"NS", "TBD"}

    def __init__(self, api_key: str, client: HttpClient | None = None) -> None:
        """Crea el proveedor.

        Args:
            api_key: Clave de api-sports.io.
            client: Cliente HTTP opcional.

        Raises:
            AuthenticationError: Si no se proporciona API key.
        """
        if not api_key:
            raise AuthenticationError("API-Football requiere una API key (variable API_FOOTBALL_KEY)")
        self.client = client or HttpClient(
            self.BASE_URL,
            headers={"x-apisports-key": api_key},
            config=HttpClientConfig(min_interval=6.5, cache_dir=Path(".cache/api_football"), cache_ttl=1800),
        )

    def _get(self, path: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        """GET con interpretación de los errores que API-Football devuelve con HTTP 200.

        Raises:
            AuthenticationError: Token inválido.
            RateLimitError: Cuota diaria agotada.
            DataFetchError: Otros errores reportados por la API.
            DataValidationError: Respuesta con forma inesperada.
        """
        data = self.client.get_json(path, params)
        if not isinstance(data, dict):
            raise DataValidationError(f"Respuesta inesperada de API-Football: {type(data).__name__}")
        errors = data.get("errors")
        if errors:
            text = json.dumps(errors).lower()
            if "token" in text or "key" in text:
                raise AuthenticationError(f"API-Football: {errors}")
            if "request" in text or "limit" in text:
                raise RateLimitError(f"API-Football: {errors}")
            raise DataFetchError(f"API-Football: {errors}")
        response = data.get("response")
        if not isinstance(response, list):
            raise DataValidationError("API-Football: campo 'response' ausente o inválido")
        return response

    def _require_id(self, league: LeagueConfig) -> int:
        """Devuelve el ID de la liga en API-Football o lanza error."""
        if league.api_football_id is None:
            raise ProviderNotSupportedError(f"{league.name} no tiene ID de API-Football configurado")
        return league.api_football_id

    def fetch_matches(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        """Descarga los partidos de una temporada (ver :class:`ResultsProvider`).

        Para partidos con prórroga se usa el marcador a 90' (``score.fulltime``),
        que es el que liquidan los mercados 1X2/OU/BTTS.
        """
        items = self._get("fixtures", {"league": self._require_id(league), "season": season})
        rows: list[dict[str, Any]] = []
        for it in items:
            try:
                status_short = it["fixture"]["status"]["short"]
                if status_short in self.FINISHED:
                    status = STATUS_FINISHED
                elif status_short in self.SCHEDULED:
                    status = STATUS_SCHEDULED
                else:
                    continue  # suspendido, cancelado, en juego...
                ft = (it.get("score") or {}).get("fulltime") or {}
                goals = it.get("goals") or {}
                rows.append({
                    "match_id": f"af-{it['fixture']['id']}",
                    "league": league.key,
                    "season": season,
                    "date": it["fixture"]["date"],
                    "home_team": it["teams"]["home"]["name"],
                    "away_team": it["teams"]["away"]["name"],
                    "home_goals": ft.get("home", goals.get("home")),
                    "away_goals": ft.get("away", goals.get("away")),
                    "home_xg": np.nan,
                    "away_xg": np.nan,
                    "status": status,
                })
            except (KeyError, TypeError) as exc:
                logger.warning("Fixture de API-Football mal formado, se omite: %s", exc)
        return validate_matches(pd.DataFrame(rows)) if rows else empty_matches()

    def fetch_fixture_xg(self, fixture_id: int) -> tuple[float | None, float | None]:
        """Obtiene el xG (local, visitante) de un partido, si la liga lo ofrece.

        Args:
            fixture_id: ID numérico del partido en API-Football.

        Returns:
            Tupla ``(xg_local, xg_visitante)``; ``None`` si no está disponible.
        """
        stats = self._get("fixtures/statistics", {"fixture": fixture_id})
        values: list[float | None] = []
        for team_block in stats[:2]:
            xg = None
            for s in team_block.get("statistics", []):
                if s.get("type") == "expected_goals" and s.get("value") is not None:
                    try:
                        xg = float(s["value"])
                    except (TypeError, ValueError):
                        xg = None
            values.append(xg)
        while len(values) < 2:
            values.append(None)
        return values[0], values[1]

    def enrich_with_xg(self, matches: pd.DataFrame, max_calls: int = 20) -> pd.DataFrame:
        """Rellena xG de los partidos más recientes respetando un presupuesto de llamadas.

        Args:
            matches: Partidos con IDs ``af-<id>``.
            max_calls: Máximo de peticiones a realizar.

        Returns:
            Copia de ``matches`` con xG donde fue posible.
        """
        out = matches.copy()
        todo = out[(out["status"] == STATUS_FINISHED) & out["home_xg"].isna()].sort_values("date", ascending=False)
        for idx in todo.index[:max_calls]:
            try:
                fixture_id = int(str(out.at[idx, "match_id"]).removeprefix("af-"))
                out.at[idx, "home_xg"], out.at[idx, "away_xg"] = self.fetch_fixture_xg(fixture_id)
            except RateLimitError:
                logger.warning("Cuota de API-Football agotada al cargar xG; se continúa sin él")
                break
            except (DataFetchError, ValueError) as exc:
                logger.warning("No se pudo obtener xG de %s: %s", out.at[idx, "match_id"], exc)
        return out

    def fetch_injuries(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        """Descarga bajas y dudas de jugadores.

        Returns:
            DataFrame con columnas ``team, player, type, reason, match_id``.
        """
        items = self._get("injuries", {"league": self._require_id(league), "season": season})
        rows = []
        for it in items:
            try:
                rows.append({
                    "team": it["team"]["name"],
                    "player": it["player"]["name"],
                    "type": it["player"].get("type"),
                    "reason": it["player"].get("reason"),
                    "match_id": f"af-{it['fixture']['id']}",
                })
            except (KeyError, TypeError) as exc:
                logger.warning("Lesión mal formada, se omite: %s", exc)
        return pd.DataFrame(rows, columns=["team", "player", "type", "reason", "match_id"])


# =============================================================================
# The Odds API (freemium)
# =============================================================================
class TheOddsApiProvider(OddsProvider):
    """Momios en vivo de múltiples casas desde The Odds API v4.

    Plan gratuito: 500 créditos/mes. Cada mercado x región consume 1 crédito
    por llamada; ``btts`` requiere el endpoint por evento (más caro).
    """

    name = "the-odds-api"
    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(
        self,
        api_key: str,
        regions: str = "eu,uk",
        markets: Sequence[str] = ("h2h", "totals"),
        include_btts: bool = False,
        client: HttpClient | None = None,
    ) -> None:
        """Crea el proveedor.

        Args:
            api_key: Clave de the-odds-api.com.
            regions: Regiones de casas de apuestas (``eu``, ``uk``, ``us``, ``au``).
            markets: Mercados del endpoint principal (``h2h``, ``totals``).
            include_btts: Si se consultan también los momios BTTS por evento.
            client: Cliente HTTP opcional.

        Raises:
            AuthenticationError: Si no se proporciona API key.
        """
        if not api_key:
            raise AuthenticationError("The Odds API requiere una API key (variable ODDS_API_KEY)")
        self.api_key = api_key
        self.regions = regions
        self.markets = tuple(markets)
        self.include_btts = include_btts
        self.client = client or HttpClient(
            self.BASE_URL, config=HttpClientConfig(cache_dir=Path(".cache/odds_api"), cache_ttl=600)
        )

    def _log_quota(self) -> None:
        """Registra los créditos restantes informados por la API."""
        remaining = self.client.last_headers.get("x-requests-remaining")
        if remaining is not None:
            logger.info("The Odds API: créditos restantes = %s", remaining)

    @staticmethod
    def parse_events(events: Any, league_key: str) -> pd.DataFrame:
        """Convierte la respuesta JSON de eventos al esquema ``ODDS_COLUMNS``.

        Args:
            events: Lista de eventos (o un único evento) de The Odds API.
            league_key: Clave interna de la liga.

        Returns:
            DataFrame de momios sin validar.

        Raises:
            DataValidationError: Si la respuesta no tiene la forma esperada.
        """
        if isinstance(events, dict):
            events = [events]
        if not isinstance(events, list):
            raise DataValidationError("The Odds API: se esperaba una lista de eventos")
        rows: list[dict[str, Any]] = []
        for ev in events:
            try:
                home, away = ev["home_team"], ev["away_team"]
                base = {"match_id": f"oa-{ev['id']}", "league": league_key, "home_team": home,
                        "away_team": away, "commence_time": ev["commence_time"]}
                for book in ev.get("bookmakers", []):
                    for mkt in book.get("markets", []):
                        for outcome in mkt.get("outcomes", []):
                            name, price = outcome.get("name"), outcome.get("price")
                            market, selection, line = None, None, None
                            if mkt.get("key") == "h2h":
                                market = "1X2"
                                selection = {home: "HOME", away: "AWAY", "Draw": "DRAW"}.get(name)
                            elif mkt.get("key") == "totals":
                                market, line = "OU", outcome.get("point")
                                selection = {"Over": "OVER", "Under": "UNDER"}.get(name)
                            elif mkt.get("key") == "btts":
                                market = "BTTS"
                                selection = {"Yes": "YES", "No": "NO"}.get(name)
                            if market and selection and price is not None:
                                rows.append({**base, "bookmaker": book.get("key", "unknown"), "market": market,
                                             "selection": selection, "line": line, "odds": price})
            except (KeyError, TypeError) as exc:
                logger.warning("Evento de The Odds API mal formado, se omite: %s", exc)
        return pd.DataFrame(rows, columns=ODDS_COLUMNS)

    def fetch_odds(self, league: LeagueConfig) -> pd.DataFrame:
        """Descarga momios actuales (ver :class:`OddsProvider`)."""
        if not league.odds_api_sport_key:
            raise ProviderNotSupportedError(f"{league.name} no tiene clave de The Odds API")
        params = {"apiKey": self.api_key, "regions": self.regions, "markets": ",".join(self.markets),
                  "oddsFormat": "decimal", "dateFormat": "iso"}
        events = self.client.get_json(f"sports/{league.odds_api_sport_key}/odds", params)
        self._log_quota()
        frames = [self.parse_events(events, league.key)]
        if self.include_btts:
            for ev in events if isinstance(events, list) else []:
                try:
                    detail = self.client.get_json(
                        f"sports/{league.odds_api_sport_key}/events/{ev['id']}/odds",
                        {**params, "markets": "btts"},
                    )
                    frames.append(self.parse_events(detail, league.key))
                except RateLimitError:
                    logger.warning("Cuota agotada al pedir BTTS; se continúa sin él")
                    break
                except (DataFetchError, KeyError) as exc:
                    logger.warning("No se pudieron obtener momios BTTS: %s", exc)
            self._log_quota()
        frames = [f for f in frames if not f.empty]
        return validate_odds(pd.concat(frames, ignore_index=True)) if frames else empty_odds()


# =============================================================================
# Proveedor simulado (demos, tests y desarrollo sin consumir cuotas)
# =============================================================================
@dataclass
class SimulationConfig:
    """Parámetros del generador sintético de ligas.

    Attributes:
        seed: Semilla global (resultados reproducibles).
        seasons: Temporadas a generar; la última queda parcialmente jugada.
        played_rounds_current: Jornadas jugadas de la temporada actual.
        base_log_rate: Log de la tasa base de goles por equipo.
        home_advantage: Ventaja de local (escala logarítmica).
        attack_sd: Dispersión de la fuerza ofensiva.
        defense_sd: Dispersión de la fuerza defensiva.
        season_drift_sd: Cambio de fuerza entre temporadas.
        xg_shape: Forma de la Gamma del ruido de xG (mayor = xG más preciso).
        bookmakers: Nombres de las casas simuladas.
        margin: Margen (overround) de las casas.
        bookmaker_noise: Error de estimación de cada casa (log-escala).
    """

    seed: int = 42
    seasons: tuple[int, ...] = (2024, 2025)
    played_rounds_current: int = 28
    base_log_rate: float = 0.20
    home_advantage: float = 0.25
    attack_sd: float = 0.30
    defense_sd: float = 0.22
    season_drift_sd: float = 0.08
    xg_shape: float = 6.0
    bookmakers: tuple[str, ...] = ("simbet", "oddsmaker", "luckyline")
    margin: float = 0.055
    bookmaker_noise: float = 0.07
    ou_lines: tuple[float, ...] = (1.5, 2.5, 3.5)


@dataclass
class _LeagueSim:
    """Estado interno simulado de una liga."""

    attack: dict[int, dict[str, float]] = field(default_factory=dict)
    defense: dict[int, dict[str, float]] = field(default_factory=dict)
    matches: dict[int, pd.DataFrame] = field(default_factory=dict)


class SimulatedDataProvider(ResultsProvider, OddsProvider):
    """Genera ligas sintéticas realistas con un modelo de Poisson latente.

    Los equipos tienen fuerzas ocultas; los resultados, el xG y los momios de
    varias casas (con margen y error de estimación propio) se derivan de ellas.
    Esto permite probar el flujo completo y medir si el sistema encuentra valor
    cuando las casas se equivocan.
    """

    name = "simulated"

    def __init__(self, config: SimulationConfig | None = None) -> None:
        """Crea el proveedor.

        Args:
            config: Parámetros de simulación.
        """
        self.config = config or SimulationConfig()
        self._state: dict[str, _LeagueSim] = {}

    # ------------------------------------------------------------ generation
    def _rng(self, *keys: int | str) -> np.random.Generator:
        """RNG determinista derivado de la semilla y claves arbitrarias."""
        ints = [k if isinstance(k, int) else zlib.crc32(k.encode()) for k in keys]
        return np.random.default_rng([self.config.seed, *ints])

    @staticmethod
    def _round_robin(teams: Sequence[str]) -> list[list[tuple[str, str]]]:
        """Calendario de doble vuelta por el método del círculo."""
        t: list[str | None] = list(teams)
        if len(t) % 2:
            t.append(None)
        n = len(t)
        rounds: list[list[tuple[str, str]]] = []
        for r in range(n - 1):
            pairs: list[tuple[str, str]] = []
            for i in range(n // 2):
                a, b = t[i], t[n - 1 - i]
                if a is not None and b is not None:
                    pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
            rounds.append(pairs)
            t = [t[0], t[-1], *t[1:-1]]
        return rounds + [[(b, a) for a, b in rnd] for rnd in rounds]

    def _ensure_strengths(self, league: LeagueConfig, season: int) -> None:
        """Genera (una vez) las fuerzas latentes por temporada."""
        st = self._state.setdefault(league.key, _LeagueSim())
        if season in st.attack:
            return
        teams = league.sample_teams
        if len(teams) < 4:
            raise ProviderNotSupportedError(f"{league.name} no tiene equipos de ejemplo para simular")
        rng = self._rng(league.key, "base")
        att = rng.normal(0, self.config.attack_sd, len(teams))
        dfn = rng.normal(0, self.config.defense_sd, len(teams))
        first = min(self.config.seasons)
        for s in range(first, season + 1):
            if s in st.attack:
                continue
            drift = self._rng(league.key, "drift", s)
            k = s - first
            att_s = att + drift.normal(0, self.config.season_drift_sd, len(teams)) * (k > 0)
            dfn_s = dfn + drift.normal(0, self.config.season_drift_sd, len(teams)) * (k > 0)
            st.attack[s] = {t: float(v - att_s.mean()) for t, v in zip(teams, att_s)}
            st.defense[s] = {t: float(v - dfn_s.mean()) for t, v in zip(teams, dfn_s)}

    def true_lambdas(self, league: LeagueConfig, season: int, home: str, away: str) -> tuple[float, float]:
        """Goles esperados "verdaderos" (desconocidos para el modelo).

        Args:
            league: Liga.
            season: Temporada.
            home: Local.
            away: Visitante.

        Returns:
            ``(lambda_local, lambda_visitante)``.
        """
        self._ensure_strengths(league, season)
        st = self._state[league.key]
        c = self.config
        lh = np.exp(c.base_log_rate + c.home_advantage + st.attack[season][home] + st.defense[season][away])
        la = np.exp(c.base_log_rate + st.attack[season][away] + st.defense[season][home])
        return float(lh), float(la)

    def fetch_matches(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        """Genera los partidos de una temporada (ver :class:`ResultsProvider`)."""
        if season not in self.config.seasons:
            raise ProviderNotSupportedError(f"Temporada {season} fuera de la simulación {self.config.seasons}")
        self._ensure_strengths(league, season)
        st = self._state[league.key]
        if season in st.matches:
            return st.matches[season].copy()
        rng = self._rng(league.key, "matches", season)
        current = max(self.config.seasons)
        start = pd.Timestamp(f"{season}-08-16 15:00", tz="UTC")
        rows: list[dict[str, Any]] = []
        for r, rnd in enumerate(self._round_robin(league.sample_teams)):
            date = start + pd.Timedelta(days=7 * r)
            played = season < current or r < self.config.played_rounds_current
            for home, away in rnd:
                lh, la = self.true_lambdas(league, season, home, away)
                row: dict[str, Any] = {
                    "match_id": f"sim-{league.key}-{season}-r{r + 1:02d}-{TeamNameNormalizer.canonical(home).replace(' ', '_')}",
                    "league": league.key, "season": season, "date": date,
                    "home_team": home, "away_team": away,
                    "home_goals": np.nan, "away_goals": np.nan, "home_xg": np.nan, "away_xg": np.nan,
                    "status": STATUS_FINISHED if played else STATUS_SCHEDULED,
                }
                if played:
                    shape = self.config.xg_shape
                    row["home_goals"] = int(rng.poisson(lh))
                    row["away_goals"] = int(rng.poisson(la))
                    row["home_xg"] = round(float(lh * rng.gamma(shape, 1 / shape)), 2)
                    row["away_xg"] = round(float(la * rng.gamma(shape, 1 / shape)), 2)
                rows.append(row)
        df = validate_matches(pd.DataFrame(rows))
        st.matches[season] = df
        return df.copy()

    def fetch_odds(self, league: LeagueConfig) -> pd.DataFrame:
        """Genera momios de la próxima jornada con margen y error por casa."""
        season = max(self.config.seasons)
        matches = self.fetch_matches(league, season)
        upcoming = matches[matches["status"] == STATUS_SCHEDULED]
        if upcoming.empty:
            return empty_odds()
        next_date = upcoming["date"].min()
        upcoming = upcoming[upcoming["date"] == next_date]
        c = self.config
        grid = np.arange(11)
        rows: list[dict[str, Any]] = []
        for _, m in upcoming.iterrows():
            lh, la = self.true_lambdas(league, season, m["home_team"], m["away_team"])
            for book in c.bookmakers:
                rng = self._rng(league.key, "odds", book, m["match_id"])
                blh = lh * np.exp(rng.normal(0, c.bookmaker_noise))
                bla = la * np.exp(rng.normal(0, c.bookmaker_noise))
                mat = np.outer(poisson.pmf(grid, blh), poisson.pmf(grid, bla))
                mat /= mat.sum()
                markets: list[tuple[str, str, float | None]] = [
                    ("1X2", "HOME", None), ("1X2", "DRAW", None), ("1X2", "AWAY", None),
                    ("BTTS", "YES", None), ("BTTS", "NO", None),
                ] + [("OU", s, ln) for ln in c.ou_lines for s in ("OVER", "UNDER")]
                for market, sel, line in markets:
                    p = float(mat[Selection.from_codes(market, sel, line).mask(10)].sum())
                    p = min(max(p, 0.01), 0.99)
                    odds = round(max(1.01, 1.0 / (p * (1 + c.margin))), 2)
                    rows.append({"match_id": m["match_id"], "league": league.key, "home_team": m["home_team"],
                                 "away_team": m["away_team"], "commence_time": m["date"], "bookmaker": book,
                                 "market": market, "selection": sel, "line": line, "odds": odds})
        return validate_odds(pd.DataFrame(rows))

    def fetch_injuries(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        """Sin lesiones en la simulación: devuelve un DataFrame vacío."""
        return pd.DataFrame(columns=["team", "player", "type", "reason", "match_id"])


# =============================================================================
# Fachada
# =============================================================================
class DataFetcher:
    """Fachada de ingesta: combina proveedores y entrega datos limpios y enlazados.

    Ejemplo::

        fetcher = DataFetcher(history_provider=FootballDataCoUkProvider(),
                              odds_provider=TheOddsApiProvider(api_key))
        history = fetcher.get_history("EPL", [2024, 2025])
        odds = fetcher.get_odds("EPL")
        fixtures = fetcher.fixtures_from_odds(odds, known_teams=history["home_team"].unique())
    """

    def __init__(
        self,
        history_provider: ResultsProvider,
        fixtures_provider: ResultsProvider | None = None,
        odds_provider: OddsProvider | None = None,
        normalizer: TeamNameNormalizer | None = None,
    ) -> None:
        """Crea la fachada.

        Args:
            history_provider: Fuente de resultados históricos.
            fixtures_provider: Fuente del calendario (por defecto la de históricos).
            odds_provider: Fuente de momios.
            normalizer: Normalizador de nombres de equipos.
        """
        self.history_provider = history_provider
        self.fixtures_provider = fixtures_provider or history_provider
        self.odds_provider = odds_provider
        self.normalizer = normalizer or TeamNameNormalizer()

    def get_history(self, league_key: str, seasons: Sequence[int]) -> pd.DataFrame:
        """Descarga partidos finalizados de varias temporadas.

        Las temporadas que fallan se registran y se omiten; solo se lanza error
        si no se pudo obtener ninguna.

        Args:
            league_key: Clave de la liga.
            seasons: Temporadas a cargar.

        Returns:
            Partidos finalizados ordenados por fecha.

        Raises:
            DataFetchError: Si ninguna temporada pudo descargarse.
        """
        league = get_league(league_key)
        frames: list[pd.DataFrame] = []
        errors: list[str] = []
        for season in seasons:
            try:
                df = self.history_provider.fetch_matches(league, season)
                frames.append(df[df["status"] == STATUS_FINISHED])
            except AuthenticationError:
                raise
            except DataFetchError as exc:
                logger.warning("No se pudo cargar %s %s: %s", league_key, season, exc)
                errors.append(f"{season}: {exc}")
        if not frames:
            raise DataFetchError(f"Sin datos históricos para {league_key}: {'; '.join(errors)}")
        history = pd.concat(frames, ignore_index=True)
        return validate_matches(history)

    def get_fixtures(self, league_key: str, season: int, next_round_only: bool = True) -> pd.DataFrame:
        """Obtiene los próximos partidos programados.

        Args:
            league_key: Clave de la liga.
            season: Temporada actual.
            next_round_only: Si solo se devuelve la fecha/jornada más próxima.

        Returns:
            Partidos con estado ``NS``.
        """
        league = get_league(league_key)
        df = self.fixtures_provider.fetch_matches(league, season)
        upcoming = df[df["status"] == STATUS_SCHEDULED].sort_values("date")
        if next_round_only and not upcoming.empty:
            first = upcoming["date"].min()
            upcoming = upcoming[upcoming["date"] <= first + pd.Timedelta(days=4)]
        return upcoming.reset_index(drop=True)

    def get_odds(self, league_key: str) -> pd.DataFrame:
        """Descarga momios actuales.

        Args:
            league_key: Clave de la liga.

        Returns:
            Momios validados.

        Raises:
            DataFetchError: Si no hay proveedor de momios configurado.
        """
        if self.odds_provider is None:
            raise DataFetchError("No hay proveedor de momios configurado")
        return self.odds_provider.fetch_odds(get_league(league_key))

    def get_injuries(self, league_key: str, season: int) -> pd.DataFrame:
        """Descarga lesiones si el proveedor de calendario lo soporta (si no, vacío)."""
        try:
            return self.fixtures_provider.fetch_injuries(get_league(league_key), season)
        except ProviderNotSupportedError as exc:
            logger.info("%s", exc)
            return pd.DataFrame(columns=["team", "player", "type", "reason", "match_id"])

    def align_team_names(self, df: pd.DataFrame, known_teams: Iterable[str]) -> pd.DataFrame:
        """Renombra equipos de ``df`` a los nombres usados en el histórico.

        Args:
            df: DataFrame con columnas ``home_team``/``away_team``.
            known_teams: Nombres de referencia (los del histórico).

        Returns:
            Copia con nombres alineados; los no emparejados se mantienen y se avisa.
        """
        known = list(dict.fromkeys(known_teams))
        out = df.copy()
        mapping: dict[str, str] = {}
        for name in pd.unique(out[["home_team", "away_team"]].values.ravel()):
            match = self.normalizer.match(str(name), known)
            if match is None:
                logger.warning("Equipo sin emparejar con el histórico: '%s'", name)
            mapping[str(name)] = match or str(name)
        out["home_team"] = out["home_team"].map(mapping)
        out["away_team"] = out["away_team"].map(mapping)
        return out

    def link_odds_to_fixtures(
        self, odds: pd.DataFrame, fixtures: pd.DataFrame, max_delta: pd.Timedelta = pd.Timedelta(hours=36)
    ) -> pd.DataFrame:
        """Asigna a cada momio el ``match_id`` del partido correspondiente del calendario.

        El emparejamiento usa nombres normalizados y la cercanía de la hora de inicio.

        Args:
            odds: Momios (de cualquier proveedor).
            fixtures: Calendario de referencia.
            max_delta: Diferencia horaria máxima aceptada.

        Returns:
            Momios con ``match_id``/equipos del calendario; los no emparejados se descartan.
        """
        if odds.empty or fixtures.empty:
            return empty_odds()
        key = self.normalizer.key
        index: dict[tuple[str, str], list[pd.Series]] = {}
        for _, f in fixtures.iterrows():
            index.setdefault((key(f["home_team"]), key(f["away_team"])), []).append(f)
        teams = pd.unique(fixtures[["home_team", "away_team"]].values.ravel())
        aligned = self.align_team_names(odds, teams)
        out_rows: list[pd.Series] = []
        unmatched = 0
        for _, row in aligned.iterrows():
            candidates = index.get((key(row["home_team"]), key(row["away_team"])), [])
            best = None
            for f in candidates:
                if pd.isna(row["commence_time"]) or abs(f["date"] - row["commence_time"]) <= max_delta:
                    best = f
                    break
            if best is None:
                unmatched += 1
                continue
            new = row.copy()
            new["match_id"], new["home_team"], new["away_team"] = best["match_id"], best["home_team"], best["away_team"]
            out_rows.append(new)
        if unmatched:
            logger.warning("%d momios no se pudieron enlazar con el calendario", unmatched)
        return pd.DataFrame(out_rows, columns=ODDS_COLUMNS).reset_index(drop=True)

    def fixtures_from_odds(self, odds: pd.DataFrame, known_teams: Iterable[str] | None = None) -> pd.DataFrame:
        """Construye el calendario a partir de los eventos de momios.

        Útil cuando no se dispone de API de calendario: los eventos de The Odds
        API ya identifican local, visitante y hora de inicio.

        Args:
            odds: Momios validados.
            known_teams: Si se indica, alinea nombres con el histórico.

        Returns:
            DataFrame de partidos programados (``MATCH_COLUMNS``).
        """
        if odds.empty:
            return empty_matches()
        src = self.align_team_names(odds, known_teams) if known_teams is not None else odds
        events = src.drop_duplicates("match_id")
        df = pd.DataFrame({
            "match_id": events["match_id"], "league": events["league"], "season": np.nan,
            "date": events["commence_time"], "home_team": events["home_team"],
            "away_team": events["away_team"], "status": STATUS_SCHEDULED,
        })
        return validate_matches(df)


def complete_markets(odds: pd.DataFrame) -> pd.DataFrame:
    """Filtra los grupos de momios (casa/mercado/línea) que están completos.

    Un grupo está completo si contiene todos los resultados del mercado
    (3 para 1X2, 2 para OU/BTTS); solo así se puede quitar el margen.

    Args:
        odds: Momios validados.

    Returns:
        Subconjunto con grupos completos.
    """
    if odds.empty:
        return odds
    sizes = odds["market"].map({m.value: n for m, n in MARKET_SIZE.items()})
    line_key = odds["line"].fillna(-1.0)
    counts = odds.groupby([odds["match_id"], odds["bookmaker"], odds["market"], line_key])["selection"].transform("nunique")
    return odds[counts == sizes]


__all__ = [
    "ApiFootballProvider", "AuthenticationError", "DataFetchError", "DataFetcher", "DataValidationError",
    "FootballDataCoUkProvider", "HttpClient", "HttpClientConfig", "OddsProvider", "ProviderNotSupportedError",
    "RateLimitError", "ResultsProvider", "SimulatedDataProvider", "SimulationConfig", "TeamNameNormalizer",
    "TheOddsApiProvider", "complete_markets", "validate_matches", "validate_odds", "Market",
]
