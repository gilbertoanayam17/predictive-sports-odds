"""Fuentes de xG y fusión con el histórico de resultados.

* :func:`merge_xg` — une el xG de CUALQUIER fuente a un histórico (por nombres
  normalizados y fecha cercana). Es la pieza que permite combinar, p. ej.,
  football-data.co.uk (resultados + momios) con xG de otra fuente.
* :class:`UnderstatProvider` — lector EXPERIMENTAL de understat.com (EPL y
  otras grandes ligas europeas; no cubre Liga MX).

.. warning::
   El lector de Understat no se ha podido verificar en vivo desde el entorno
   de desarrollo (acceso de red bloqueado). Se apoya en el formato conocido de
   la web (``datesData`` embebido como JSON escapado en el HTML) y en el
   endpoint JSON alternativo ``getLeagueData``; si el sitio cambia, lanza
   :class:`DataValidationError` con un mensaje claro en lugar de devolver
   datos corruptos. El scraping puede estar sujeto a los términos de uso del sitio.
"""

from __future__ import annotations

import codecs
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import LeagueConfig
from .data_fetcher import (
    STATUS_FINISHED,
    STATUS_SCHEDULED,
    DataFetchError,
    DataValidationError,
    HttpClient,
    HttpClientConfig,
    ProviderNotSupportedError,
    ResultsProvider,
    TeamNameNormalizer,
    empty_matches,
    validate_matches,
)

logger = logging.getLogger(__name__)

#: Claves de liga en Understat.
UNDERSTAT_LEAGUES: dict[str, str] = {"EPL": "EPL"}

_DATES_RE = re.compile(r"datesData\s*=\s*JSON\.parse\('(?P<data>.*?)'\)", re.DOTALL)


def parse_understat_html(html: str) -> list[dict[str, Any]]:
    """Extrae ``datesData`` de la página de liga de Understat.

    Args:
        html: HTML de ``/league/<liga>/<temporada>``.

    Returns:
        Lista de partidos en el formato crudo de Understat.

    Raises:
        DataValidationError: Si el bloque no existe o no es JSON válido.
    """
    match = _DATES_RE.search(html)
    if not match:
        raise DataValidationError("Understat: no se encontró 'datesData' (¿cambió el formato del sitio?)")
    try:
        decoded = codecs.decode(match.group("data"), "unicode_escape")
        data = json.loads(decoded)
    except (ValueError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Understat: datesData ilegible: {exc}") from exc
    if not isinstance(data, list):
        raise DataValidationError("Understat: datesData no es una lista")
    return data


def understat_to_matches(items: list[dict[str, Any]], league_key: str, season: int) -> pd.DataFrame:
    """Convierte partidos crudos de Understat al esquema estándar.

    Args:
        items: Partidos crudos (``id``, ``isResult``, ``h``, ``a``, ``goals``, ``xG``, ``datetime``).
        league_key: Clave interna de la liga.
        season: Temporada.

    Returns:
        DataFrame con ``MATCH_COLUMNS`` (incluye xG).
    """
    rows = []
    for it in items:
        try:
            done = bool(it.get("isResult"))
            rows.append({
                "match_id": f"us-{it['id']}", "league": league_key, "season": season,
                "date": pd.Timestamp(it["datetime"], tz="UTC"),
                "home_team": it["h"]["title"], "away_team": it["a"]["title"],
                "home_goals": float(it["goals"]["h"]) if done else np.nan,
                "away_goals": float(it["goals"]["a"]) if done else np.nan,
                "home_xg": float(it["xG"]["h"]) if done and it["xG"]["h"] is not None else np.nan,
                "away_xg": float(it["xG"]["a"]) if done and it["xG"]["a"] is not None else np.nan,
                "status": STATUS_FINISHED if done else STATUS_SCHEDULED,
            })
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Partido de Understat mal formado, se omite: %s", exc)
    return validate_matches(pd.DataFrame(rows)) if rows else empty_matches()


class UnderstatProvider(ResultsProvider):
    """Resultados + xG desde understat.com (EXPERIMENTAL, ver advertencia del módulo)."""

    name = "understat"
    BASE_URL = "https://understat.com"

    def __init__(self, client: HttpClient | None = None) -> None:
        """Crea el proveedor (con caché de 12 h y throttling de 3 s por cortesía).

        Args:
            client: Cliente HTTP opcional.
        """
        self.client = client or HttpClient(
            self.BASE_URL,
            config=HttpClientConfig(min_interval=3.0, cache_dir=Path(".cache/understat"), cache_ttl=12 * 3600),
        )

    def fetch_matches(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        """Descarga resultados y xG de una temporada.

        Intenta primero la página HTML (``datesData``) y, si su formato no se
        reconoce, el endpoint JSON ``getLeagueData``.

        Raises:
            ProviderNotSupportedError: Si la liga no está en Understat.
            DataValidationError: Si ningún formato es reconocible.
        """
        code = UNDERSTAT_LEAGUES.get(league.key)
        if code is None:
            raise ProviderNotSupportedError(f"{league.name} no está disponible en Understat")
        try:
            items = parse_understat_html(self.client.get_text(f"league/{code}/{season}"))
        except DataValidationError as html_exc:
            logger.info("Formato HTML no reconocido (%s); probando getLeagueData", html_exc)
            try:
                payload = self.client.get_json(f"getLeagueData/{code}/{season}")
            except DataFetchError as exc:
                raise DataValidationError(f"Understat: ningún formato disponible ({html_exc}; {exc})") from exc
            dates = payload.get("dates") if isinstance(payload, dict) else None
            if not isinstance(dates, list):
                raise DataValidationError("Understat: respuesta de getLeagueData sin 'dates'") from html_exc
            items = dates
        return understat_to_matches(items, league.key, season)


def merge_xg(matches: pd.DataFrame, xg: pd.DataFrame, normalizer: TeamNameNormalizer | None = None,
             max_days: int = 2, overwrite: bool = False) -> pd.DataFrame:
    """Añade xG de otra fuente a un histórico de partidos.

    Empareja por (local, visitante) normalizados y fecha a ≤ ``max_days`` días.
    Además, verifica que el marcador coincida cuando ambas fuentes lo tienen:
    si no coincide, no se fusiona (evita mezclar partidos distintos).

    Args:
        matches: Histórico destino (esquema estándar).
        xg: Partidos con ``home_xg``/``away_xg`` (esquema estándar).
        normalizer: Normalizador de nombres.
        max_days: Diferencia máxima de fechas.
        overwrite: Si se sobrescribe el xG existente.

    Returns:
        Copia de ``matches`` con el xG fusionado.
    """
    norm = normalizer or TeamNameNormalizer()
    out = matches.copy()
    src = xg.dropna(subset=["home_xg", "away_xg"])
    if src.empty or out.empty:
        return out
    known = pd.unique(out[["home_team", "away_team"]].values.ravel())
    names = {n: (norm.match(str(n), known) or str(n)) for n in pd.unique(src[["home_team", "away_team"]].values.ravel())}
    index: dict[tuple[str, str], list[pd.Series]] = {}
    for _, r in src.iterrows():
        index.setdefault((norm.key(names[r["home_team"]]), norm.key(names[r["away_team"]])), []).append(r)
    merged = mismatched = 0
    for i, row in out.iterrows():
        if not overwrite and pd.notna(row["home_xg"]):
            continue
        for cand in index.get((norm.key(row["home_team"]), norm.key(row["away_team"])), []):
            if abs((cand["date"] - row["date"]).days) > max_days:
                continue
            if pd.notna(row["home_goals"]) and pd.notna(cand["home_goals"]) and (
                    row["home_goals"] != cand["home_goals"] or row["away_goals"] != cand["away_goals"]):
                mismatched += 1
                continue
            out.at[i, "home_xg"], out.at[i, "away_xg"] = cand["home_xg"], cand["away_xg"]
            merged += 1
            break
    if mismatched:
        logger.warning("merge_xg: %d candidatos descartados por marcador distinto", mismatched)
    logger.info("merge_xg: xG añadido a %d de %d partidos", merged, len(out))
    return out


class XgEnrichedProvider(ResultsProvider):
    """Envuelve un proveedor de resultados y le añade xG de otra fuente.

    Si la fuente de xG falla, se registra un aviso y se devuelven los
    resultados sin xG (el modelo funciona igual, usando goles).
    """

    name = "xg-enriched"

    def __init__(self, base: ResultsProvider, xg_source: ResultsProvider,
                 normalizer: TeamNameNormalizer | None = None) -> None:
        """Crea el envoltorio.

        Args:
            base: Proveedor principal (resultados, p. ej. football-data.co.uk).
            xg_source: Proveedor con xG (p. ej. Understat).
            normalizer: Normalizador de nombres.
        """
        self.base = base
        self.xg_source = xg_source
        self.normalizer = normalizer or TeamNameNormalizer()

    def fetch_matches(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        """Resultados del proveedor base con el xG fusionado cuando es posible."""
        matches = self.base.fetch_matches(league, season)
        try:
            xg = self.xg_source.fetch_matches(league, season)
        except DataFetchError as exc:
            logger.warning("Sin xG de %s para %s %s: %s", self.xg_source.name, league.key, season, exc)
            return matches
        return merge_xg(matches, xg, self.normalizer)

    def fetch_injuries(self, league: LeagueConfig, season: int) -> pd.DataFrame:
        """Delegado al proveedor base."""
        return self.base.fetch_injuries(league, season)


__all__ = ["UNDERSTAT_LEAGUES", "UnderstatProvider", "XgEnrichedProvider", "merge_xg", "parse_understat_html", "understat_to_matches"]
