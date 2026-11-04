"""Estudio por liga y mercado: ¿dónde tiene ventaja real el modelo?

Ejecuta un backtest de varias temporadas por liga, agrega el rendimiento por
mercado y emite un *perfil* con los mercados recomendados. El perfil se
guarda en JSON y la app/CLI lo aplican automáticamente
(``ValueBetConfig.allowed_markets``).

Criterio del veredicto:

* **CLV medio > 0** con al menos ``min_bets`` apuestas. El CLV converge mucho
  más rápido que el ROI (que con unos cientos de apuestas es casi ruido).
* En ligas cuyo histórico solo trae momios de cierre (formato "extra" de
  football-data, p. ej. Liga MX) el CLV no es comparable: se exige ROI > 0 y el
  doble de muestra, y el informe lo indica.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from .backtester import BacktestConfig, Backtester
from .config import ConfigurationError, current_season, get_league
from .data_fetcher import DataFetchError, DataFetcher, ResultsProvider
from .statistical_models import InsufficientDataError

logger = logging.getLogger(__name__)

DEFAULT_PROFILE_PATH = Path("league_profiles.json")


@dataclass(frozen=True)
class MarketStats:
    """Rendimiento de un mercado en el backtest."""

    market: str
    bets: int
    staked: float
    profit: float
    roi: float
    clv: float | None
    clv_positive: float | None
    recommended: bool
    reason: str


@dataclass
class LeagueProfile:
    """Perfil recomendado para una liga."""

    league: str
    seasons: list[int]
    markets: list[str]
    clv_comparable: bool
    stats: list[MarketStats] = field(default_factory=list)
    generated: str = ""

    @property
    def enabled(self) -> bool:
        """Si hay al menos un mercado recomendado."""
        return bool(self.markets)


@dataclass(frozen=True)
class StudyConfig:
    """Parámetros del estudio.

    Attributes:
        seasons_back: Temporadas anteriores a la actual que se incluyen.
        min_bets: Apuestas mínimas por mercado para poder recomendarlo.
        use_ml: Si el backtest usa el ensamble con ML (más lento).
    """

    seasons_back: int = 3
    min_bets: int = 80
    use_ml: bool = False


def _market_stats(bets: pd.DataFrame, min_bets: int, clv_comparable: bool) -> list[MarketStats]:
    """Agrega las apuestas por mercado y decide la recomendación."""
    out: list[MarketStats] = []
    singles = bets[bets["type"] == "single"] if "type" in bets else bets
    for market, g in singles.groupby("market"):
        staked = float(g["stake"].sum())
        profit = float(g["pnl"].sum())
        roi = profit / staked if staked else 0.0
        clv_vals = g["clv"].dropna()
        clv = float(clv_vals.mean()) if len(clv_vals) else None
        clv_pos = float((clv_vals > 0).mean()) if len(clv_vals) else None
        n = len(g)
        if clv_comparable and clv is not None:
            ok = n >= min_bets and clv > 0
            reason = (f"CLV {clv:+.1%} con {n} apuestas" if n >= min_bets
                      else f"muestra insuficiente ({n} < {min_bets})")
        else:
            ok = n >= 2 * min_bets and roi > 0
            reason = (f"ROI {roi:+.1%} con {n} apuestas (CLV no comparable)" if n >= 2 * min_bets
                      else f"muestra insuficiente ({n} < {2 * min_bets}; CLV no comparable)")
        out.append(MarketStats(str(market), n, staked, profit, roi, clv, clv_pos, ok, reason))
    return sorted(out, key=lambda m: m.bets, reverse=True)


def study_league(provider: ResultsProvider, league_key: str, config: StudyConfig = StudyConfig(),
                 season: int | None = None) -> LeagueProfile:
    """Backtest multi-temporada de una liga y perfil recomendado.

    Args:
        provider: Proveedor con resultados y ``fetch_historical_odds`` (football-data o simulado).
        league_key: Liga.
        config: Parámetros del estudio.
        season: Temporada actual (por defecto la en curso).

    Returns:
        :class:`LeagueProfile`.

    Raises:
        DataFetchError: Si no hay datos.
        InsufficientDataError: Si el histórico es demasiado corto.
    """
    league = get_league(league_key)
    current = season if season is not None else current_season()
    seasons = list(range(current - config.seasons_back, current + 1))
    history = DataFetcher(provider).get_history(league_key, seasons)
    frames = []
    for s in seasons:
        try:
            frames.append(provider.fetch_historical_odds(league, s))  # type: ignore[attr-defined]
        except DataFetchError as exc:
            logger.warning("Sin momios históricos de %s %s: %s", league_key, s, exc)
    if not frames:
        raise DataFetchError(f"Sin momios históricos para {league.name}")
    odds = pd.concat(frames, ignore_index=True)
    result = Backtester(history, odds, BacktestConfig(use_ml=config.use_ml)).run()
    clv_comparable = league.football_data_format != "extra"
    stats = _market_stats(result.bets, config.min_bets, clv_comparable)
    return LeagueProfile(
        league=league_key, seasons=seasons, markets=[m.market for m in stats if m.recommended],
        clv_comparable=clv_comparable, stats=stats,
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def run_study(provider: ResultsProvider, leagues: Sequence[str], config: StudyConfig = StudyConfig(),
              season: int | None = None) -> tuple[dict[str, LeagueProfile], dict[str, str]]:
    """Estudia varias ligas; las que fallan se informan sin detener al resto.

    Returns:
        ``(perfiles, errores)``.
    """
    profiles: dict[str, LeagueProfile] = {}
    errors: dict[str, str] = {}
    for key in leagues:
        try:
            profiles[key] = study_league(provider, key, config, season)
        except (ConfigurationError, DataFetchError, InsufficientDataError, AttributeError) as exc:
            logger.warning("Estudio de %s omitido: %s", key, exc)
            errors[key] = str(exc)
    return profiles, errors


def study_frame(profiles: dict[str, LeagueProfile]) -> pd.DataFrame:
    """Tabla plana (liga × mercado) para mostrar."""
    rows = [{"liga": p.league, "mercado": m.market, "apuestas": m.bets, "roi": m.roi, "clv": m.clv,
             "clv_positivo": m.clv_positive, "recomendado": m.recommended, "motivo": m.reason}
            for p in profiles.values() for m in p.stats]
    return pd.DataFrame(rows, columns=["liga", "mercado", "apuestas", "roi", "clv", "clv_positivo",
                                       "recomendado", "motivo"])


def save_profiles(profiles: dict[str, LeagueProfile], path: str | Path = DEFAULT_PROFILE_PATH) -> Path:
    """Guarda (fusionando con los existentes) los perfiles en JSON.

    Raises:
        OSError: Si no se puede escribir.
    """
    target = Path(path)
    existing = {k: asdict(v) for k, v in load_profiles(target).items()}
    existing.update({k: asdict(v) for k, v in profiles.items()})
    target.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def load_profiles(path: str | Path = DEFAULT_PROFILE_PATH) -> dict[str, LeagueProfile]:
    """Carga los perfiles guardados (vacío si no hay archivo o está corrupto)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, LeagueProfile] = {}
    for key, data in raw.items() if isinstance(raw, dict) else []:
        try:
            stats = [MarketStats(**m) for m in data.get("stats", [])]
            out[key] = LeagueProfile(data["league"], list(data["seasons"]), list(data["markets"]),
                                     bool(data.get("clv_comparable", True)), stats, data.get("generated", ""))
        except (KeyError, TypeError) as exc:
            logger.warning("Perfil de %s ilegible, se ignora: %s", key, exc)
    return out


def allowed_markets_for(league_key: str, path: str | Path = DEFAULT_PROFILE_PATH) -> tuple[str, ...] | None:
    """Mercados permitidos para una liga según su perfil (``None`` si no hay perfil).

    Un perfil sin mercados recomendados devuelve una tupla vacía: no se sugieren apuestas.
    """
    profile = load_profiles(path).get(league_key)
    return None if profile is None else tuple(profile.markets)


__all__ = ["DEFAULT_PROFILE_PATH", "LeagueProfile", "MarketStats", "StudyConfig", "allowed_markets_for",
           "load_profiles", "run_study", "save_profiles", "study_frame", "study_league"]
