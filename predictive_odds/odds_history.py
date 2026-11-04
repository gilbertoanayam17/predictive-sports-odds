"""Histórico de momios (snapshots en SQLite): movimiento de línea y cierre propio.

Ejecutando el análisis periódicamente (p. ej. cada pocas horas) y guardando
cada captura se obtiene:

* el **movimiento de línea** de cada selección (apertura → último precio);
* una **línea de cierre propia** (última captura antes del inicio) para medir
  el CLV de las apuestas registradas sin depender de fuentes de terceros.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd

from .data_fetcher import CLOSING_BOOKMAKER, validate_odds
from .domain import ODDS_COLUMNS, Selection
from .parlay_builder import ValueBetFinder

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    taken_at      TEXT NOT NULL,
    match_id      TEXT NOT NULL,
    league        TEXT,
    home_team     TEXT,
    away_team     TEXT,
    commence_time TEXT,
    bookmaker     TEXT NOT NULL,
    market        TEXT NOT NULL,
    selection     TEXT NOT NULL,
    line          REAL,
    odds          REAL NOT NULL CHECK (odds > 1),
    PRIMARY KEY (taken_at, match_id, bookmaker, market, selection, line)
);
CREATE INDEX IF NOT EXISTS idx_snap_match ON odds_snapshots(match_id, taken_at);
"""


class OddsHistoryError(Exception):
    """Error de lectura/escritura del histórico de momios."""


class OddsHistory:
    """Almacén de capturas de momios."""

    def __init__(self, db_path: str | Path = "odds_history.sqlite") -> None:
        """Abre (o crea) la base de datos.

        Args:
            db_path: Ruta del archivo SQLite (``":memory:"`` para pruebas).

        Raises:
            OddsHistoryError: Si no se puede abrir.
        """
        self.db_path = str(db_path)
        try:
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.executescript(SCHEMA)
        except (sqlite3.Error, OSError) as exc:
            raise OddsHistoryError(f"No se pudo abrir {self.db_path}: {exc}") from exc

    def close(self) -> None:
        """Cierra la conexión."""
        self._conn.close()

    def __enter__(self) -> OddsHistory:
        """Soporte de ``with``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Cierra al salir del bloque."""
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Transacción con traducción de errores."""
        try:
            with self._conn:
                yield self._conn
        except sqlite3.Error as exc:
            raise OddsHistoryError(f"Error de base de datos: {exc}") from exc

    def save_snapshot(self, odds: pd.DataFrame, taken_at: datetime | None = None) -> int:
        """Guarda una captura de momios (las filas inválidas se descartan).

        Args:
            odds: Momios en el esquema estándar.
            taken_at: Momento de la captura (por defecto, ahora en UTC).

        Returns:
            Filas guardadas.
        """
        if odds.empty:
            return 0
        clean = validate_odds(odds)
        clean = clean[clean["bookmaker"] != CLOSING_BOOKMAKER]
        ts = (taken_at or datetime.now(timezone.utc)).isoformat(timespec="seconds")
        kickoff = [None if pd.isna(t) else t.isoformat() for t in pd.to_datetime(clean["commence_time"], utc=True)]
        lines = [None if pd.isna(v) else float(v) for v in pd.to_numeric(clean["line"], errors="coerce")]
        rows = list(zip(
            [ts] * len(clean), clean["match_id"].astype(str), clean["league"].astype(str),
            clean["home_team"].astype(str), clean["away_team"].astype(str), kickoff,
            clean["bookmaker"].astype(str), clean["market"].astype(str), clean["selection"].astype(str),
            lines, clean["odds"].astype(float),
        ))
        with self._tx() as conn:
            conn.executemany("INSERT OR REPLACE INTO odds_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def snapshots(self, match_ids: list[str] | None = None) -> pd.DataFrame:
        """Todas las capturas (opcionalmente de ciertos partidos).

        Args:
            match_ids: Partidos a filtrar.

        Returns:
            DataFrame con ``taken_at`` + ``ODDS_COLUMNS``.
        """
        query = "SELECT * FROM odds_snapshots"
        params: tuple[str, ...] = ()
        if match_ids:
            query += f" WHERE match_id IN ({','.join('?' * len(match_ids))})"
            params = tuple(match_ids)
        with self._tx() as conn:
            rows = conn.execute(query + " ORDER BY taken_at", params)
            cols = [d[0] for d in rows.description]
            df = pd.DataFrame(rows.fetchall(), columns=cols)
        if not df.empty:
            df["taken_at"] = pd.to_datetime(df["taken_at"], utc=True)
            df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
        return df

    def line_movement(self, match_ids: list[str] | None = None) -> pd.DataFrame:
        """Movimiento del MEJOR precio por selección entre la primera y la última captura.

        Args:
            match_ids: Partidos a filtrar.

        Returns:
            DataFrame ``match_id, market, selection, line, apertura, ultimo, cambio_pct, capturas``
            ordenado por el mayor acortamiento (señal de dinero informado).
        """
        df = self.snapshots(match_ids)
        if df.empty:
            return pd.DataFrame(columns=["match_id", "market", "selection", "line", "apertura", "ultimo",
                                         "cambio_pct", "capturas"])
        best = pd.DataFrame(df.assign(line=df["line"].fillna(-1.0))
                            .groupby(["match_id", "market", "selection", "line", "taken_at"])["odds"].max()
                            .reset_index())
        g = best.sort_values("taken_at").groupby(["match_id", "market", "selection", "line"])
        out = g["odds"].agg(apertura="first", ultimo="last", capturas="count").reset_index()
        out["cambio_pct"] = 100 * (out["ultimo"] / out["apertura"] - 1)
        out["line"] = out["line"].where(out["line"] != -1.0)
        return out.sort_values("cambio_pct").reset_index(drop=True)

    def closing_odds(self, match_ids: list[str] | None = None) -> pd.DataFrame:
        """Última captura anterior al inicio de cada partido, como momios de cierre.

        Args:
            match_ids: Partidos a filtrar.

        Returns:
            Momios (``ODDS_COLUMNS``) de la última captura pre-partido de cada encuentro.
        """
        df = self.snapshots(match_ids)
        if df.empty:
            return pd.DataFrame(columns=ODDS_COLUMNS)
        pre = df[df["commence_time"].isna() | (df["taken_at"] <= df["commence_time"])]
        last = pre.groupby("match_id")["taken_at"].transform("max")
        return pre[pre["taken_at"] == last][ODDS_COLUMNS].reset_index(drop=True)

    def closing_consensus(self, match_ids: list[str] | None = None) -> dict[tuple[str, Selection], float]:
        """Probabilidad justa de cierre (sin margen) por (partido, selección)."""
        return ValueBetFinder().market_consensus(self.closing_odds(match_ids))


__all__ = ["OddsHistory", "OddsHistoryError"]
