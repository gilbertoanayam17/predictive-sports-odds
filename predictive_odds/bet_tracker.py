"""Registro persistente de apuestas, liquidación y seguimiento del bankroll (SQLite).

Flujo típico:

1. Tras un análisis, :meth:`BetTracker.record_pick` / :meth:`record_parlay`
   guardan las apuestas sugeridas (idempotente: no duplica si se reejecuta).
2. :meth:`BetTracker.update_closing` guarda la probabilidad justa de cierre
   para medir el CLV.
3. Cuando hay resultados, :meth:`BetTracker.settle` liquida las apuestas
   pendientes (los parlays, cuando todas sus patas terminaron).
4. :meth:`BetTracker.summary` / :meth:`by_market` / :meth:`bankroll_history`
   devuelven el rendimiento real.

El sistema solo registra lo que el usuario decide apostar; nunca coloca apuestas.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from .backtester import selection_result, selection_wins, settlement_pnl
from .data_fetcher import TeamNameNormalizer
from .domain import STATUS_FINISHED, Selection
from .parlay_builder import Parlay, Pick, ValueBetFinder

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    bet_type      TEXT NOT NULL CHECK (bet_type IN ('single', 'parlay')),
    bet_key       TEXT NOT NULL UNIQUE,
    league        TEXT,
    description   TEXT NOT NULL,
    legs          TEXT NOT NULL,
    odds          REAL NOT NULL CHECK (odds > 1),
    model_prob    REAL NOT NULL,
    stake         REAL NOT NULL CHECK (stake > 0),
    status        TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'won', 'lost', 'void')),
    pnl           REAL,
    closing_prob  REAL,
    settled_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status);
"""


class TrackerError(Exception):
    """Error de lectura/escritura en el registro de apuestas."""


def selection_code(selection: Selection) -> str:
    """Serializa una selección como ``MERCADO|RESULTADO|LÍNEA``."""
    return f"{selection.market.value}|{selection.outcome.value}|{'' if selection.line is None else selection.line}"


def parse_selection_code(code: str) -> Selection:
    """Inverso de :func:`selection_code`.

    Raises:
        ValueError: Si el código no es válido.
    """
    parts = code.split("|")
    if len(parts) != 3:
        raise ValueError(f"Código de selección inválido: {code}")
    market, outcome, line = parts
    return Selection.from_codes(market, outcome, float(line) if line else None)


@dataclass(frozen=True)
class Leg:
    """Pata almacenada de una apuesta.

    ``home``/``away``/``kickoff`` permiten liquidar aunque la fuente de
    resultados use otros IDs que la de momios (p. ej. The Odds API frente a
    football-data.co.uk). En registros antiguos pueden faltar.
    """

    match_id: str
    selection: str
    odds: float
    fixture: str
    home: str = ""
    away: str = ""
    kickoff: str = ""

    def teams(self) -> tuple[str, str]:
        """``(local, visitante)``; si faltan, se deducen de ``fixture`` ("A vs B")."""
        if self.home and self.away:
            return self.home, self.away
        parts = self.fixture.split(" vs ")
        return (parts[0], parts[1]) if len(parts) == 2 else ("", "")


def _kickoff(value: Any) -> str:
    """Fecha de inicio en ISO (cadena vacía si no se conoce)."""
    if value is None:
        return ""
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return ""
    return "" if pd.isna(ts) else ts.isoformat()


def _now() -> str:
    """Marca de tiempo UTC en ISO-8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class BetTracker:
    """Registro de apuestas en SQLite."""

    def __init__(self, db_path: str | Path = "bets.sqlite") -> None:
        """Abre (o crea) la base de datos.

        Args:
            db_path: Ruta del archivo SQLite (``":memory:"`` para pruebas).

        Raises:
            TrackerError: Si no se puede abrir o inicializar.
        """
        self.db_path = str(db_path)
        try:
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA)
        except (sqlite3.Error, OSError) as exc:
            raise TrackerError(f"No se pudo abrir el registro {self.db_path}: {exc}") from exc

    def close(self) -> None:
        """Cierra la conexión."""
        self._conn.close()

    def __enter__(self) -> BetTracker:
        """Soporte de ``with``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Cierra al salir del bloque ``with``."""
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Transacción con traducción de errores."""
        try:
            with self._conn:
                yield self._conn
        except sqlite3.Error as exc:
            raise TrackerError(f"Error de base de datos: {exc}") from exc

    # ---------------------------------------------------------------- registro
    def _insert(self, bet_type: str, league: str, description: str, legs: Sequence[Leg],
                odds: float, model_prob: float, stake: float) -> int | None:
        """Inserta una apuesta; devuelve ``None`` si ya existía (idempotencia)."""
        if stake <= 0:
            raise ValueError("El stake debe ser positivo")
        key = bet_type + ":" + ";".join(sorted(f"{l.match_id}@{l.selection}" for l in legs))
        legs_json = json.dumps([l.__dict__ for l in legs], ensure_ascii=False)
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO bets (created_at, bet_type, bet_key, league, description, legs, odds,"
                " model_prob, stake) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_now(), bet_type, key, league, description, legs_json, odds, model_prob, stake))
            return int(cur.lastrowid) if cur.rowcount and cur.lastrowid is not None else None

    def record_pick(self, pick: Pick, stake: float) -> int | None:
        """Registra una apuesta sencilla.

        Args:
            pick: Pick apostado.
            stake: Importe apostado (unidades monetarias).

        Returns:
            ID de la apuesta, o ``None`` si ya estaba registrada.
        """
        leg = Leg(pick.match_id, selection_code(pick.selection), pick.odds, pick.fixture,
                  pick.home_team, pick.away_team, _kickoff(pick.kickoff))
        return self._insert("single", pick.league, pick.describe(), [leg], pick.odds, pick.model_prob, stake)

    def record_parlay(self, parlay: Parlay, stake: float) -> int | None:
        """Registra un parlay.

        Args:
            parlay: Parlay apostado.
            stake: Importe apostado.

        Returns:
            ID de la apuesta, o ``None`` si ya estaba registrada.
        """
        legs = [Leg(l.match_id, selection_code(l.selection), l.odds, l.fixture, l.home_team, l.away_team,
                    _kickoff(l.kickoff)) for l in parlay.legs]
        desc = " + ".join(f"{l.fixture}: {l.selection.label}" for l in parlay.legs)
        league = parlay.legs[0].league if parlay.legs else ""
        return self._insert("parlay", league, desc, legs, parlay.combined_odds, parlay.joint_prob, stake)

    def record_analysis(self, value_bets: Sequence[Pick], parlays: Sequence[Parlay], bankroll: float) -> int:
        """Registra todas las sugerencias de un análisis con su stake sugerido.

        Args:
            value_bets: Sencillas sugeridas.
            parlays: Parlays sugeridos.
            bankroll: Bankroll para convertir fracciones en importes.

        Returns:
            Número de apuestas nuevas registradas.
        """
        new = 0
        for p in value_bets:
            amount = round(p.stake_fraction * bankroll, 2)
            if amount > 0 and self.record_pick(p, amount) is not None:
                new += 1
        for par in parlays:
            amount = round(par.stake_fraction * bankroll, 2)
            if amount > 0 and self.record_parlay(par, amount) is not None:
                new += 1
        return new

    # ------------------------------------------------------------- liquidación
    def _rows(self, where: str = "", params: Sequence[object] = ()) -> list[sqlite3.Row]:
        """Consulta filas de la tabla de apuestas."""
        with self._tx() as conn:
            return list(conn.execute(f"SELECT * FROM bets {where} ORDER BY id", params))

    @staticmethod
    def _legs(row: sqlite3.Row) -> list[Leg]:
        """Decodifica las patas de una fila."""
        return [Leg(**d) for d in json.loads(row["legs"])]

    def settle(self, results: pd.DataFrame) -> int:
        """Liquida las apuestas pendientes cuyos partidos ya terminaron.

        Args:
            results: Partidos (esquema estándar); se usan los finalizados. Se
                emparejan por ``match_id`` y, si no coincide (otra fuente de
                datos), por equipos normalizados y fecha de inicio (±2 días).

        Returns:
            Número de apuestas liquidadas.
        """
        done = results[results["status"] == STATUS_FINISHED]
        scores: dict[str, tuple[int, int]] = {
            str(mid): (int(float(hg)), int(float(ag)))
            for mid, hg, ag in zip(done["match_id"], done["home_goals"], done["away_goals"])
        }
        # Índice por equipos para cuando los IDs de momios y resultados no coinciden.
        norm = TeamNameNormalizer()
        by_teams: dict[tuple[str, str], list[tuple[pd.Timestamp, tuple[int, int]]]] = {}
        dates = pd.to_datetime(done["date"], utc=True, errors="coerce") if "date" in done else pd.Series(dtype=object)
        if "date" in done:
            for h, a, d, hg, ag in zip(done["home_team"], done["away_team"], dates,
                                       done["home_goals"], done["away_goals"]):
                by_teams.setdefault((norm.key(str(h)), norm.key(str(a))), []).append(
                    (d, (int(float(hg)), int(float(ag)))))

        result_teams = sorted(set(done["home_team"].astype(str)) | set(done["away_team"].astype(str))) \
            if "home_team" in done else []

        def find(leg: Leg, created_at: str) -> tuple[int, int] | None:
            """Marcador de una pata: por ID y, si no, por equipos + fecha (±2 días)."""
            if leg.match_id in scores:
                return scores[leg.match_id]
            home, away = leg.teams()
            home = norm.match(home, result_teams) or home  # "Hull City" (momios) ↔ "Hull" (resultados)
            away = norm.match(away, result_teams) or away
            candidates = by_teams.get((norm.key(home), norm.key(away)), [])
            if leg.kickoff:
                ref = pd.Timestamp(leg.kickoff)
                ref = ref.tz_localize("UTC") if ref.tzinfo is None else ref
                close = [sc for d, sc in candidates if pd.notna(d) and abs((d - ref).days) <= 2]
            else:  # registros antiguos: primer partido entre ambos tras registrar la apuesta
                since = pd.Timestamp(created_at) - pd.Timedelta(days=1)
                close = [sc for d, sc in sorted(candidates, key=lambda c: c[0]) if pd.notna(d) and d >= since][:1]
            return close[0] if len(close) == 1 else None

        settled = 0
        for row in self._rows("WHERE status = 'pending'"):
            legs = self._legs(row)
            found = {l.match_id: find(l, row["created_at"]) for l in legs}
            if any(v is None for v in found.values()):
                continue
            scores.update({k: v for k, v in found.items() if v is not None})
            try:
                if len(legs) == 1:  # sencilla: admite push y medios resultados (DNB/AH)
                    result = selection_result(parse_selection_code(legs[0].selection), *scores[legs[0].match_id])
                else:
                    won_all = all(selection_wins(parse_selection_code(l.selection), *scores[l.match_id]) for l in legs)
                    result = 1.0 if won_all else -1.0
            except ValueError as exc:
                logger.error("Apuesta %s con selección inválida, se anula: %s", row["id"], exc)
                status, pnl = "void", 0.0
            else:
                status = "won" if result > 0 else "lost" if result < 0 else "void"
                pnl = settlement_pnl(result, row["odds"], row["stake"])
            with self._tx() as conn:
                conn.execute("UPDATE bets SET status = ?, pnl = ?, settled_at = ? WHERE id = ?",
                             (status, pnl, _now(), row["id"]))
            settled += 1
        return settled

    def void(self, bet_id: int) -> None:
        """Anula una apuesta (p. ej. partido suspendido): se devuelve el stake.

        Raises:
            TrackerError: Si la apuesta no existe.
        """
        with self._tx() as conn:
            cur = conn.execute("UPDATE bets SET status = 'void', pnl = 0, settled_at = ? WHERE id = ?",
                               (_now(), bet_id))
            if cur.rowcount == 0:
                raise TrackerError(f"No existe la apuesta {bet_id}")

    def update_closing(self, closing_odds: pd.DataFrame) -> int:
        """Guarda la probabilidad justa de cierre de las sencillas pendientes (para CLV).

        Args:
            closing_odds: Momios de cierre (esquema estándar).

        Returns:
            Número de apuestas actualizadas.
        """
        fair = ValueBetFinder().market_consensus(closing_odds)
        updated = 0
        for row in self._rows("WHERE bet_type = 'single' AND status = 'pending'"):
            leg = self._legs(row)[0]
            p = fair.get((leg.match_id, parse_selection_code(leg.selection)))
            if p is not None:
                with self._tx() as conn:
                    conn.execute("UPDATE bets SET closing_prob = ? WHERE id = ?", (p, row["id"]))
                updated += 1
        return updated

    # ------------------------------------------------------------- informes
    def bets(self) -> pd.DataFrame:
        """Todas las apuestas como DataFrame (con CLV calculado si hay cierre)."""
        rows = [dict(r) for r in self._rows()]
        df = pd.DataFrame(rows, columns=["id", "created_at", "bet_type", "bet_key", "league", "description", "legs",
                                         "odds", "model_prob", "stake", "status", "pnl", "closing_prob", "settled_at"])
        df["clv"] = df["odds"] * df["closing_prob"] - 1
        return df

    def summary(self, initial_bankroll: float | None = None) -> dict[str, float]:
        """Resumen del rendimiento.

        Args:
            initial_bankroll: Si se indica, se calcula el bankroll actual.

        Returns:
            Apuestas, pendientes, exposición abierta, beneficio, ROI, acierto y CLV medio.
        """
        df = self.bets()
        settled = df[df["status"].isin(["won", "lost"])]
        staked = float(settled["stake"].sum())
        out: dict[str, float] = {
            "apuestas": float(len(df)),
            "pendientes": float((df["status"] == "pending").sum()),
            "exposicion_abierta": float(df.loc[df["status"] == "pending", "stake"].sum()),
            "liquidadas": float(len(settled)),
            "total_apostado": staked,
            "beneficio": float(settled["pnl"].sum()),
            "roi": float(settled["pnl"].sum() / staked) if staked else 0.0,
            "acierto": float((settled["status"] == "won").mean()) if len(settled) else 0.0,
            "clv_medio": float(df["clv"].mean()) if df["clv"].notna().any() else float("nan"),
        }
        if initial_bankroll is not None:
            out["bankroll_actual"] = initial_bankroll + out["beneficio"]
        return out

    def by_market(self) -> pd.DataFrame:
        """Rendimiento por mercado (sencillas) y por tipo (parlays)."""
        df = self.bets()
        settled = df[df["status"].isin(["won", "lost"])].copy()
        if settled.empty:
            return pd.DataFrame(columns=["grupo", "apuestas", "stake", "beneficio", "roi"])
        settled["grupo"] = [
            "PARLAY" if t == "parlay" else json.loads(l)[0]["selection"].split("|")[0]
            for t, l in zip(settled["bet_type"], settled["legs"])
        ]
        g = settled.groupby("grupo")
        out = pd.DataFrame({"apuestas": g.size(), "stake": g["stake"].sum(), "beneficio": g["pnl"].sum()})
        out["roi"] = out["beneficio"] / out["stake"]
        return out.reset_index()

    def bankroll_history(self, initial_bankroll: float) -> pd.DataFrame:
        """Evolución del bankroll según el orden de liquidación.

        Args:
            initial_bankroll: Bankroll inicial.

        Returns:
            DataFrame ``settled_at, pnl, bankroll``.
        """
        df = self.bets()
        settled = df[df["status"].isin(["won", "lost", "void"])].sort_values(["settled_at", "id"])
        return pd.DataFrame({"settled_at": settled["settled_at"], "pnl": settled["pnl"],
                             "bankroll": initial_bankroll + np.cumsum(settled["pnl"].to_numpy(float))})


__all__ = ["BetTracker", "Leg", "TrackerError", "parse_selection_code", "selection_code"]
