"""Análisis y modelado estadístico.

Contenido:

* :class:`HeuristicAnalyzer` — análisis base: tabla, forma (general/local/
  visitante), H2H, diferencial de goles y fuerza del calendario.
* :class:`EloRatingSystem` — rating dinámico para medir el nivel del rival.
* :class:`FeatureEngineer` — features pre-partido calculadas en orden
  cronológico (sin fuga de información del futuro).
* :class:`PoissonModel` — modelo de Dixon-Coles (Poisson bivariado con
  corrección de marcadores bajos), decaimiento temporal y mezcla de xG.
* :class:`MLOutcomeModel` — Regresión Logística / Random Forest para 1X2,
  Over/Under 2.5 y BTTS.
* :class:`EnsemblePredictor` — combina Poisson + ML en una :class:`MatchPrediction`.
* :func:`evaluate_predictions` — log-loss / Brier para comparar modelos.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .domain import STATUS_FINISHED, FloatMatrix, Market, MatchPrediction, Outcome, Selection

logger = logging.getLogger(__name__)

Venue = Literal["all", "home", "away"]


class ModelNotFittedError(RuntimeError):
    """Se intentó predecir con un modelo no entrenado."""


class InsufficientDataError(ValueError):
    """No hay datos suficientes para entrenar un modelo de forma fiable."""


def _finished(matches: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo partidos finalizados con goles válidos, ordenados por fecha."""
    m = matches[(matches["status"] == STATUS_FINISHED) & matches["home_goals"].notna() & matches["away_goals"].notna()]
    return m.sort_values("date", kind="stable").reset_index(drop=True)


def _points(gf: float, ga: float) -> int:
    """Puntos obtenidos (3/1/0) según goles a favor y en contra."""
    return 3 if gf > ga else (1 if gf == ga else 0)


# =============================================================================
# Análisis heurístico
# =============================================================================
@dataclass(frozen=True)
class FormSummary:
    """Forma reciente de un equipo.

    Attributes:
        team: Equipo.
        venue: ``all``, ``home`` o ``away``.
        results: Secuencia de resultados, el más reciente al final (p. ej. ``"WWDLW"``).
        points: Puntos obtenidos.
        ppg: Puntos por partido.
        goals_for: Goles a favor.
        goals_against: Goles en contra.
        matches: Partidos considerados.
    """

    team: str
    venue: Venue
    results: str
    points: int
    ppg: float
    goals_for: int
    goals_against: int
    matches: int

    @property
    def goal_diff(self) -> int:
        """Diferencial de goles."""
        return self.goals_for - self.goals_against


@dataclass(frozen=True)
class H2HSummary:
    """Historial de enfrentamientos directos desde la óptica de ``team_a``."""

    team_a: str
    team_b: str
    matches: int
    a_wins: int
    draws: int
    b_wins: int
    a_goals: int
    b_goals: int
    avg_total_goals: float
    btts_rate: float
    last_results: tuple[str, ...]


@dataclass(frozen=True)
class TeamSnapshot:
    """Fotografía de un equipo antes de un partido."""

    team: str
    position: int | None
    points: int
    played: int
    goal_diff: int
    form_overall: FormSummary
    form_venue: FormSummary
    strength_of_schedule: float
    adjusted_goal_diff: float


@dataclass(frozen=True)
class MatchReport:
    """Informe heurístico de un partido.

    Attributes:
        home: Datos del local.
        away: Datos del visitante.
        h2h: Enfrentamientos directos.
        score: Índice heurístico en [-1, 1] (>0 favorece al local).
        lean: Tendencia derivada del índice.
        notes: Observaciones legibles.
    """

    home: TeamSnapshot
    away: TeamSnapshot
    h2h: H2HSummary
    score: float
    lean: Outcome
    notes: tuple[str, ...]


class HeuristicAnalyzer:
    """Análisis base a partir de resultados históricos (sin modelos)."""

    def __init__(self, matches: pd.DataFrame) -> None:
        """Crea el analizador.

        Args:
            matches: Partidos con el esquema estándar (se usan los finalizados).
        """
        self.matches = _finished(matches)

    def _team_matches(self, team: str, venue: Venue = "all", season: int | None = None,
                      before: pd.Timestamp | None = None) -> pd.DataFrame:
        """Partidos de un equipo, normalizados a columnas ``gf/ga/opponent/is_home``."""
        m = self.matches
        if season is not None:
            m = m[m["season"] == season]
        if before is not None:
            m = m[m["date"] < before]
        home = m[m["home_team"] == team].assign(gf=lambda d: d["home_goals"], ga=lambda d: d["away_goals"],
                                                 opponent=lambda d: d["away_team"], is_home=True)
        away = m[m["away_team"] == team].assign(gf=lambda d: d["away_goals"], ga=lambda d: d["home_goals"],
                                                 opponent=lambda d: d["home_team"], is_home=False)
        parts = {"all": [home, away], "home": [home], "away": [away]}[venue]
        return pd.concat(parts).sort_values("date", kind="stable")

    def standings(self, season: int | None = None) -> pd.DataFrame:
        """Tabla de posiciones.

        Args:
            season: Temporada (por defecto la más reciente).

        Returns:
            DataFrame ordenado con ``pos, team, P, W, D, L, GF, GA, GD, Pts, PPG``.
        """
        if self.matches.empty:
            return pd.DataFrame(columns=["pos", "team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts", "PPG"])
        season = self.matches["season"].max() if season is None else season
        m = self.matches[self.matches["season"] == season]
        records = []
        for side, gf, ga in (("home_team", "home_goals", "away_goals"), ("away_team", "away_goals", "home_goals")):
            d = pd.DataFrame({"team": m[side], "GF": m[gf], "GA": m[ga]})
            records.append(d)
        r = pd.concat(records)
        r["W"] = (r["GF"] > r["GA"]).astype(int)
        r["D"] = (r["GF"] == r["GA"]).astype(int)
        r["L"] = (r["GF"] < r["GA"]).astype(int)
        t = r.groupby("team").agg(P=("GF", "size"), W=("W", "sum"), D=("D", "sum"), L=("L", "sum"),
                                  GF=("GF", "sum"), GA=("GA", "sum"))
        t["GD"] = t["GF"] - t["GA"]
        t["Pts"] = 3 * t["W"] + t["D"]
        t["PPG"] = (t["Pts"] / t["P"]).round(2)
        t = t.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index()
        t.insert(0, "pos", np.arange(1, len(t) + 1))
        return t.astype({c: int for c in ["P", "W", "D", "L", "GF", "GA", "GD", "Pts"]})

    def form(self, team: str, n: int = 5, venue: Venue = "all", before: pd.Timestamp | None = None) -> FormSummary:
        """Forma de los últimos ``n`` partidos.

        Args:
            team: Equipo.
            n: Número de partidos.
            venue: ``all`` (general), ``home`` (como local) o ``away`` (como visitante).
            before: Solo partidos anteriores a esta fecha.

        Returns:
            Resumen de forma (vacío si no hay partidos).
        """
        tm = self._team_matches(team, venue, before=before).tail(n)
        res = "".join("W" if gf > ga else "D" if gf == ga else "L" for gf, ga in zip(tm["gf"], tm["ga"]))
        pts = sum(_points(gf, ga) for gf, ga in zip(tm["gf"], tm["ga"]))
        k = len(tm)
        return FormSummary(team, venue, res, int(pts), round(pts / k, 2) if k else 0.0,
                           int(tm["gf"].sum()), int(tm["ga"].sum()), k)

    def head_to_head(self, team_a: str, team_b: str, n: int = 10) -> H2HSummary:
        """Historial de enfrentamientos directos (cualquier sede).

        Args:
            team_a: Equipo de referencia.
            team_b: Rival.
            n: Máximo de partidos recientes.

        Returns:
            Resumen H2H desde la óptica de ``team_a``.
        """
        m = self.matches
        h2h = m[((m["home_team"] == team_a) & (m["away_team"] == team_b))
                | ((m["home_team"] == team_b) & (m["away_team"] == team_a))].tail(n)
        a_goals = np.where(h2h["home_team"] == team_a, h2h["home_goals"], h2h["away_goals"])
        b_goals = np.where(h2h["home_team"] == team_a, h2h["away_goals"], h2h["home_goals"])
        k = len(h2h)
        return H2HSummary(
            team_a, team_b, k,
            int((a_goals > b_goals).sum()), int((a_goals == b_goals).sum()), int((a_goals < b_goals).sum()),
            int(a_goals.sum()), int(b_goals.sum()),
            float((a_goals + b_goals).mean()) if k else 0.0,
            float(((a_goals > 0) & (b_goals > 0)).mean()) if k else 0.0,
            tuple(f"{int(x)}-{int(y)}" for x, y in zip(a_goals, b_goals)),
        )

    def strength_of_schedule(self, team: str, n: int = 5, season: int | None = None) -> tuple[float, float]:
        """Nivel de los últimos ``n`` rivales y diferencial ajustado por rival.

        El diferencial ajustado suma a cada resultado la diferencia entre los
        PPG del rival y la media de la liga: golear a un equipo fuerte vale más.

        Args:
            team: Equipo.
            n: Partidos a considerar.
            season: Temporada de la tabla de referencia.

        Returns:
            ``(ppg_medio_rivales, diferencial_ajustado_por_partido)``.
        """
        table = self.standings(season)
        if table.empty:
            return 0.0, 0.0
        ppg = dict(zip(table["team"], table["PPG"]))
        avg = float(table["PPG"].mean())
        tm = self._team_matches(team).tail(n)
        if tm.empty:
            return avg, 0.0
        opp = np.array([ppg.get(o, avg) for o in tm["opponent"]])
        adj = (tm["gf"].to_numpy() - tm["ga"].to_numpy()) + (opp - avg)
        return float(opp.mean()), float(adj.mean())

    def snapshot(self, team: str, venue: Venue, n: int = 5, season: int | None = None) -> TeamSnapshot:
        """Construye la fotografía de un equipo (tabla + forma + calendario)."""
        table = self.standings(season)
        row = table[table["team"] == team]
        sos, adj = self.strength_of_schedule(team, n, season)
        return TeamSnapshot(
            team=team,
            position=int(row["pos"].iloc[0]) if not row.empty else None,
            points=int(row["Pts"].iloc[0]) if not row.empty else 0,
            played=int(row["P"].iloc[0]) if not row.empty else 0,
            goal_diff=int(row["GD"].iloc[0]) if not row.empty else 0,
            form_overall=self.form(team, n, "all"),
            form_venue=self.form(team, n, venue),
            strength_of_schedule=round(sos, 2),
            adjusted_goal_diff=round(adj, 2),
        )

    def match_report(self, home: str, away: str, n: int = 5, season: int | None = None) -> MatchReport:
        """Informe heurístico completo de un partido.

        El índice combina (pesos fijos y transparentes): forma por sede (35%),
        forma general (25%), diferencial ajustado por rival (20%),
        PPG en la tabla (10%) y H2H (10%).

        Args:
            home: Local.
            away: Visitante.
            n: Ventana de forma.
            season: Temporada de la tabla.

        Returns:
            Informe con índice y tendencia (no se usa para apostar por sí solo).
        """
        h, a = self.snapshot(home, "home", n, season), self.snapshot(away, "away", n, season)
        h2h = self.head_to_head(home, away)
        ppg_h = h.points / h.played if h.played else 0.0
        ppg_a = a.points / a.played if a.played else 0.0
        h2h_term = (h2h.a_wins - h2h.b_wins) / h2h.matches if h2h.matches else 0.0
        score = (
            0.35 * (h.form_venue.ppg - a.form_venue.ppg) / 3
            + 0.25 * (h.form_overall.ppg - a.form_overall.ppg) / 3
            + 0.20 * float(np.tanh((h.adjusted_goal_diff - a.adjusted_goal_diff) / 3))
            + 0.10 * (ppg_h - ppg_a) / 3
            + 0.10 * h2h_term
        )
        lean = Outcome.HOME if score > 0.08 else Outcome.AWAY if score < -0.08 else Outcome.DRAW
        notes = []
        if h.form_venue.matches and h.form_venue.ppg >= 2.2:
            notes.append(f"{home} es muy fuerte como local ({h.form_venue.results}).")
        if a.form_venue.matches and a.form_venue.ppg <= 0.8:
            notes.append(f"{away} sufre como visitante ({a.form_venue.results}).")
        if h2h.matches >= 3 and h2h.avg_total_goals >= 3.0:
            notes.append(f"H2H con muchos goles ({h2h.avg_total_goals:.1f} por partido).")
        if abs(h.strength_of_schedule - a.strength_of_schedule) > 0.4:
            harder = home if h.strength_of_schedule > a.strength_of_schedule else away
            notes.append(f"{harder} viene de un calendario más exigente.")
        return MatchReport(h, a, h2h, round(float(score), 3), lean, tuple(notes))


# =============================================================================
# Elo
# =============================================================================
@dataclass
class EloRatingSystem:
    """Rating Elo estilo *World Football Elo* con multiplicador por margen.

    Attributes:
        k_factor: Velocidad de actualización.
        home_advantage: Puntos Elo de ventaja de local.
        initial_rating: Rating inicial de un equipo nuevo.
        season_regression: Fracción de regresión a la media entre temporadas.
    """

    k_factor: float = 20.0
    home_advantage: float = 65.0
    initial_rating: float = 1500.0
    season_regression: float = 0.25
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        """Rating actual de un equipo (inicial si es nuevo)."""
        return self.ratings.get(team, self.initial_rating)

    def expected_home(self, home: str, away: str, neutral: bool = False) -> float:
        """Resultado esperado del local (1=victoria, 0.5=empate) según Elo."""
        diff = self.rating(home) + (0.0 if neutral else self.home_advantage) - self.rating(away)
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update(self, home: str, away: str, home_goals: int, away_goals: int, neutral: bool = False) -> None:
        """Actualiza los ratings tras un partido.

        Args:
            home: Local.
            away: Visitante.
            home_goals: Goles del local.
            away_goals: Goles del visitante.
            neutral: Si se jugó en campo neutral.
        """
        expected = self.expected_home(home, away, neutral)
        actual = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        gd = abs(home_goals - away_goals)
        g = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = self.k_factor * g * (actual - expected)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta

    def regress_to_mean(self) -> None:
        """Aplica regresión a la media (al cambiar de temporada)."""
        for team, r in self.ratings.items():
            self.ratings[team] = r + self.season_regression * (self.initial_rating - r)


# =============================================================================
# Features
# =============================================================================
FEATURE_COLUMNS: list[str] = [
    "elo_diff", "elo_expected_home",
    "home_form_ppg", "away_form_ppg", "home_venue_ppg", "away_venue_ppg",
    "home_gf_avg", "home_ga_avg", "away_gf_avg", "away_ga_avg",
    "home_xgf_avg", "home_xga_avg", "away_xgf_avg", "away_xga_avg",
    "home_sos", "away_sos", "home_season_ppg", "away_season_ppg",
    "h2h_home_ppg", "h2h_goal_diff",
]


@dataclass
class _TeamEntry:
    """Un partido en el historial de un equipo."""

    season: int
    is_home: bool
    gf: float
    ga: float
    xgf: float
    xga: float
    points: int
    opp_elo: float


class FeatureEngineer:
    """Calcula features pre-partido en orden cronológico (sin *data leakage*).

    Tras :meth:`fit_transform`, el estado interno refleja todo el histórico y
    :meth:`features_for` produce las features de un partido futuro.
    """

    def __init__(self, form_window: int = 5, h2h_window: int = 6, elo: EloRatingSystem | None = None) -> None:
        """Crea el generador.

        Args:
            form_window: Partidos usados para la forma.
            h2h_window: Enfrentamientos directos considerados.
            elo: Sistema Elo (se crea uno por defecto).
        """
        self.form_window = form_window
        self.h2h_window = h2h_window
        self.elo = elo or EloRatingSystem()
        self._history: dict[str, deque[_TeamEntry]] = {}
        self._h2h: dict[frozenset[str], deque[tuple[str, float, float]]] = {}
        self._season: int | None = None

    def _reset(self) -> None:
        """Reinicia el estado."""
        self.elo.ratings.clear()
        self._history.clear()
        self._h2h.clear()
        self._season = None

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        """Media que devuelve NaN para listas vacías."""
        return float(np.mean(values)) if len(values) else float("nan")

    def _team_block(self, team: str, prefix: str, venue_home: bool, season: int | None) -> dict[str, float]:
        """Features de un equipo a partir de su historial."""
        hist = list(self._history.get(team, ()))
        last = hist[-self.form_window:]
        venue = [e for e in hist if e.is_home == venue_home][-self.form_window:]
        season_entries = [e for e in hist if e.season == season]
        return {
            f"{prefix}_form_ppg": self._mean([e.points for e in last]),
            f"{prefix}_venue_ppg": self._mean([e.points for e in venue]),
            f"{prefix}_gf_avg": self._mean([e.gf for e in last]),
            f"{prefix}_ga_avg": self._mean([e.ga for e in last]),
            f"{prefix}_xgf_avg": self._mean([e.xgf for e in last]),
            f"{prefix}_xga_avg": self._mean([e.xga for e in last]),
            f"{prefix}_sos": self._mean([e.opp_elo - self.elo.initial_rating for e in last]),
            f"{prefix}_season_ppg": self._mean([e.points for e in season_entries]),
            f"{prefix}_games": float(len(hist)),
        }

    def _features(self, home: str, away: str, season: int | None, neutral: bool = False) -> dict[str, float]:
        """Features del partido según el estado actual."""
        feats: dict[str, float] = {
            "elo_diff": self.elo.rating(home) - self.elo.rating(away),
            "elo_expected_home": self.elo.expected_home(home, away, neutral),
        }
        feats.update(self._team_block(home, "home", True, season))
        feats.update(self._team_block(away, "away", False, season))
        meetings = list(self._h2h.get(frozenset((home, away)), ()))[-self.h2h_window:]
        pts, gd = [], []
        for h, hg, ag in meetings:
            gf, ga = (hg, ag) if h == home else (ag, hg)
            pts.append(_points(gf, ga))
            gd.append(gf - ga)
        feats["h2h_home_ppg"] = self._mean(pts)
        feats["h2h_goal_diff"] = self._mean(gd)
        return feats

    def _update(self, row: pd.Series) -> None:
        """Incorpora un partido finalizado al estado."""
        home, away = row["home_team"], row["away_team"]
        hg, ag = float(row["home_goals"]), float(row["away_goals"])
        hxg = float(row["home_xg"]) if pd.notna(row["home_xg"]) else hg
        axg = float(row["away_xg"]) if pd.notna(row["away_xg"]) else ag
        season = int(row["season"]) if pd.notna(row["season"]) else 0
        elo_h, elo_a = self.elo.rating(home), self.elo.rating(away)
        self._history.setdefault(home, deque(maxlen=60)).append(
            _TeamEntry(season, True, hg, ag, hxg, axg, _points(hg, ag), elo_a))
        self._history.setdefault(away, deque(maxlen=60)).append(
            _TeamEntry(season, False, ag, hg, axg, hxg, _points(ag, hg), elo_h))
        self._h2h.setdefault(frozenset((home, away)), deque(maxlen=20)).append((home, hg, ag))
        self.elo.update(home, away, int(hg), int(ag))

    def fit_transform(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Recorre el histórico y devuelve features pre-partido + objetivos.

        Args:
            matches: Partidos (se usan los finalizados).

        Returns:
            DataFrame con ``match_id``, ``date``, ``FEATURE_COLUMNS``, número de
            partidos previos de cada equipo y los objetivos ``target_result``
            (0=local, 1=empate, 2=visitante), ``target_over25`` y ``target_btts``.
        """
        self._reset()
        rows = []
        for _, row in _finished(matches).iterrows():
            season = int(row["season"]) if pd.notna(row["season"]) else None
            if season is not None and self._season is not None and season > self._season:
                self.elo.regress_to_mean()
            self._season = season if season is not None else self._season
            feats = self._features(row["home_team"], row["away_team"], season)
            hg, ag = row["home_goals"], row["away_goals"]
            feats.update({
                "match_id": row["match_id"], "date": row["date"],
                "target_result": 0 if hg > ag else 1 if hg == ag else 2,
                "target_over25": int(hg + ag > 2.5),
                "target_btts": int(hg > 0 and ag > 0),
            })
            rows.append(feats)
            self._update(row)
        return pd.DataFrame(rows)

    def features_for(self, home: str, away: str, season: int | None = None, neutral: bool = False) -> dict[str, float]:
        """Features de un partido futuro con el estado actual.

        Args:
            home: Local.
            away: Visitante.
            season: Temporada del partido (por defecto la última vista).
            neutral: Campo neutral.

        Returns:
            Diccionario de features.
        """
        return self._features(home, away, self._season if season is None else season, neutral)


# =============================================================================
# Poisson / Dixon-Coles
# =============================================================================
@dataclass
class PoissonModelConfig:
    """Configuración del modelo de Poisson.

    Attributes:
        half_life_days: Vida media del peso temporal (partidos viejos pesan menos).
        xg_weight: Peso del xG frente a los goles reales en el objetivo (0-1).
        prior_sd: Desviación del prior gaussiano de fuerzas (regularización).
        max_goals: Goles máximos por equipo en la matriz.
        dixon_coles: Si se aplica la corrección de marcadores bajos.
        min_matches: Partidos mínimos para entrenar.
    """

    half_life_days: float = 240.0
    xg_weight: float = 0.4
    prior_sd: float = 0.6
    max_goals: int = 10
    dixon_coles: bool = True
    min_matches: int = 30


class PoissonModel:
    """Modelo de Dixon-Coles con decaimiento temporal y mezcla de xG.

    ``log λ_local = μ + h + ataque_local + defensa_visitante``
    ``log λ_visit = μ + ataque_visitante + defensa_local``

    Se ajusta por máxima verosimilitud ponderada (L-BFGS con gradiente
    analítico) y luego se estima ``ρ`` de Dixon-Coles por separado.
    """

    def __init__(self, config: PoissonModelConfig | None = None) -> None:
        """Crea el modelo.

        Args:
            config: Configuración.
        """
        self.config = config or PoissonModelConfig()
        self.teams: list[str] = []
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.mu: float = 0.0
        self.home_adv: float = 0.0
        self.rho: float = 0.0
        self._fitted = False
        self._warned: set[str] = set()

    @property
    def is_fitted(self) -> bool:
        """Indica si el modelo está entrenado."""
        return self._fitted

    def fit(self, matches: pd.DataFrame, reference_date: pd.Timestamp | None = None) -> PoissonModel:
        """Entrena el modelo.

        Args:
            matches: Partidos (se usan los finalizados).
            reference_date: Fecha respecto a la cual se calcula el decaimiento
                (por defecto, el último partido).

        Returns:
            ``self``.

        Raises:
            InsufficientDataError: Si hay menos partidos que ``min_matches``.
            RuntimeError: Si la optimización no produce parámetros finitos.
        """
        m = _finished(matches)
        if len(m) < self.config.min_matches:
            raise InsufficientDataError(f"Se requieren ≥{self.config.min_matches} partidos, hay {len(m)}")
        self.teams = sorted(set(m["home_team"]) | set(m["away_team"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        hi = m["home_team"].map(idx).to_numpy()
        ai = m["away_team"].map(idx).to_numpy()
        goals_h = m["home_goals"].to_numpy(float)
        goals_a = m["away_goals"].to_numpy(float)
        w_xg = self.config.xg_weight
        yh = np.where(m["home_xg"].notna(), (1 - w_xg) * goals_h + w_xg * m["home_xg"].fillna(0).to_numpy(), goals_h)
        ya = np.where(m["away_xg"].notna(), (1 - w_xg) * goals_a + w_xg * m["away_xg"].fillna(0).to_numpy(), goals_a)
        ref = reference_date if reference_date is not None else m["date"].max()
        age = np.clip((ref - m["date"]).dt.total_seconds().to_numpy() / 86400.0, 0, None)
        w = 0.5 ** (age / self.config.half_life_days)
        w = w / w.mean()
        ridge = 1.0 / (2 * self.config.prior_sd ** 2)

        def nll(theta: np.ndarray) -> tuple[float, np.ndarray]:
            mu, home = theta[0], theta[1]
            att, dfn = theta[2:2 + n], theta[2 + n:]
            log_lh = mu + home + att[hi] + dfn[ai]
            log_la = mu + att[ai] + dfn[hi]
            lh, la = np.exp(log_lh), np.exp(log_la)
            ll = np.sum(w * (yh * log_lh - lh + ya * log_la - la))
            value = -ll + ridge * (att @ att + dfn @ dfn)
            rh, ra = w * (yh - lh), w * (ya - la)
            grad = np.empty_like(theta)
            grad[0] = -(rh.sum() + ra.sum())
            grad[1] = -rh.sum()
            grad[2:2 + n] = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * ridge * att
            grad[2 + n:] = -(np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2 * ridge * dfn
            return float(value), grad

        theta0 = np.zeros(2 + 2 * n)
        theta0[0] = np.log(max(np.mean(np.r_[yh, ya]), 0.1))
        theta0[1] = 0.2
        res = minimize(nll, theta0, jac=True, method="L-BFGS-B")
        if not np.all(np.isfinite(res.x)):
            raise RuntimeError(f"El ajuste de Poisson no convergió: {res.message}")
        if not res.success:
            logger.warning("Optimización de Poisson con aviso: %s", res.message)
        self.mu, self.home_adv = float(res.x[0]), float(res.x[1])
        self.attack = {t: float(res.x[2 + i]) for t, i in idx.items()}
        self.defense = {t: float(res.x[2 + n + i]) for t, i in idx.items()}

        self.rho = 0.0
        if self.config.dixon_coles:
            lh = np.exp(self.mu + self.home_adv + res.x[2 + hi] + res.x[2 + n + ai])
            la = np.exp(self.mu + res.x[2 + ai] + res.x[2 + n + hi])
            x, y = goals_h.astype(int), goals_a.astype(int)

            def neg_tau_ll(rho: float) -> float:
                tau = np.ones_like(lh)
                tau = np.where((x == 0) & (y == 0), 1 - lh * la * rho, tau)
                tau = np.where((x == 0) & (y == 1), 1 + lh * rho, tau)
                tau = np.where((x == 1) & (y == 0), 1 + la * rho, tau)
                tau = np.where((x == 1) & (y == 1), 1 - rho, tau)
                return float(-np.sum(w * np.log(np.clip(tau, 1e-10, None))))

            self.rho = float(minimize_scalar(neg_tau_ll, bounds=(-0.2, 0.2), method="bounded").x)
        self._fitted = True
        self._warned.clear()
        return self

    def _strength(self, team: str) -> tuple[float, float]:
        """Ataque y defensa de un equipo (0 = promedio si es desconocido)."""
        if team not in self.attack and team not in self._warned:
            logger.warning("Equipo '%s' sin historial: se asume fuerza promedio", team)
            self._warned.add(team)
        return self.attack.get(team, 0.0), self.defense.get(team, 0.0)

    def expected_goals(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        """Goles esperados de cada equipo.

        Args:
            home: Local.
            away: Visitante.
            neutral: Campo neutral (sin ventaja de local).

        Returns:
            ``(λ_local, λ_visitante)``.

        Raises:
            ModelNotFittedError: Si el modelo no está entrenado.
        """
        if not self._fitted:
            raise ModelNotFittedError("PoissonModel no está entrenado")
        att_h, def_h = self._strength(home)
        att_a, def_a = self._strength(away)
        lh = np.exp(self.mu + (0.0 if neutral else self.home_adv) + att_h + def_a)
        la = np.exp(self.mu + att_a + def_h)
        return float(lh), float(la)

    def score_matrix(self, lambda_home: float, lambda_away: float) -> FloatMatrix:
        """Matriz de probabilidades de marcador con corrección Dixon-Coles.

        Args:
            lambda_home: Goles esperados del local.
            lambda_away: Goles esperados del visitante.

        Returns:
            Matriz normalizada ``(max_goals+1)²``.
        """
        g = np.arange(self.config.max_goals + 1)
        mat = np.outer(poisson.pmf(g, lambda_home), poisson.pmf(g, lambda_away))
        if self.config.dixon_coles and self.rho != 0.0:
            r = self.rho
            mat[0, 0] *= 1 - lambda_home * lambda_away * r
            mat[0, 1] *= 1 + lambda_home * r
            mat[1, 0] *= 1 + lambda_away * r
            mat[1, 1] *= 1 - r
            mat = np.clip(mat, 0.0, None)
        return mat / mat.sum()

    def predict(self, home: str, away: str, match_id: str = "", neutral: bool = False,
                league: str = "", kickoff: pd.Timestamp | None = None) -> MatchPrediction:
        """Predicción completa de un partido.

        Args:
            home: Local.
            away: Visitante.
            match_id: ID del partido.
            neutral: Campo neutral.
            league: Liga.
            kickoff: Fecha/hora.

        Returns:
            :class:`MatchPrediction` con la matriz de marcadores.
        """
        lh, la = self.expected_goals(home, away, neutral)
        return MatchPrediction(match_id or f"{home}-{away}", home, away, lh, la,
                               self.score_matrix(lh, la), league=league, kickoff=kickoff)

    def team_ratings(self) -> pd.DataFrame:
        """Ratings de ataque/defensa (exp: multiplicador sobre la media).

        Returns:
            DataFrame ordenado por fuerza neta.
        """
        if not self._fitted:
            raise ModelNotFittedError("PoissonModel no está entrenado")
        df = pd.DataFrame({"team": self.teams,
                           "attack": [np.exp(self.attack[t]) for t in self.teams],
                           "defense": [np.exp(self.defense[t]) for t in self.teams]})
        df["net"] = np.log(df["attack"]) - np.log(df["defense"])
        return df.sort_values("net", ascending=False).reset_index(drop=True)


# =============================================================================
# Machine Learning
# =============================================================================
ML_SELECTIONS: dict[str, list[Selection]] = {
    "result": [Selection.home(), Selection.draw(), Selection.away()],
    "over25": [Selection.under(2.5), Selection.over(2.5)],
    "btts": [Selection.btts(False), Selection.btts(True)],
}


class MLOutcomeModel:
    """Clasificadores (1X2, Over/Under 2.5, BTTS) sobre las features pre-partido."""

    def __init__(self, estimator: Literal["logistic", "random_forest"] = "logistic", C: float = 0.3,
                 min_samples: int = 150, min_prior_games: int = 3, random_state: int = 0,
                 features: Sequence[str] = tuple(FEATURE_COLUMNS)) -> None:
        """Crea el modelo.

        Args:
            estimator: ``logistic`` (calibrado, recomendado) o ``random_forest``.
            C: Inversa de la regularización (logística).
            min_samples: Muestras mínimas de entrenamiento.
            min_prior_games: Partidos previos mínimos por equipo para usar una fila.
            random_state: Semilla.
            features: Columnas de entrada.
        """
        self.estimator = estimator
        self.C = C
        self.min_samples = min_samples
        self.min_prior_games = min_prior_games
        self.random_state = random_state
        self.features = list(features)
        self.models: dict[str, Pipeline] = {}

    @property
    def is_fitted(self) -> bool:
        """Indica si el modelo está entrenado."""
        return bool(self.models)

    def _pipeline(self) -> Pipeline:
        """Construye el pipeline de preprocesado + estimador."""
        if self.estimator == "random_forest":
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_depth=6,
                                         random_state=self.random_state, n_jobs=-1)
        else:
            clf = LogisticRegression(C=self.C, max_iter=2000)
        return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler()), ("clf", clf)])

    def fit(self, feature_df: pd.DataFrame) -> MLOutcomeModel:
        """Entrena los tres clasificadores.

        Args:
            feature_df: Salida de :meth:`FeatureEngineer.fit_transform`.

        Returns:
            ``self``.

        Raises:
            InsufficientDataError: Si no hay suficientes filas utilizables.
        """
        usable = feature_df[(feature_df["home_games"] >= self.min_prior_games)
                            & (feature_df["away_games"] >= self.min_prior_games)]
        if len(usable) < self.min_samples:
            raise InsufficientDataError(f"ML requiere ≥{self.min_samples} filas, hay {len(usable)}")
        x = usable[self.features]
        self.models = {}
        for name, target in (("result", "target_result"), ("over25", "target_over25"), ("btts", "target_btts")):
            y = usable[target]
            if y.nunique() < 2:
                raise InsufficientDataError(f"El objetivo {target} tiene una sola clase")
            self.models[name] = self._pipeline().fit(x, y)
        return self

    def predict_proba(self, features: Mapping[str, float] | pd.DataFrame) -> dict[Selection, float]:
        """Probabilidades por selección para un partido.

        Args:
            features: Features de un partido (dict o DataFrame de una fila).

        Returns:
            Probabilidades de 1X2, Over/Under 2.5 y BTTS.

        Raises:
            ModelNotFittedError: Si el modelo no está entrenado.
        """
        if not self.is_fitted:
            raise ModelNotFittedError("MLOutcomeModel no está entrenado")
        x = features if isinstance(features, pd.DataFrame) else pd.DataFrame([dict(features)])
        x = x.reindex(columns=self.features)
        out: dict[Selection, float] = {}
        for name, sels in ML_SELECTIONS.items():
            model = self.models[name]
            proba = model.predict_proba(x)[0]
            classes = list(model.classes_)
            for cls_value, sel in enumerate(sels):
                out[sel] = float(proba[classes.index(cls_value)]) if cls_value in classes else 0.0
        return out


# =============================================================================
# Ensamble
# =============================================================================
class EnsemblePredictor:
    """Combina Poisson (estructura de marcadores) con ML (features de contexto).

    La matriz de Poisson se mantiene para derivar todos los mercados y las
    correlaciones; las marginales de 1X2, OU 2.5 y BTTS se sustituyen por la
    mezcla lineal ``(1-w)·Poisson + w·ML``.
    """

    def __init__(self, poisson_model: PoissonModel, ml_model: MLOutcomeModel | None = None,
                 feature_engineer: FeatureEngineer | None = None, ml_weight: float = 0.35) -> None:
        """Crea el ensamble.

        Args:
            poisson_model: Modelo de Poisson entrenado.
            ml_model: Modelo ML entrenado (opcional).
            feature_engineer: Generador de features con el estado actualizado.
            ml_weight: Peso del ML en la mezcla (0-1).

        Raises:
            ValueError: Si ``ml_weight`` está fuera de [0, 1].
        """
        if not 0.0 <= ml_weight <= 1.0:
            raise ValueError("ml_weight debe estar en [0, 1]")
        self.poisson = poisson_model
        self.ml = ml_model if ml_model is not None and ml_model.is_fitted else None
        self.fe = feature_engineer
        self.ml_weight = ml_weight

    def predict(self, home: str, away: str, match_id: str = "", league: str = "",
                kickoff: pd.Timestamp | None = None, season: int | None = None,
                neutral: bool = False, features: Mapping[str, float] | None = None) -> MatchPrediction:
        """Predicción de un partido combinando modelos.

        Args:
            home: Local.
            away: Visitante.
            match_id: ID del partido.
            league: Liga.
            kickoff: Fecha/hora.
            season: Temporada (para las features).
            neutral: Campo neutral.
            features: Features precalculadas (si no, se piden al ``feature_engineer``).

        Returns:
            Predicción con ``sources`` = {"poisson", "ml"} y ``overrides`` del ensamble.
        """
        pred = self.poisson.predict(home, away, match_id, neutral, league, kickoff)
        poisson_probs = {s: pred.matrix_probability(s) for sels in ML_SELECTIONS.values() for s in sels}
        pred.sources["poisson"] = poisson_probs
        if self.ml is None or self.ml_weight == 0.0:
            return pred
        if features is None:
            if self.fe is None:
                return pred
            features = self.fe.features_for(home, away, season, neutral)
        try:
            ml_probs = self.ml.predict_proba(features)
        except (ValueError, KeyError) as exc:
            logger.warning("ML falló para %s vs %s (%s); se usa solo Poisson", home, away, exc)
            return pred
        pred.sources["ml"] = ml_probs
        w = self.ml_weight
        for sels in ML_SELECTIONS.values():
            blended = np.array([(1 - w) * poisson_probs[s] + w * ml_probs[s] for s in sels])
            blended /= blended.sum()
            pred.overrides.update({s: float(p) for s, p in zip(sels, blended)})
        return pred


# =============================================================================
# Evaluación
# =============================================================================
def evaluate_predictions(probs: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    """Log-loss y Brier multiclase.

    Args:
        probs: Matriz ``(n, k)`` de probabilidades.
        outcomes: Vector ``(n,)`` con el índice de la clase real.

    Returns:
        ``{"log_loss": ..., "brier": ..., "accuracy": ..., "n": ...}``.

    Raises:
        ValueError: Si las dimensiones no coinciden o no hay datos.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    if probs.ndim != 2 or len(probs) != len(outcomes) or len(outcomes) == 0:
        raise ValueError("Dimensiones inválidas para evaluar")
    p_true = np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-12, 1.0)
    onehot = np.eye(probs.shape[1])[outcomes]
    return {
        "log_loss": float(-np.mean(np.log(p_true))),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "accuracy": float(np.mean(probs.argmax(axis=1) == outcomes)),
        "n": float(len(outcomes)),
    }


__all__ = [
    "EloRatingSystem", "EnsemblePredictor", "FEATURE_COLUMNS", "FeatureEngineer", "FormSummary", "H2HSummary",
    "HeuristicAnalyzer", "InsufficientDataError", "MLOutcomeModel", "MatchReport", "ModelNotFittedError",
    "PoissonModel", "PoissonModelConfig", "TeamSnapshot", "evaluate_predictions", "Market",
]
