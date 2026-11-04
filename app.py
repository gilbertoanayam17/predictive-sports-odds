"""Interfaz web (Streamlit) sobre el pipeline de análisis.

Ejecutar::

    pip install -r requirements-ui.txt
    streamlit run app.py

La app no contiene lógica de negocio: solo llama a la API de
``predictive_odds`` (pipeline, backtester, registro) y muestra sus DataFrames.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pandas as pd
import streamlit as st

from predictive_odds.backtester import BacktestConfig, Backtester, BacktestResult
from predictive_odds.bet_tracker import BetTracker, TrackerError
from predictive_odds.config import LEAGUES, ConfigurationError, get_league
from predictive_odds.data_fetcher import (
    AuthenticationError,
    DataFetcher,
    DataFetchError,
    FootballDataCoUkProvider,
    SimulatedDataProvider,
    SimulationConfig,
    TheOddsApiProvider,
)
from predictive_odds.parlay_builder import ParlayConfig, ValueBetConfig
from predictive_odds.pipeline import AnalysisPipeline, AnalysisResult, PipelineConfig
from predictive_odds.statistical_models import InsufficientDataError

DISCLAIMER = "Herramienta analítica: no coloca apuestas. Apostar implica riesgo de pérdida; juega con responsabilidad."


def build_fetcher(source: str, seed: int, seasons: tuple[int, int], odds_key: str) -> DataFetcher:
    """Crea la fachada de datos según la fuente elegida.

    Args:
        source: ``"Simulada"`` o ``"Real"``.
        seed: Semilla de la simulación.
        seasons: Temporadas (anterior, actual).
        odds_key: API key de The Odds API (solo en modo real).

    Returns:
        ``DataFetcher`` configurado.

    Raises:
        AuthenticationError: Si falta la API key en modo real.
    """
    if source == "Simulada":
        sim = SimulatedDataProvider(SimulationConfig(seed=seed, seasons=seasons))
        return DataFetcher(sim, odds_provider=sim)
    return DataFetcher(FootballDataCoUkProvider(), odds_provider=TheOddsApiProvider(odds_key))


@st.cache_data(show_spinner="Entrenando modelos y buscando valor…", ttl=900)
def run_analysis(source: str, seed: int, league_key: str, season: int, bankroll: float, use_ml: bool,
                 min_edge: float, value_weight: float, max_legs: int, min_leg_prob: float,
                 same_match: bool, exposure: float, odds_key: str) -> AnalysisResult:
    """Ejecuta el pipeline (memoizado por parámetros).

    Returns:
        Resultado del análisis.
    """
    seasons = (season - 1, season)
    league = get_league(league_key)
    config = PipelineConfig(
        league_key=league_key, seasons=seasons, bankroll=bankroll, use_ml=use_ml, max_total_exposure=exposure,
        value=replace(ValueBetConfig(), min_edge=min_edge, value_model_weight=value_weight),
        parlay=replace(ParlayConfig(), max_legs=max_legs, min_leg_prob=min_leg_prob, allow_same_match=same_match),
        extra_leagues=("EPL",) if league.is_international and source == "Simulada" else (),
    )
    return AnalysisPipeline(build_fetcher(source, seed, seasons, odds_key), config).run()


@st.cache_data(show_spinner="Ejecutando backtest walk-forward (puede tardar ~15 s)…", ttl=3600)
def run_backtest(seed: int, league_key: str, season: int, bankroll: float, use_ml: bool,
                 value_weight: float, parlays: bool) -> BacktestResult:
    """Backtest sobre datos simulados (memoizado).

    Returns:
        Resultado del backtest.
    """
    seasons = (season - 1, season)
    league = get_league(league_key)
    sim = SimulatedDataProvider(SimulationConfig(seed=seed, seasons=seasons))
    history = DataFetcher(sim).get_history(league_key, seasons)
    odds = pd.concat([sim.fetch_historical_odds(league, s) for s in seasons], ignore_index=True)
    config = BacktestConfig(initial_bankroll=bankroll, use_ml=use_ml, include_parlays=parlays,
                            value=replace(ValueBetConfig(), value_model_weight=value_weight))
    return Backtester(history, odds, config).run()


def pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Formatea columnas de probabilidad como porcentaje para mostrar."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (100 * out[c].astype(float)).round(1)
    return out


def main() -> None:
    """Construye la interfaz."""
    st.set_page_config(page_title="Predictive Sports Odds", page_icon="⚽", layout="wide")
    st.title("⚽ Predictive Sports Odds")
    st.caption(DISCLAIMER)

    with st.sidebar:
        st.header("Datos")
        source = st.radio("Fuente", ["Simulada", "Real"], horizontal=True,
                          help="Real: histórico de football-data.co.uk + momios de The Odds API")
        league_key = st.selectbox("Liga", list(LEAGUES), format_func=lambda k: LEAGUES[k].name)
        season = int(st.number_input("Temporada actual", 2015, 2030, 2025))
        seed = int(st.number_input("Semilla (simulación)", 0, 10_000, 7)) if source == "Simulada" else 0
        odds_key = ""
        if source == "Real":
            odds_key = st.text_input("ODDS_API_KEY", value=os.environ.get("ODDS_API_KEY", ""), type="password")
        st.header("Modelo y apuestas")
        bankroll = float(st.number_input("Bankroll", 10.0, 1e7, 1000.0, step=100.0))
        use_ml = st.toggle("Ensamble con ML", value=True)
        min_edge = st.slider("EV mínimo", 0.0, 0.15, 0.04, 0.01)
        value_weight = st.slider("Peso del modelo frente al mercado", 0.0, 1.0, 0.5, 0.05,
                                 help="0.5 corrige la 'maldición del ganador' (ver backtest)")
        max_legs = st.slider("Patas máximas por parlay", 2, 4, 3)
        min_leg_prob = st.slider("Probabilidad mínima por pata", 0.4, 0.8, 0.55, 0.01)
        same_match = st.toggle("Permitir patas del mismo partido", value=False)
        exposure = st.slider("Exposición máxima del bankroll", 0.02, 0.3, 0.10, 0.01)
        db_path = st.text_input("Registro de apuestas (SQLite)", "bets.sqlite")

    try:
        result = run_analysis(source, seed, league_key, season, bankroll, use_ml, min_edge, value_weight,
                              max_legs, min_leg_prob, same_match, exposure, odds_key)
    except (AuthenticationError, ConfigurationError) as exc:
        st.error(f"Configuración: {exc}")
        st.stop()
    except (DataFetchError, InsufficientDataError) as exc:
        st.error(f"No se pudieron obtener datos suficientes: {exc}")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Partidos analizados", len(result.predictions))
    c2.metric("Value bets", len(result.value_bets))
    c3.metric("Parlays", len(result.parlays))
    c4.metric("Exposición sugerida", f"{result.total_stake:,.2f}", f"{100 * result.total_stake / bankroll:.1f}% del bankroll")

    tabs = st.tabs(["Predicciones", "Value bets", "Parlays", "Tabla y ratings", "Backtest", "Registro"])

    with tabs[0]:
        st.dataframe(pct(result.predictions_frame().drop(columns="match_id"), ["P(1)", "P(X)", "P(2)", "P(O2.5)", "P(BTTS)"]),
                     hide_index=True, use_container_width=True)
        labels = {mid: f"{p.home_team} vs {p.away_team}" for mid, p in result.predictions.items()}
        mid = st.selectbox("Detalle del partido", list(labels), format_func=labels.__getitem__)
        pred, rep = result.predictions[mid], result.reports[mid]
        left, right = st.columns(2)
        with left:
            st.subheader("Análisis heurístico")
            st.write(f"**{pred.home_team}** — pos {rep.home.position}, forma `{rep.home.form_overall.results}`, "
                     f"en casa `{rep.home.form_venue.results}`")
            st.write(f"**{pred.away_team}** — pos {rep.away.position}, forma `{rep.away.form_overall.results}`, "
                     f"fuera `{rep.away.form_venue.results}`")
            st.write(f"H2H: {rep.h2h.a_wins}G-{rep.h2h.draws}E-{rep.h2h.b_wins}P · índice {rep.score:+.3f} → **{rep.lean.value}**")
            for note in rep.notes:
                st.write(f"• {note}")
            for team in (pred.home_team, pred.away_team):
                if team in result.adjustments:
                    a = result.adjustments[team]
                    st.warning(f"{team}: ataque ×{a.attack:.2f}, goles recibidos ×{a.defense:.2f} ({a.reason})")
        with right:
            st.subheader("Matriz de marcadores (%)")
            n = 6
            mat = pd.DataFrame(100 * pred.score_matrix[:n, :n], index=[f"{i}" for i in range(n)],
                               columns=[f"{j}" for j in range(n)]).round(1)
            mat.index.name, mat.columns.name = pred.home_team, pred.away_team
            st.dataframe(mat, use_container_width=True)

    with tabs[1]:
        vb = result.value_bets_frame()
        if vb.empty:
            st.info("Ninguna selección supera el umbral de valor.")
        else:
            st.dataframe(pct(vb, ["p_modelo", "p_implícita", "p_mercado", "EV_modelo", "EV_ajustado"]),
                         hide_index=True, use_container_width=True)
            st.caption("Probabilidades y EV en %. El stake ya incluye Kelly modificado y el tope de exposición.")

    with tabs[2]:
        pf = result.parlays_frame()
        if pf.empty:
            st.info("No hay combinaciones que cumplan los criterios de riesgo y valor.")
        else:
            st.dataframe(pct(pf, ["p_conjunta", "p_implícita", "EV"]), hide_index=True, use_container_width=True)
            for i, par in enumerate(result.parlays, 1):
                for w in par.warnings:
                    st.warning(f"Parlay {i}: {w}")

    with tabs[3]:
        left, right = st.columns(2)
        left.subheader("Tabla")
        left.dataframe(result.standings, hide_index=True, use_container_width=True)
        right.subheader("Ratings Dixon-Coles")
        right.dataframe(result.team_ratings.round(3), hide_index=True, use_container_width=True)
        if result.evaluation:
            st.subheader("Validación fuera de muestra (1X2)")
            st.dataframe(result.evaluation_frame().round(4), use_container_width=True)

    with tabs[4]:
        if source != "Simulada":
            st.info("El backtest de la app usa datos simulados; con datos reales usa `python main.py --backtest --live`.")
        parlays = st.toggle("Incluir parlays en el backtest", value=False)
        if st.button("Ejecutar backtest", type="primary"):
            try:
                bt = run_backtest(seed, league_key, season, bankroll, use_ml, value_weight, parlays)
            except (InsufficientDataError, DataFetchError) as exc:
                st.error(str(exc))
            else:
                m = bt.metrics
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("ROI", f"{100 * m.get('roi', 0):.2f}%")
                k2.metric("CLV medio", f"{100 * m.get('clv_medio', 0):.2f}%")
                k3.metric("Crecimiento", f"{100 * m['crecimiento']:.1f}%")
                k4.metric("Drawdown máx.", f"{100 * m.get('max_drawdown', 0):.1f}%")
                st.line_chart(bt.bankroll.set_index("date")["bankroll"])
                st.dataframe(bt.by_market().round(3), use_container_width=True)
                st.subheader("Calibración")
                st.dataframe(bt.calibration().round(3), hide_index=True, use_container_width=True)
                st.caption("Un CLV positivo sostenido es la mejor señal de edge real; el ROI a corto plazo es ruidoso.")

    with tabs[5]:
        try:
            with BetTracker(db_path) as tracker:
                if st.button("Registrar las apuestas sugeridas"):
                    new = tracker.record_analysis(result.value_bets, result.parlays, result.bankroll)
                    st.success(f"{new} apuesta(s) nueva(s) registrada(s).")
                summary = tracker.summary(initial_bankroll=bankroll)
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Apuestas", int(summary["apuestas"]))
                s2.metric("Pendientes", int(summary["pendientes"]))
                s3.metric("Beneficio", f"{summary['beneficio']:,.2f}")
                s4.metric("ROI", f"{100 * summary['roi']:.2f}%")
                bets = tracker.bets()
                st.dataframe(bets.drop(columns=["bet_key", "legs"]), hide_index=True, use_container_width=True)
                history = tracker.bankroll_history(bankroll)
                if not history.empty:
                    st.line_chart(history.reset_index(drop=True)["bankroll"])
        except TrackerError as exc:
            st.error(f"Registro de apuestas: {exc}")


if __name__ == "__main__":
    main()
