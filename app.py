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
from predictive_odds.config import LEAGUES, ConfigurationError, current_season, get_league, selectable_leagues
from predictive_odds.connectivity import API_FOOTBALL_SIGNUP, ODDS_API_SIGNUP, check_all
from predictive_odds.data_fetcher import (
    ApiFootballProvider,
    AuthenticationError,
    DataFetcher,
    DataFetchError,
    FootballDataCoUkProvider,
    ResultsProvider,
    SimulatedDataProvider,
    SimulationConfig,
    TheOddsApiProvider,
)
from predictive_odds.parlay_builder import ParlayConfig, ValueBetConfig
from predictive_odds.pipeline import AnalysisPipeline, AnalysisResult, PipelineConfig, SlateResult, analyze_slate
from predictive_odds.report import render_html, render_markdown
from predictive_odds.risk import RiskConfig, risk_table
from predictive_odds.settings import load_env
from predictive_odds.statistical_models import InsufficientDataError

DISCLAIMER = "Herramienta analítica: no coloca apuestas. Apostar implica riesgo de pérdida; juega con responsabilidad."


def build_fetcher(source: str, seed: int, seasons: tuple[int, int], odds_key: str,
                  af_key: str = "", league_key: str = "EPL") -> DataFetcher:
    """Crea la fachada de datos según la fuente elegida.

    En modo real: histórico de football-data.co.uk (si la liga está ahí; si no,
    de API-Football), momios de The Odds API y, con clave de API-Football,
    calendario y lesiones. Sin API-Football el calendario se deduce de los momios.

    Args:
        source: ``"Simulada"`` o ``"Real"``.
        seed: Semilla de la simulación.
        seasons: Temporadas (anterior, actual).
        odds_key: API key de The Odds API (solo en modo real).
        af_key: API key de API-Football (opcional).
        league_key: Liga principal (decide la fuente del histórico).

    Returns:
        ``DataFetcher`` configurado.

    Raises:
        AuthenticationError: Si falta una API key necesaria.
    """
    if source == "Simulada":
        sim = SimulatedDataProvider(SimulationConfig(seed=seed, seasons=seasons))
        return DataFetcher(sim, odds_provider=sim)
    api_football = ApiFootballProvider(af_key) if af_key.strip() else None
    if get_league(league_key).football_data_code:
        history: ResultsProvider = FootballDataCoUkProvider()
    elif api_football is not None:
        history = api_football
    else:
        raise AuthenticationError(f"{get_league(league_key).name} requiere una clave de API-Football")
    return DataFetcher(history, fixtures_provider=api_football or history,
                       odds_provider=TheOddsApiProvider(odds_key.strip()))


@st.cache_data(show_spinner="Entrenando modelos y buscando valor…", ttl=900)
def run_analysis(source: str, seed: int, league_key: str, season: int, bankroll: float, use_ml: bool,
                 min_edge: float, value_weight: float, max_legs: int, min_leg_prob: float,
                 same_match: bool, exposure: float, odds_key: str, af_key: str = "") -> AnalysisResult:
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
    return AnalysisPipeline(build_fetcher(source, seed, seasons, odds_key, af_key, league_key), config).run()


@st.cache_data(show_spinner="Ejecutando backtest walk-forward (puede tardar ~15 s)…", ttl=3600)
def run_backtest(seed: int, league_key: str, season: int, bankroll: float, use_ml: bool,
                 value_weight: float, parlays: bool, real: bool = False) -> BacktestResult:
    """Backtest memoizado sobre datos simulados o REALES.

    Con ``real=True`` usa football-data.co.uk (resultados, momios de varias casas
    y cierre de Pinnacle), que no requiere ninguna clave.

    Returns:
        Resultado del backtest.

    Raises:
        ProviderNotSupportedError: Si la liga no tiene histórico en football-data.
    """
    seasons = (season - 1, season)
    league = get_league(league_key)
    provider: SimulatedDataProvider | FootballDataCoUkProvider
    if real:
        provider = FootballDataCoUkProvider()
    else:
        provider = SimulatedDataProvider(SimulationConfig(seed=seed, seasons=seasons))
    history = DataFetcher(provider).get_history(league_key, seasons)
    odds = pd.concat([provider.fetch_historical_odds(league, s) for s in seasons], ignore_index=True)
    config = BacktestConfig(initial_bankroll=bankroll, use_ml=use_ml, include_parlays=parlays,
                            value=replace(ValueBetConfig(), value_model_weight=value_weight))
    return Backtester(history, odds, config).run()


@st.cache_data(show_spinner="Analizando la jornada multiliga…", ttl=900)
def run_slate(source: str, seed: int, leagues: tuple[str, ...], season: int, bankroll: float, use_ml: bool,
              max_singles: int, odds_key: str, af_key: str = "") -> SlateResult:
    """Jornada multiliga memoizada (parlays entre ligas y tope global).

    Returns:
        Resultado multiliga.
    """
    seasons = (season - 1, season)
    config = PipelineConfig(seasons=seasons, bankroll=bankroll, use_ml=use_ml, max_singles=max_singles)
    return analyze_slate(build_fetcher(source, seed, seasons, odds_key, af_key, leagues[0]), list(leagues), config)


def settle_pending(tracker: BetTracker, source: str, seed: int, league_key: str, season: int,
                   odds_key: str, af_key: str) -> int:
    """Liquida las apuestas pendientes con los resultados de la temporada.

    Returns:
        Número de apuestas liquidadas.
    """
    fetcher = build_fetcher(source, seed, (season - 1, season), odds_key or "sin-clave", af_key, league_key)
    league = get_league(league_key)
    results = fetcher.history_provider.fetch_matches(league, season)
    if fetcher.fixtures_provider is not fetcher.history_provider:  # API-Football: resultados más recientes
        try:
            results = pd.concat([results, fetcher.fixtures_provider.fetch_matches(league, season)], ignore_index=True)
        except DataFetchError:
            pass
    return tracker.settle(results)


def show_backtest(bt: BacktestResult) -> None:
    """Muestra las métricas, la curva de bankroll y la calibración de un backtest."""
    m = bt.metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ROI", f"{100 * m.get('roi', 0):.2f}%")
    k2.metric("CLV medio", f"{100 * m.get('clv_medio', 0):.2f}%")
    k3.metric("Crecimiento", f"{100 * m['crecimiento']:.1f}%")
    k4.metric("Drawdown máx.", f"{100 * m.get('max_drawdown', 0):.1f}%")
    st.caption(f"{int(m['apuestas'])} apuestas · acierto {100 * m.get('acierto', 0):.1f}% · "
               f"momio medio {m.get('momio_medio', 0):.2f}")
    if not bt.bankroll.empty:
        st.line_chart(bt.bankroll.set_index("date")["bankroll"])
    st.dataframe(bt.by_market().round(3), use_container_width=True)
    st.subheader("Calibración")
    st.dataframe(bt.calibration().round(3), hide_index=True, use_container_width=True)
    st.caption("Un CLV positivo sostenido es la mejor señal de edge real; el ROI a corto plazo es ruidoso.")


def backtest_panel(source: str, seed: int, league_key: str, season: int, bankroll: float, use_ml: bool,
                   value_weight: float, key: str) -> None:
    """Controles + resultado del backtest (simulado o real, según la fuente)."""
    real = source == "Real"
    if real:
        st.caption("Backtest con datos REALES de football-data.co.uk (resultados, momios y cierre de Pinnacle): "
                   "**no necesita clave**. Disponible para todas las ligas de la lista.")
    parlays = st.toggle("Incluir parlays en el backtest", value=False, key=f"parlays_{key}")
    if st.button("Ejecutar backtest", type="primary", key=f"bt_{key}"):
        try:
            bt = run_backtest(seed, league_key, season, bankroll, use_ml, value_weight, parlays, real)
        except (InsufficientDataError, DataFetchError) as exc:
            st.error(str(exc))
        else:
            show_backtest(bt)


def pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Formatea columnas de probabilidad como porcentaje para mostrar."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (100 * out[c].astype(float)).round(1)
    return out


def main() -> None:
    """Construye la interfaz."""
    env_loaded = load_env()  # claves desde .env (si existe); las variables del sistema tienen prioridad
    st.set_page_config(page_title="Predictive Sports Odds", page_icon="⚽", layout="wide")
    st.title("⚽ Predictive Sports Odds")
    st.caption(DISCLAIMER)

    with st.sidebar:
        st.header("Datos")
        source = st.radio("Fuente", ["Simulada", "Real"], horizontal=True,
                          help="Real: histórico de football-data.co.uk + momios de The Odds API")
        league_key = st.selectbox("Liga", selectable_leagues(), format_func=lambda k: LEAGUES[k].name)
        season = int(st.number_input("Temporada actual", 2015, 2035, current_season(),
                                     help="Año de inicio de la temporada en curso (2026 = 2026/27). "
                                          "Se entrena con esta y la anterior."))
        seed = int(st.number_input("Semilla (simulación)", 0, 10_000, 7)) if source == "Simulada" else 0
        odds_key = af_key = ""
        if source == "Real":
            st.caption(f"[Conseguir clave de momios (gratis)]({ODDS_API_SIGNUP}) · "
                       f"[Clave de API-Football (opcional)]({API_FOOTBALL_SIGNUP})")
            if env_loaded:
                st.caption(f"✓ Cargado desde `.env`: {', '.join(env_loaded)}")
            elif not os.environ.get("ODDS_API_KEY"):
                st.caption("Consejo: guarda tus claves en un archivo `.env` (ver `.env.example`) "
                           "para no tener que pegarlas cada vez.")
            odds_key = st.text_input("Clave de The Odds API", value=os.environ.get("ODDS_API_KEY", ""),
                                     type="password", help="Obligatoria. Plan gratuito: 500 créditos/mes")
            af_key = st.text_input("Clave de API-Football (opcional)", value=os.environ.get("API_FOOTBALL_KEY", ""),
                                   type="password", help="Activa el ajuste por lesiones. Requiere plan de pago: "
                                                         "el gratuito no cubre la temporada actual")
            credits = TheOddsApiProvider.read_credits()
            if credits:
                st.caption(f"Créditos de momios restantes: **{credits.get('remaining', '?')}** "
                           f"(actualizado {str(credits.get('updated', ''))[:16].replace('T', ' ')} UTC). "
                           "Cada liga analizada cuesta ~2; sin partidos próximos, 0.")
            if st.button("Probar conexión", help="No consume créditos"):
                with st.spinner("Comprobando fuentes…"):
                    for chk in check_all(odds_key, af_key, league_key, season):
                        text = f"**{chk.service}**: {chk.message}"
                        if chk.ok:
                            st.success(text)
                        elif chk.required:
                            st.error(text)
                        else:
                            st.warning(text)
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
        st.header("Informe")
        report_slot = st.empty()

    if source == "Real" and not odds_key.strip():
        st.info(
            "### Cómo usar datos reales\n"
            f"1. **Crea una clave gratuita de The Odds API** en [the-odds-api.com]({ODDS_API_SIGNUP}): plan "
            "*Starter*, 500 créditos al mes. Te llega por correo.\n"
            "2. **Pégala** en la barra lateral (*Clave de The Odds API*) y pulsa **Probar conexión**: no gasta créditos.\n"
            f"3. *(Opcional, de pago)* Una clave de [API-Football]({API_FOOTBALL_SIGNUP}) activa el ajuste por "
            "lesiones. Su plan gratuito solo cubre temporadas antiguas, así que no sirve para la actual.\n\n"
            "El histórico de la Premier League, La Liga, Serie A, Bundesliga y Liga MX se descarga de "
            "football-data.co.uk, sin clave. "
            "Cada liga analizada gasta ~2 créditos de momios (0 si no tiene partidos en los próximos 4 días) y los resultados se guardan en caché 10 minutos.\n\n"
            "**Consejo:** antes de apostar, revisa la pestaña *Backtest* y registra las sugerencias en *Registro* "
            "durante unas semanas sin apostar, para medir el CLV real."
        )
        st.subheader("Mientras tanto: valida el modelo con datos reales, sin clave")
        backtest_panel(source, seed, league_key, season, bankroll, use_ml, value_weight, "welcome")
        st.stop()

    try:
        result = run_analysis(source, seed, league_key, season, bankroll, use_ml, min_edge, value_weight,
                              max_legs, min_leg_prob, same_match, exposure, odds_key, af_key)
    except (AuthenticationError, ConfigurationError) as exc:
        st.error(f"Configuración: {exc}")
        st.stop()
    except (DataFetchError, InsufficientDataError) as exc:
        st.error(f"No se pudieron obtener datos suficientes: {exc}")
        st.stop()

    if result.unknown_teams:
        st.warning(f"Sin historial: **{', '.join(result.unknown_teams)}**. El modelo les asigna fuerza promedio, "
                   "así que sus predicciones son poco fiables. Si acaban de ascender, es normal al inicio de la "
                   f"temporada; si no, revisa que *Temporada actual* sea {current_season()}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Partidos analizados", len(result.predictions))
    c2.metric("Value bets", len(result.value_bets))
    c3.metric("Parlays", len(result.parlays))
    c4.metric("Exposición sugerida", f"{result.total_stake:,.2f}", f"{100 * result.total_stake / bankroll:.1f}% del bankroll")

    simulated = source == "Simulada"
    with report_slot.container():
        st.download_button("Descargar informe HTML", render_html(result, simulated),
                           file_name=f"informe_{league_key}.html", mime="text/html")
        st.download_button("Descargar informe Markdown", render_markdown(result, simulated),
                           file_name=f"informe_{league_key}.md", mime="text/markdown")

    tabs = st.tabs(["Predicciones", "Value bets", "Parlays", "Tabla y ratings", "Backtest", "Registro",
                    "Riesgo", "Multiliga"])

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
        backtest_panel(source, seed, league_key, season, bankroll, use_ml, value_weight, "tab")

    with tabs[5]:
        try:
            with BetTracker(db_path) as tracker:
                st.caption(f"Archivo del registro: `{os.path.abspath(db_path)}`. Registrar no coloca apuestas: "
                           "sirve para hacer seguimiento (o *paper trading*). Volver a registrar no duplica.")
                b1, b2 = st.columns(2)
                if b1.button("Registrar las apuestas sugeridas"):
                    new = tracker.record_analysis(result.value_bets, result.parlays, result.bankroll)
                    st.success(f"{new} apuesta(s) nueva(s) registrada(s).")
                if b2.button("Liquidar con resultados", help="Busca los resultados de los partidos ya jugados"):
                    try:
                        with st.spinner("Descargando resultados…"):
                            settled = settle_pending(tracker, source, seed, league_key, season, odds_key, af_key)
                        if settled:
                            st.success(f"{settled} apuesta(s) liquidada(s).")
                        else:
                            st.info("No hay apuestas pendientes con resultado disponible todavía. football-data.co.uk "
                                    "suele publicar los resultados 1-2 días después del partido.")
                    except (DataFetchError, AuthenticationError) as exc:
                        st.error(f"No se pudieron obtener resultados: {exc}")
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

    with tabs[6]:
        st.subheader("Riesgo Monte Carlo del staking")
        st.caption("Repite la jornada actual N veces. recorte_edge: 0 = el modelo acierta; 0.5 = edge real a la "
                   "mitad; 1 = el modelo no aporta; 1.5 = el modelo se equivoca en contra.")
        rounds = st.slider("Jornadas a simular", 5, 76, 38)
        if not result.value_bets:
            st.info("Sin value bets en esta jornada: nada que simular.")
        elif st.button("Simular riesgo", type="primary"):
            table = risk_table(result.value_bets, config=RiskConfig(n_rounds=rounds, initial_bankroll=bankroll))
            st.dataframe(table.round(3), hide_index=True, use_container_width=True)
            st.caption("p_ruina = caer por debajo del 50% del bankroll; p_dd_30 = drawdown mayor al 30%.")

    with tabs[7]:
        st.subheader("Jornada multiliga")
        st.caption("Analiza varias ligas, permite parlays entre ellas y aplica el tope de exposición de forma global.")
        leagues = st.multiselect("Ligas", selectable_leagues(), default=["EPL", "LA_LIGA", "SERIE_A"],
                                 format_func=lambda k: LEAGUES[k].name)
        max_singles = st.slider("Máximo de sencillas", 5, 50, 15)
        if leagues and st.button("Analizar jornada", type="primary"):
            try:
                slate = run_slate(source, seed, tuple(leagues), season, bankroll, use_ml, max_singles, odds_key,
                                  af_key)
            except (DataFetchError, AuthenticationError) as exc:
                st.error(str(exc))
            else:
                for key, err in slate.errors.items():
                    st.warning(f"{LEAGUES[key].name} omitida: {err}")
                st.metric("Exposición total", f"{slate.total_stake:,.2f}",
                          f"{100 * slate.total_stake / bankroll:.1f}% del bankroll")
                st.dataframe(pct(slate.value_bets_frame(), ["p_modelo", "p_implícita", "p_mercado", "EV_modelo",
                                                            "EV_ajustado"]), hide_index=True, use_container_width=True)
                st.dataframe(pct(slate.parlays_frame(), ["p_conjunta", "p_implícita", "EV"]),
                             hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
