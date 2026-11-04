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
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from predictive_odds.backtester import BacktestConfig, Backtester, BacktestResult
from predictive_odds.bet_tracker import BetTracker, TrackerError, restore_backup
from predictive_odds.config import LEAGUES, ConfigurationError, current_season, get_league, selectable_leagues
from predictive_odds.connectivity import API_FOOTBALL_SIGNUP, ODDS_API_SIGNUP, check_all
from predictive_odds.data_fetcher import (
    AuthenticationError,
    DataFetcher,
    DataFetchError,
    FootballDataCoUkProvider,
    SimulatedDataProvider,
    SimulationConfig,
    TheOddsApiProvider,
)
from predictive_odds.parlay_builder import ParlayConfig, ValueBetConfig, ValueBetFinder
from predictive_odds import ui_html as ui
from predictive_odds.domain import MatchPrediction
from predictive_odds.performance import breakdown, cumulative, summarize
from predictive_odds.pipeline import AnalysisPipeline, AnalysisResult, PipelineConfig, SlateResult, analyze_slate
from predictive_odds.league_study import LeagueProfile, StudyConfig, run_study, save_profiles, study_frame
from predictive_odds.report import render_html, render_markdown
from predictive_odds.risk import RiskConfig, risk_table
from predictive_odds.settings import env_file_keys, load_env
from predictive_odds.sources import make_fetcher
from predictive_odds.statistical_models import InsufficientDataError

DISCLAIMER = "Herramienta analítica: no coloca apuestas. Apostar implica riesgo de pérdida; juega con responsabilidad."


def build_fetcher(source: str, seed: int, seasons: tuple[int, int], odds_key: str,
                  af_key: str = "", league_key: str = "EPL") -> DataFetcher:
    """Fachada de datos de la app (misma lógica que el CLI: :func:`predictive_odds.sources.make_fetcher`)."""
    return make_fetcher(source == "Real", league_key, seasons, odds_key, af_key, seed=seed)


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
        st.altair_chart(line_chart(bt.bankroll, "date", {"bankroll": "Bankroll"}, "Bankroll"),
                        use_container_width=True)
    st.dataframe(bt.by_market().round(3).rename_axis("mercado"), use_container_width=True)
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


@st.cache_data(show_spinner="Estudiando cada liga (puede tardar 1-2 minutos)…", ttl=3600)
def run_study_cached(source: str, seed: int, season: int, leagues: tuple[str, ...],
                     seasons_back: int) -> tuple[dict[str, LeagueProfile], dict[str, str]]:
    """Estudio por liga memoizado (datos reales de football-data o simulados)."""
    provider: SimulatedDataProvider | FootballDataCoUkProvider
    if source == "Real":
        provider = FootballDataCoUkProvider()
    else:
        provider = SimulatedDataProvider(SimulationConfig(seed=seed, seasons=tuple(range(season - seasons_back,
                                                                                         season + 1))))
    return run_study(provider, leagues, StudyConfig(seasons_back=seasons_back), season)


def study_panel(source: str, seed: int, season: int) -> None:
    """Estudio por liga y mercado: tabla de resultados y guardado del perfil."""
    st.subheader("Estudio por liga y mercado")
    st.caption("Backtest de varias temporadas por liga: indica en qué mercados el modelo tiene ventaja real "
               "(CLV positivo) y guarda un perfil para que la app solo sugiera esos mercados.")
    leagues = st.multiselect("Ligas a estudiar", selectable_leagues(), default=selectable_leagues(),
                             format_func=lambda k: LEAGUES[k].name, key="study_leagues")
    seasons_back = st.slider("Temporadas anteriores", 1, 5, 3, key="study_seasons")
    if leagues and st.button("Ejecutar estudio", key="study_run"):
        try:
            profiles, errors = run_study_cached(source, seed, season, tuple(leagues), seasons_back)
        except DataFetchError as exc:
            st.error(str(exc))
            return
        for key, err in errors.items():
            st.warning(f"{LEAGUES[key].name} omitida: {err}")
        frame = study_frame(profiles)
        st.dataframe(pct(frame, ["roi", "clv", "clv_positivo"]), hide_index=True, use_container_width=True)
        for key, prof in profiles.items():
            st.write(f"**{LEAGUES[key].name}**: {', '.join(prof.markets) if prof.markets else 'ningún mercado'}")
        if source == "Real" and profiles:
            path = save_profiles(profiles)
            st.success(f"Perfiles guardados en `{path.resolve()}`. Se aplican automáticamente en los análisis.")
        elif profiles:
            st.info("Datos simulados: los perfiles no se guardan. Cambia a *Real* para el estudio de verdad.")


SECRET_KEYS = ("ODDS_API_KEY", "API_FOOTBALL_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


def backup_panel(db_path: str) -> None:
    """Copia de seguridad del registro (imprescindible en la nube, donde el disco se borra al reiniciar)."""
    with st.expander("Copia de seguridad del registro"):
        st.caption("En Streamlit Community Cloud el disco se borra cuando la app se reinicia: descarga una copia "
                   "después de registrar o liquidar y restáurala cuando haga falta.")
        path = Path(db_path)
        if path.is_file():
            st.download_button("Descargar copia (.sqlite)", path.read_bytes(), file_name=path.name,
                               mime="application/octet-stream", key=f"dl_{db_path}")
        uploaded = st.file_uploader("Restaurar desde una copia", type=["sqlite", "db"], key=f"up_{db_path}")
        if uploaded is not None:
            confirm = st.checkbox("Entiendo que esto reemplaza el registro actual", key=f"ok_{db_path}")
            if st.button("Restaurar copia", disabled=not confirm, key=f"restore_{db_path}"):
                try:
                    n = restore_backup(uploaded.getvalue(), db_path)
                    st.success(f"Registro restaurado: {n} apuesta(s).")
                except TrackerError as exc:
                    st.error(str(exc))


def load_streamlit_secrets() -> list[str]:
    """Copia a las variables de entorno las claves definidas como *secrets* de Streamlit.

    En Streamlit Community Cloud los secrets se configuran en el panel de la app.
    Las variables de entorno existentes (p. ej. del ``.env``) tienen prioridad.

    Returns:
        Nombres de las claves copiadas (nunca sus valores).
    """
    loaded = []
    try:
        for key in SECRET_KEYS:
            if not os.environ.get(key) and key in st.secrets:
                os.environ[key] = str(st.secrets[key])
                loaded.append(key)
    except Exception:  # sin archivo de secrets (uso local): no hay nada que cargar
        return []
    return loaded


ICON = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 38 38'>"
        "<rect width='38' height='38' rx='7' fill='%230f1613'/><path d='M8 27 C13 15, 23 11, 30 10' fill='none' "
        "stroke='%23e8a93b' stroke-width='3.4' stroke-linecap='round'/></svg>")
CHALK, CHALK_3, LINE = "#e9ece6", "#7c877f", "#26332c"
LEAN = {"HOME": "Local", "DRAW": "Empate", "AWAY": "Visitante"}
SPANISH_MONTHS = ("['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic'][month(datum.value)]"
                  " + ' ' + timeFormat(datum.value, '%y')")


def _themed(chart: alt.Chart) -> alt.Chart:
    """Estilo común de las gráficas: fondo transparente, ejes discretos, tipografía de la app."""
    return (chart.configure(background="transparent", font="IBM Plex Sans")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=CHALK_3, titleColor=CHALK_3, gridColor=LINE, domainColor=LINE,
                            tickColor=LINE, labelFontSize=11, titleFontSize=11, titleFontWeight="normal")
            .configure_legend(labelColor=CHALK, titleColor=CHALK_3, orient="top", labelFontSize=12))


def line_chart(df: pd.DataFrame, x: str, series: dict[str, str], y_title: str) -> alt.Chart:
    """Líneas finas (2 px) con tooltip; leyenda solo si hay más de una serie.

    Args:
        df: Datos.
        x: Columna del eje X.
        series: ``columna -> etiqueta`` de cada serie (colores de la paleta validada, en orden fijo).
        y_title: Título del eje Y.
    """
    colors = [ui.HOME, ui.AWAY]
    long = df.melt(id_vars=[x], value_vars=list(series), var_name="serie", value_name="valor")
    long["serie"] = long["serie"].map(series)
    scale = alt.Scale(domain=list(series.values()), range=colors[:len(series)])
    legend = alt.Legend(title=None) if len(series) > 1 else None
    if pd.api.types.is_numeric_dtype(df[x]):
        x_enc = alt.X(f"{x}:Q", title=None)
    else:  # meses en español (Vega formatea las fechas en inglés)
        x_enc = alt.X(f"{x}:T", title=None, axis=alt.Axis(labelExpr=SPANISH_MONTHS))
    chart = alt.Chart(long).mark_line(strokeWidth=2).encode(
        x=x_enc,
        y=alt.Y("valor:Q", title=y_title, scale=alt.Scale(zero=False)),
        color=alt.Color("serie:N", scale=scale, legend=legend),
        tooltip=[alt.Tooltip(f"{x}", title="Punto"), alt.Tooltip("serie:N", title="Serie"),
                 alt.Tooltip("valor:Q", title=y_title, format=",.2f")],
    ).properties(height=260)
    return _themed(chart)


def score_heatmap(pred: MatchPrediction, n: int = 6) -> alt.Chart:
    """Mapa de calor de marcadores (escala secuencial de un solo tono) con el % escrito."""
    rows = [{"local": str(i), "visitante": str(j), "p": float(pred.score_matrix[i, j])}
            for i in range(n) for j in range(n)]
    data = pd.DataFrame(rows)
    base = alt.Chart(data).encode(
        x=alt.X("visitante:O", title=f"Goles {pred.away_team}", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("local:O", title=f"Goles {pred.home_team}", sort="descending"),
    )
    cells = base.mark_rect(cornerRadius=3, stroke="#0f1613", strokeWidth=2).encode(
        color=alt.Color("p:Q", scale=alt.Scale(range=["#18242f", "#3987e5"]), legend=None),
        tooltip=[alt.Tooltip("local:O", title="Local"), alt.Tooltip("visitante:O", title="Visitante"),
                 alt.Tooltip("p:Q", title="Probabilidad", format=".1%")],
    )
    labels = base.mark_text(font="IBM Plex Mono", fontSize=11).encode(
        text=alt.condition("datum.p >= 0.01", alt.Text("p:Q", format=".0%"), alt.value("")),
        color=alt.condition("datum.p >= 0.08", alt.value("#0b0f0d"), alt.value(CHALK)),
    )
    return _themed((cells + labels).properties(height=280))


def pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Formatea columnas de probabilidad como porcentaje para mostrar."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = (100 * out[c].astype(float)).round(1)
    return out


def main() -> None:
    """Construye la interfaz."""
    load_env()  # claves desde .env (si existe); las variables del sistema tienen prioridad
    load_streamlit_secrets()  # en Streamlit Community Cloud las claves llegan como secrets
    env_loaded = env_file_keys()  # en cada recarga, no solo la primera
    st.set_page_config(page_title="Predictive Sports Odds", page_icon=ICON, layout="wide")
    st.markdown(ui.CSS, unsafe_allow_html=True)
    header_slot = st.empty()

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
            elif os.environ.get("ODDS_API_KEY"):
                st.caption("✓ Clave de momios configurada en el servidor")
            elif not os.environ.get("ODDS_API_KEY"):
                st.caption("Consejo: guarda tus claves en un archivo `.env` (ver `.env.example`) "
                           "para no tener que pegarlas cada vez.")
            # Las claves del servidor (.env o secrets de la nube) NUNCA se envían al navegador: el campo queda
            # vacío y se usa la del servidor salvo que el usuario escriba otra.
            server_odds = os.environ.get("ODDS_API_KEY", "")
            server_af = os.environ.get("API_FOOTBALL_KEY", "")
            typed_odds = st.text_input("Clave de The Odds API", type="password",
                                       placeholder="Configurada en el servidor" if server_odds else "",
                                       help="Obligatoria. Plan gratuito: 500 créditos/mes")
            typed_af = st.text_input("Clave de API-Football (opcional)", type="password",
                                     placeholder="Configurada en el servidor" if server_af else "",
                                     help="Activa el ajuste por lesiones. Requiere plan de pago: el gratuito "
                                          "no cubre la temporada actual")
            odds_key = typed_odds.strip() or server_odds
            af_key = typed_af.strip() or server_af
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
        db_path = st.text_input("Registro de apuestas (SQLite)", "bets.sqlite",
                                help="Sugerencias registradas (seguimiento / paper trading)")
        real_db_path = st.text_input("Mis apuestas reales (SQLite)", "mis_apuestas.sqlite",
                                     help="Lo que realmente apostaste, con el momio y el monto que tomaste")
        my_book = st.text_input("Tu casa de apuestas", os.environ.get("MI_CASA", "Caliente"),
                                help="Casa sin momios automáticos: la app te dice el momio mínimo para que haya valor")
        st.header("Informe")
        report_slot = st.empty()

    header_slot.markdown(ui.masthead(LEAGUES[league_key].name, source == "Real", season), unsafe_allow_html=True)
    with st.sidebar:
        st.caption(DISCLAIMER)

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

    if result.profile_markets is not None:
        allowed = ", ".join(result.profile_markets) or "ninguno"
        st.info(f"Perfil de **{LEAGUES[league_key].name}** aplicado (estudio por liga): solo se sugieren "
                f"apuestas en **{allowed}**. Puedes rehacer el estudio en la pestaña *Backtest*.")
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
                    "Riesgo", "Multiliga", "Rendimiento", "Mis apuestas"])

    with tabs[0]:
        counts = ui.value_counts(result.value_bets)
        st.markdown(ui.grid(ui.match_card(p, counts.get(mid, 0)) for mid, p in result.predictions.items()),
                    unsafe_allow_html=True)
        with st.expander("Ver como tabla"):
            st.dataframe(pct(result.predictions_frame().drop(columns="match_id"),
                             ["P(1)", "P(X)", "P(2)", "P(O2.5)", "P(BTTS)"]), hide_index=True, use_container_width=True)
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
            st.write(f"H2H: {rep.h2h.a_wins}G-{rep.h2h.draws}E-{rep.h2h.b_wins}P · índice {rep.score:+.3f} → **{LEAN[rep.lean.value]}**")
            for note in rep.notes:
                st.write(f"• {note}")
            for team in (pred.home_team, pred.away_team):
                if team in result.adjustments:
                    a = result.adjustments[team]
                    st.warning(f"{team}: ataque ×{a.attack:.2f}, goles recibidos ×{a.defense:.2f} ({a.reason})")
        with right:
            st.subheader("Marcadores más probables")
            st.altair_chart(score_heatmap(pred), use_container_width=True)

    with tabs[1]:
        vb = result.value_bets_frame()
        if vb.empty:
            st.info("Ninguna selección supera el umbral de valor.")
        else:
            st.markdown('<p class="note">Ordenadas por EV ajustado (modelo encogido hacia el mercado). El stake ya '
                        'incluye Kelly modificado y el tope de exposición.</p>', unsafe_allow_html=True)
            st.markdown(ui.grid(ui.pick_ticket(p, result.bankroll, my_book, min_edge) for p in result.value_bets),
                        unsafe_allow_html=True)
            with st.expander("Ver como tabla"):
                st.dataframe(pct(vb, ["p_modelo", "p_implícita", "p_mercado", "EV_modelo", "EV_ajustado"]),
                             hide_index=True, use_container_width=True)

    with tabs[2]:
        pf = result.parlays_frame()
        if pf.empty:
            st.info("No hay combinaciones que cumplan los criterios de riesgo y valor.")
        else:
            st.markdown(ui.grid(ui.parlay_ticket(par, result.bankroll) for par in result.parlays),
                        unsafe_allow_html=True)
            with st.expander("Ver como tabla"):
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
        st.divider()
        study_panel(source, seed, season)

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
                    curve = history.reset_index(drop=True).reset_index(names="apuesta")
                    st.altair_chart(line_chart(curve, "apuesta", {"bankroll": "Bankroll"}, "Bankroll"),
                                    use_container_width=True)
            backup_panel(db_path)
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

    with tabs[8]:
        which = st.radio("Registro", ["Mis apuestas reales", "Sugerencias"], horizontal=True,
                         key="perf_registry")
        performance_panel(real_db_path if which == "Mis apuestas reales" else db_path)

    with tabs[9]:
        my_bets_panel(result, real_db_path, my_book or "Mi casa", min_edge, value_weight, bankroll,
                      source, seed, league_key, season, odds_key, af_key)


def my_bets_panel(result: AnalysisResult, real_db_path: str, my_book: str, min_edge: float, value_weight: float,
                  bankroll: float, source: str, seed: int, league_key: str, season: int, odds_key: str,
                  af_key: str) -> None:
    """Momios mínimos para tu casa y registro de las apuestas que realmente hiciste."""
    st.subheader(f"Momios mínimos en {my_book}")
    st.caption(f"Si {my_book} paga **igual o más** que el momio mínimo, la selección tiene valor (EV ajustado ≥ "
               f"{min_edge:.0%}). Se calcula con la probabilidad del modelo encogida hacia el mercado.")
    candidates = sorted((p for p in result.all_picks if 0.12 <= (p.decision_prob or p.model_prob) <= 0.92),
                        key=lambda p: (p.kickoff is None, str(p.kickoff), p.fixture, p.minimum_odds(min_edge)))
    if not candidates:
        st.info("No hay selecciones con momios de referencia en esta jornada.")
        return
    table = pd.DataFrame([{"partido": p.fixture, "selección": p.selection.label,
                           "momio mínimo": round(p.minimum_odds(min_edge), 2),
                           "mejor momio visto": p.odds, "casa": p.bookmaker,
                           "prob. (modelo+mercado)": round(100 * (p.decision_prob or p.model_prob), 1)}
                          for p in candidates])
    match = st.selectbox("Partido", ["Todos"] + list(dict.fromkeys(table["partido"])), key="min_match")
    view = table if match == "Todos" else table[table["partido"] == match]
    st.dataframe(view, hide_index=True, use_container_width=True)

    st.subheader("Registrar una apuesta real")
    labels = {i: f"{p.fixture} · {p.selection.label} (mín. {p.minimum_odds(min_edge):.2f})"
              for i, p in enumerate(candidates)}
    idx = st.selectbox("Selección", list(labels), format_func=labels.__getitem__, key="real_pick")
    pick = candidates[idx]
    c1, c2 = st.columns(2)
    taken = c1.number_input(f"Momio que tomaste en {my_book}", min_value=1.01, max_value=100.0,
                            value=float(round(max(pick.minimum_odds(min_edge), 1.01), 2)), step=0.05,
                            key="real_odds")
    finder = ValueBetFinder(replace(ValueBetConfig(), min_edge=min_edge, value_model_weight=value_weight))
    priced = finder.reprice(pick, taken, my_book)
    suggested = round(priced.stake_fraction * bankroll, 2)
    amount = c2.number_input("Monto apostado", min_value=0.0, value=float(suggested), step=10.0, key="real_amount",
                             help=f"Stake sugerido por Kelly modificado: {suggested:,.2f}")
    if priced.adjusted_edge >= min_edge:
        st.success(f"Con {taken:.2f} hay valor: EV ajustado {priced.adjusted_edge:+.1%}.")
    else:
        st.warning(f"Con {taken:.2f} NO hay valor (EV ajustado {priced.adjusted_edge:+.1%}; mínimo "
                   f"{pick.minimum_odds(min_edge):.2f}). Puedes registrarla igualmente.")
    try:
        with BetTracker(real_db_path) as tracker:
            if st.button("Guardar apuesta real", type="primary", disabled=amount <= 0):
                new = tracker.record_pick(priced, amount)
                if new is None:
                    st.info("Esa apuesta ya estaba registrada.")
                else:
                    st.success(f"Guardada en {Path(real_db_path).name}.")
            if st.button("Liquidar mis apuestas con resultados", key="settle_real"):
                try:
                    with st.spinner("Descargando resultados…"):
                        n = settle_pending(tracker, source, seed, league_key, season, odds_key, af_key)
                    if n:
                        st.success(f"{n} apuesta(s) liquidada(s).")
                    else:
                        st.info("Nada que liquidar todavía (los resultados tardan 1-2 días en publicarse).")
                except (DataFetchError, AuthenticationError) as exc:
                    st.error(f"No se pudieron obtener resultados: {exc}")
            bets = tracker.bets()
            if not bets.empty:
                st.dataframe(bets.drop(columns=["bet_key", "legs"]), hide_index=True, use_container_width=True)
        backup_panel(real_db_path)
    except TrackerError as exc:
        st.error(f"Mis apuestas: {exc}")


def performance_panel(db_path: str) -> None:
    """Rendimiento real del registro: habilidad (según el cierre) frente a suerte."""
    st.subheader("Rendimiento real: habilidad frente a suerte")
    st.caption("El beneficio *esperado* usa la probabilidad justa del momio de cierre (la mejor estimación de "
               "la probabilidad real) cuando se conoce; si no, la del modelo. Real − esperado = suerte.")
    try:
        with BetTracker(db_path) as tracker:
            bets = tracker.bets()
    except TrackerError as exc:
        st.error(f"Registro de apuestas: {exc}")
        return
    summary = summarize(bets)
    if summary.bets == 0:
        st.info("Aún no hay apuestas liquidadas. Registra sugerencias en la pestaña *Registro* y liquídalas "
                "cuando se jueguen los partidos.")
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Beneficio real", f"{summary.profit:,.2f}", f"ROI {summary.roi:+.1%}")
    m2.metric("Beneficio esperado", f"{summary.expected_profit:,.2f}", f"ROI {summary.expected_roi:+.1%}")
    m3.metric("Suerte", f"{summary.luck:+,.2f}", f"{summary.luck_z:+.1f} desv. típicas", delta_color="off")
    m4.metric("Racha actual", f"{summary.current_streak:+d}", f"peor racha perdedora: {summary.longest_losing_streak}",
              delta_color="off")
    st.info(summary.verdict())
    if summary.closing_coverage < 0.5:
        st.caption(f"Solo el {summary.closing_coverage:.0%} de las apuestas tiene momio de cierre: el esperado "
                   "usa sobre todo la probabilidad del modelo (optimista). Usa capturas de momios para tener "
                   "el cierre propio.")
    st.altair_chart(line_chart(cumulative(bets), "n", {"real": "Real", "esperado": "Esperado (cierre)"},
                               "Beneficio acumulado"), use_container_width=True)
    left, right = st.columns(2)
    left.markdown("**Por liga**")
    left.dataframe(pct(breakdown(bets, "league"), ["roi", "roi_esperado", "clv_medio", "acierto"]).round(2),
                   hide_index=True, use_container_width=True)
    right.markdown("**Por mercado**")
    right.dataframe(pct(breakdown(bets, "market"), ["roi", "roi_esperado", "clv_medio", "acierto"]).round(2),
                    hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
