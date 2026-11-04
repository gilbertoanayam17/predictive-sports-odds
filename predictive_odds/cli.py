"""Interfaz de línea de comandos (análisis, backtest y registro de apuestas).

Uso::

    python main.py                       # Premier League simulada (sin red ni API keys)
    python main.py --league LIGA_MX --bankroll 5000
    python main.py --live --league EPL   # datos reales: football-data.co.uk + The Odds API
    python main.py --backtest            # backtest walk-forward (ROI, CLV, drawdown, calibración)
    python main.py --track bets.sqlite   # guarda las apuestas sugeridas
    python main.py --track bets.sqlite --settle   # liquida con resultados y muestra el P&L

En modo ``--live`` se requiere la variable de entorno ``ODDS_API_KEY``
(y opcionalmente ``API_FOOTBALL_KEY`` para calendario/ligas sin CSV).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

from predictive_odds.config import ConfigurationError, current_season, get_league, season_for, selectable_leagues
from predictive_odds.data_fetcher import (
    AuthenticationError,
    DataFetcher,
    DataFetchError,
    FootballDataCoUkProvider,
    SimulatedDataProvider,
    SimulationConfig,
)
from predictive_odds.bet_tracker import BetTracker, TrackerError
from predictive_odds.backtester import BacktestConfig, Backtester, BacktestResult
from predictive_odds.parlay_builder import ParlayConfig, Pick, ValueBetConfig
from predictive_odds.statistical_models import InsufficientDataError
from predictive_odds.odds_history import OddsHistory, OddsHistoryError
from predictive_odds.league_study import StudyConfig, run_study, save_profiles, study_frame
from predictive_odds.notifier import (
    NotifierError,
    TelegramNotifier,
    format_weekly,
    mark_sent,
    notify_value_bets,
    select_alerts,
)
from predictive_odds.performance import summarize
from predictive_odds.report import save_report
from predictive_odds.risk import RiskConfig, risk_table
from predictive_odds.settings import load_env
from predictive_odds.sources import make_fetcher
from predictive_odds.tuning import tune_ml_weight, tune_poisson
from predictive_odds.pipeline import AnalysisPipeline, AnalysisResult, PipelineConfig, SlateResult, analyze_slate

SEPARATOR = "=" * 100


def section(title: str) -> None:
    """Imprime un encabezado de sección."""
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")


def show(df: pd.DataFrame, empty_msg: str = "(sin resultados)") -> None:
    """Imprime un DataFrame con formato legible."""
    if df.empty:
        print(empty_msg)
        return
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_colwidth", 70,
                           "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))


def build_fetcher(args: argparse.Namespace) -> DataFetcher:
    """Fachada de datos según los argumentos (ver :func:`predictive_odds.sources.make_fetcher`).

    Raises:
        AuthenticationError: Si faltan API keys en modo en vivo.
    """
    return make_fetcher(args.live, args.league, (args.season - 1, args.season),
                        os.environ.get("ODDS_API_KEY", ""), os.environ.get("API_FOOTBALL_KEY", ""),
                        seed=args.seed, played_rounds=args.played_rounds, include_btts=args.btts,
                        understat=args.understat)


def run_backtest(args: argparse.Namespace) -> BacktestResult:
    """Ejecuta un backtest walk-forward sobre las dos últimas temporadas.

    En modo simulado usa los momios históricos sintéticos (con línea de cierre);
    en vivo, los CSV de football-data.co.uk (con el cierre de Pinnacle).

    Args:
        args: Argumentos de línea de comandos.

    Returns:
        Resultado del backtest.

    Raises:
        ProviderNotSupportedError: Si la liga no tiene momios históricos.
    """
    league = get_league(args.league)
    seasons = (args.season - 1, args.season)
    provider: SimulatedDataProvider | FootballDataCoUkProvider
    if args.live:
        provider = FootballDataCoUkProvider()
    else:
        provider = SimulatedDataProvider(SimulationConfig(seed=args.seed, seasons=seasons))
    history = DataFetcher(provider).get_history(league.key, seasons)
    odds = pd.concat([provider.fetch_historical_odds(league, s) for s in seasons], ignore_index=True)
    config = BacktestConfig(initial_bankroll=args.bankroll, use_ml=not args.no_ml, include_parlays=args.parlays)
    return Backtester(history, odds, config).run()


def run_tuning(args: argparse.Namespace) -> None:
    """Ajuste de hiperparámetros con validación cronológica e informe en la terminal.

    Args:
        args: Argumentos de línea de comandos.
    """
    league = get_league(args.league)
    seasons = (args.season - 1, args.season)
    provider: SimulatedDataProvider | FootballDataCoUkProvider
    if args.live:
        provider = FootballDataCoUkProvider()
    else:
        provider = SimulatedDataProvider(SimulationConfig(seed=args.seed, seasons=seasons))
    history = DataFetcher(provider).get_history(league.key, seasons)
    tag = "" if args.live else " [DATOS SIMULADOS: solo valida la herramienta]"
    section(f"AJUSTE DE POISSON (rolling origin; log-loss medio 1X2/OU2.5/BTTS){tag}")
    res = tune_poisson(history, {"half_life_days": [120, 240, 480], "xg_weight": [0.0, 0.4, 0.7],
                                 "prior_sd": [0.4, 0.6, 1.0]})
    show(res.table.head(8).round(4))
    print(f"Mejor: {res.best} | mejora frente a la configuración actual: {100 * res.improvement:.2f}%")
    if not args.no_ml:
        section("PESO DEL ML EN EL ENSAMBLE")
        try:
            ml = tune_ml_weight(history, [0.0, 0.2, 0.35, 0.5])
            show(ml.table.round(4))
        except InsufficientDataError as exc:
            print(f"Omitido: {exc}")
    print("\nCambia los valores por defecto solo si la mejora es consistente sobre datos REALES.")


def send_alerts(picks: list[Pick], bankroll: float, title: str, min_edge: float) -> int:
    """Envía por Telegram los value bets nuevos; devuelve un código de salida (0 = bien).

    Args:
        picks: Sencillas de valor.
        bankroll: Bankroll para el stake en dinero.
        title: Encabezado del aviso.
        min_edge: EV ajustado mínimo.
    """
    notifier = TelegramNotifier.from_env()
    if notifier is None:
        print("Alertas: falta TELEGRAM_BOT_TOKEN en el .env (ver --telegram-setup).", file=sys.stderr)
        return 9
    try:
        sent = notify_value_bets(notifier, picks, bankroll, title, min_edge)
    except NotifierError as exc:
        print(f"Alertas: {exc}", file=sys.stderr)
        return 9
    print(f"\nTelegram: {sent} aviso(s) nuevo(s) enviado(s)." if sent
          else f"\nTelegram: nada nuevo con EV ajustado ≥ {min_edge:.0%}.")
    return 0


def telegram_setup() -> int:
    """Asistente: valida el token, encuentra el chat_id y envía un mensaje de prueba."""
    section("CONFIGURACIÓN DE TELEGRAM")
    notifier = TelegramNotifier.from_env()
    if notifier is None:
        print("1. En Telegram, abre @BotFather y envía /newbot; copia el token que te da.\n"
              "2. Añade a tu .env la línea  TELEGRAM_BOT_TOKEN=el_token\n"
              "3. Escribe cualquier mensaje a tu bot y vuelve a ejecutar:  python main.py --telegram-setup")
        return 9
    try:
        print(f"Bot: @{notifier.bot_name()} (token válido)")
        if not notifier.chat_id:
            chats = notifier.discover_chat_ids()
            if not chats:
                print("No encuentro chats: escribe cualquier mensaje a tu bot y repite el comando.")
                return 9
            for chat_id, name in chats:
                print(f"Chat encontrado: {name or '(sin nombre)'} → añade a tu .env:  TELEGRAM_CHAT_ID={chat_id}")
            notifier.chat_id = chats[0][0]
        notifier.send("✅ <b>Predictive Sports Odds</b>: las alertas de Telegram funcionan.")
        print("Mensaje de prueba enviado. Revisa Telegram.")
    except NotifierError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 9
    return 0


def run_league_study(args: argparse.Namespace) -> None:
    """Estudio por liga y mercado; guarda los perfiles que la app/CLI aplican después.

    Args:
        args: Argumentos de línea de comandos.
    """
    leagues = [k.upper() for k in args.leagues] if args.leagues else selectable_leagues()
    if args.live:
        provider: SimulatedDataProvider | FootballDataCoUkProvider = FootballDataCoUkProvider()
        season = args.season
    else:
        season = args.season
        provider = SimulatedDataProvider(SimulationConfig(
            seed=args.seed, seasons=tuple(range(season - args.study_seasons, season + 1))))
    tag = "" if args.live else " [DATOS SIMULADOS: solo valida la herramienta]"
    section(f"ESTUDIO POR LIGA Y MERCADO ({args.study_seasons + 1} temporadas){tag}")
    profiles, errors = run_study(provider, leagues, StudyConfig(seasons_back=args.study_seasons, use_ml=False),
                                 season)
    for key, err in errors.items():
        print(f"⚠ {key} omitida: {err}")
    frame = study_frame(profiles)
    for col in ("roi", "clv", "clv_positivo"):
        frame[col] = (100 * frame[col].astype(float)).round(1)
    show(frame)
    for key, prof in profiles.items():
        print(f"  {get_league(key).name}: {', '.join(prof.markets) if prof.markets else 'NINGÚN mercado'} recomendado(s)")
    if profiles and args.live:
        print(f"\nPerfiles guardados en {save_profiles(profiles)}: la app y el CLI solo sugerirán esos mercados.")
    elif profiles:
        print("\n(Datos simulados: los perfiles NO se guardan. Usa --live para el estudio real.)")


def print_backtest(result: BacktestResult, league_key: str, simulated: bool) -> None:
    """Imprime el informe del backtest.

    Args:
        result: Resultado del backtest.
        league_key: Liga.
        simulated: Si los datos son simulados.
    """
    tag = " [DATOS SIMULADOS]" if simulated else ""
    section(f"BACKTEST WALK-FORWARD — {get_league(league_key).name}{tag}")
    m = result.metrics
    pct = {"crecimiento", "roi", "acierto", "edge_esperado_medio", "clv_medio", "clv_positivo_%", "max_drawdown"}
    for k, v in m.items():
        print(f"  {k:<22} {100 * v:>10.2f}%" if k in pct else f"  {k:<22} {v:>11,.2f}")
    section("RENDIMIENTO POR MERCADO")
    show(result.by_market().round(3).reset_index())
    section("CALIBRACIÓN (todas las selecciones evaluadas)")
    show(result.calibration().round(3))
    print("\nCLV > 0 de forma sostenida es la mejor señal de un edge real; el ROI a corto plazo es muy ruidoso.")


def print_slate(slate: SlateResult, simulated: bool) -> None:
    """Imprime el resumen de una jornada multiliga.

    Args:
        slate: Resultado multiliga.
        simulated: Si los datos son simulados.
    """
    tag = " [DATOS SIMULADOS]" if simulated else ""
    section(f"JORNADA MULTILIGA: {', '.join(slate.results)}{tag}")
    for key, err in slate.errors.items():
        print(f"⚠ {key} omitida: {err}")
    for key, res in slate.results.items():
        print(f"  {get_league(key).name}: {len(res.predictions)} partidos, {len(res.all_picks)} selecciones evaluadas")
    section("SENCILLAS DE VALOR (todas las ligas, tope de exposición global)")
    show(slate.value_bets_frame().head(20), "Ninguna selección supera el umbral de valor.")
    if len(slate.value_bets) > 20:
        print(f"... y {len(slate.value_bets) - 20} más (usa --max-singles para limitarlas)")
    section("PARLAYS (pueden combinar ligas)")
    show(slate.parlays_frame(), "No hay combinaciones que cumplan los criterios.")
    print(f"\nExposición total: {slate.total_stake:,.2f} ({100 * slate.total_stake / slate.bankroll:.1f}% del bankroll)")


def print_risk(result: AnalysisResult, n_rounds: int) -> None:
    """Imprime la matriz Monte Carlo de riesgo para la jornada analizada.

    Args:
        result: Resultado del análisis (sus value bets son la "jornada tipo").
        n_rounds: Jornadas a simular.
    """
    section(f"RIESGO MONTE CARLO — repetir esta jornada {n_rounds} veces (bankroll final en múltiplos)")
    if not result.value_bets:
        print("Sin value bets: nada que simular.")
        return
    table = risk_table(result.value_bets, config=RiskConfig(n_rounds=n_rounds, initial_bankroll=result.bankroll))
    show(table.round(3))
    print("recorte_edge: 0 = el modelo acierta; 0.5 = edge real a la mitad; 1 = el modelo no aporta;"
          " 1.5 = el modelo se equivoca en contra. p_ruina = caer por debajo del 50% del bankroll.")


def print_report(result: AnalysisResult, simulated: bool) -> None:
    """Imprime el informe completo en la terminal.

    Args:
        result: Resultado del pipeline.
        simulated: Si los datos son simulados.
    """
    tag = " [DATOS SIMULADOS]" if simulated else ""
    section(f"1. TABLA DE POSICIONES — {get_league(result.league).name}{tag}")
    show(result.standings.head(10))

    section("2. RATINGS DEL MODELO DIXON-COLES (multiplicadores sobre la media de la liga)")
    show(result.team_ratings.head(8).round(3))

    section("3. VALIDACIÓN FUERA DE MUESTRA (1X2, partición cronológica) — menor log-loss es mejor")
    show(result.evaluation_frame().round(4).reset_index(names="modelo"), "(histórico insuficiente)")

    section("4. ANÁLISIS HEURÍSTICO + PREDICCIONES DE LA PRÓXIMA JORNADA")
    show(result.predictions_frame().drop(columns="match_id").round(3))
    for mid, rep in list(result.reports.items())[:3]:
        pred = result.predictions[mid]
        print(f"\n▸ {pred.home_team} vs {pred.away_team}")
        print(f"   Local:     pos {rep.home.position}, forma {rep.home.form_overall.results or '-'} "
              f"(en casa {rep.home.form_venue.results or '-'}), calendario rival PPG {rep.home.strength_of_schedule}")
        print(f"   Visitante: pos {rep.away.position}, forma {rep.away.form_overall.results or '-'} "
              f"(fuera {rep.away.form_venue.results or '-'}), calendario rival PPG {rep.away.strength_of_schedule}")
        print(f"   H2H: {rep.h2h.a_wins}G-{rep.h2h.draws}E-{rep.h2h.b_wins}P ({', '.join(rep.h2h.last_results) or '-'})")
        print(f"   Índice heurístico {rep.score:+.3f} → tendencia {rep.lean.value}; "
              f"marcadores más probables: " + ", ".join(f"{h}-{a} ({p:.1%})" for h, a, p in pred.most_likely_scores(3)))
        if "ml" in pred.sources:
            from predictive_odds.domain import Selection
            s = Selection.home()
            print(f"   P(Local): Poisson {pred.sources['poisson'][s]:.3f} | ML {pred.sources['ml'][s]:.3f} "
                  f"| Ensamble {pred.probability(s):.3f}")
        for note in rep.notes:
            print(f"   • {note}")

    if result.adjustments:
        print("\nAjustes por lesiones aplicados:")
        for team, adj in result.adjustments.items():
            print(f"   • {team}: ataque x{adj.attack:.2f}, goles recibidos x{adj.defense:.2f} — {adj.reason}")

    if result.unknown_teams:
        print(f"\n⚠ Sin historial (fuerza promedio asumida): {', '.join(result.unknown_teams)}. "
              "Si no son recién ascendidos, revisa --season.")

    section(f"5. APUESTAS DE VALOR (sencillas) — bankroll {result.bankroll:,.0f}")
    vb = result.value_bets_frame()
    show(vb.head(15), "Ninguna selección supera el umbral de valor.")
    if len(vb) > 15:
        print(f"... y {len(vb) - 15} más (ordenadas por EV)")

    section("6. PARLAYS SUGERIDOS (Kelly modificado, sin patas contradictorias)")
    show(result.parlays_frame(), "No hay combinaciones que cumplan los criterios de riesgo/valor.")
    for i, parlay in enumerate(result.parlays, 1):
        for w in parlay.warnings:
            print(f"   ⚠ Parlay {i}: {w}")

    section("RESUMEN")
    print(f"Partidos analizados: {len(result.predictions)} | selecciones evaluadas: {len(result.all_picks)} | "
          f"value bets: {len(result.value_bets)} | parlays: {len(result.parlays)}")
    print(f"Exposición total sugerida: {result.total_stake:,.2f} ({100 * result.total_stake / result.bankroll:.1f}% "
          f"del bankroll)")
    print("Aviso: herramienta analítica; las apuestas implican riesgo de pérdida. Juega con responsabilidad.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos."""
    p = argparse.ArgumentParser(description="Value bets y parlays con modelos estadísticos")
    p.add_argument("--league", default="EPL", help="EPL, LA_LIGA, SERIE_A, BUNDESLIGA o LIGA_MX")
    p.add_argument("--leagues", nargs="+", default=None,
                   help="Jornada multiliga (p. ej. --leagues EPL LIGA_MX UCL): parlays entre ligas y tope global")
    p.add_argument("--max-singles", type=int, default=None, help="Máximo de sencillas sugeridas (mayor EV ajustado)")
    p.add_argument("--season", type=int, default=None,
                   help="Temporada actual (año de inicio; por defecto la temporada en curso de la liga)")
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=7, help="Semilla de la simulación")
    p.add_argument("--played-rounds", type=int, default=28,
                   help="Simulación: jornadas jugadas de la temporada actual (sube a 29 para liquidar)")
    p.add_argument("--live", action="store_true", help="Usar APIs reales (requiere ODDS_API_KEY)")
    p.add_argument("--understat", action="store_true",
                   help="En vivo: añadir xG de Understat al histórico (experimental; solo ligas europeas)")
    p.add_argument("--btts", action="store_true", help="En vivo: pedir también momios BTTS (más créditos)")
    p.add_argument("--no-ml", action="store_true", help="Desactivar el modelo de ML")
    p.add_argument("--same-match", action="store_true", help="Permitir patas del mismo partido (Bet Builder)")
    p.add_argument("--extra-leagues", nargs="*", default=None,
                   help="Ligas domésticas para el modelo multiliga (por defecto EPL si la liga es internacional)")
    p.add_argument("--model-path", default=None, help="Bundle .joblib: reutiliza modelos guardados y guarda al reentrenar")
    p.add_argument("--track", default=None, help="Registro SQLite: guarda las apuestas sugeridas del análisis")
    p.add_argument("--settle", action="store_true", help="Con --track: liquida pendientes y muestra el rendimiento")
    p.add_argument("--snapshot", default=None,
                   help="SQLite de histórico de momios: guarda una captura (y con --settle aporta el cierre propio)")
    p.add_argument("--movement", action="store_true", help="Con --snapshot: muestra el movimiento de línea")
    p.add_argument("--report", default=None, help="Exportar el análisis a un archivo .md o .html")
    p.add_argument("--risk", action="store_true", help="Añadir simulación Monte Carlo de riesgo del bankroll")
    p.add_argument("--risk-rounds", type=int, default=38, help="Jornadas a simular en --risk")
    p.add_argument("--alert", action="store_true",
                   help="Enviar por Telegram los value bets nuevos (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID en .env)")
    p.add_argument("--alert-min-ev", type=float, default=0.05, help="EV ajustado mínimo para avisar (0.05 = 5%%)")
    p.add_argument("--weekly", action="store_true",
                   help="Ciclo semanal: liquida, analiza las ligas, registra sugerencias y avisa por Telegram")
    p.add_argument("--real-track", default="mis_apuestas.sqlite", help="Registro de tus apuestas reales")
    p.add_argument("--telegram-setup", action="store_true",
                   help="Comprueba el bot de Telegram, encuentra tu chat_id y envía un mensaje de prueba")
    p.add_argument("--study", action="store_true",
                   help="Estudio por liga y mercado (backtest multi-temporada) y perfiles recomendados")
    p.add_argument("--study-seasons", type=int, default=3, help="Temporadas anteriores incluidas en --study")
    p.add_argument("--tune", action="store_true", help="Ajustar hiperparámetros con validación cronológica")
    p.add_argument("--backtest", action="store_true", help="Ejecutar un backtest walk-forward en vez del análisis")
    p.add_argument("--parlays", action="store_true", help="Backtest: incluir también los parlays")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada: ejecuta el flujo completo y devuelve el código de salida."""
    load_env()  # ODDS_API_KEY / API_FOOTBALL_KEY desde .env (las variables del sistema tienen prioridad)
    args = parse_args(argv)
    if args.season is None:
        try:
            args.season = season_for(args.league)
        except ConfigurationError:
            args.season = current_season()  # la liga inválida se informa más adelante
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.telegram_setup:
        return telegram_setup()
    if args.weekly:
        return run_weekly(args)
    if args.settle:
        if not args.track:
            print("--settle requiere --track <archivo.sqlite>", file=sys.stderr)
            return 2
        return run_tracker(args)
    try:
        league = get_league(args.league)
        if args.tune:
            run_tuning(args)
            return 0
        if args.study:
            run_league_study(args)
            return 0
        if args.backtest:
            print_backtest(run_backtest(args), league.key, simulated=not args.live)
            return 0
        fetcher = build_fetcher(args)
        if args.leagues:
            slate_cfg = PipelineConfig(seasons=(args.season - 1, args.season), bankroll=args.bankroll,
                                       use_ml=not args.no_ml, max_singles=args.max_singles,
                                       parlay=ParlayConfig(allow_same_match=args.same_match))
            slate = analyze_slate(fetcher, [k.upper() for k in args.leagues], slate_cfg)
            print_slate(slate, not args.live)
            if args.alert:
                return send_alerts(slate.value_bets, args.bankroll, "Jornada " + ", ".join(slate.results),
                                   args.alert_min_ev)
            return 0
        config = PipelineConfig(
            league_key=league.key, seasons=(args.season - 1, args.season), bankroll=args.bankroll,
            use_ml=not args.no_ml, parlay=ParlayConfig(allow_same_match=args.same_match),
            model_path=args.model_path, max_singles=args.max_singles,
            extra_leagues=tuple(args.extra_leagues) if args.extra_leagues is not None
            else (("EPL",) if league.is_international else ()),
        )
        result = AnalysisPipeline(fetcher, config).run()
    except ConfigurationError as exc:
        print(f"Error de configuración: {exc}", file=sys.stderr)
        return 2
    except AuthenticationError as exc:
        print(f"Error de autenticación: {exc}", file=sys.stderr)
        return 3
    except DataFetchError as exc:
        print(f"Error al obtener datos: {exc}", file=sys.stderr)
        return 4
    except InsufficientDataError as exc:
        print(f"Datos insuficientes: {exc}", file=sys.stderr)
        return 5
    print_report(result, simulated=not args.live)
    if args.alert:
        code = send_alerts(result.value_bets, result.bankroll, get_league(result.league).name, args.alert_min_ev)
        if code:
            return code
    if args.risk:
        print_risk(result, args.risk_rounds)
    if args.report:
        try:
            print(f"\nInforme guardado en {save_report(result, args.report, simulated=not args.live)}")
        except (OSError, ValueError) as exc:
            print(f"No se pudo guardar el informe: {exc}", file=sys.stderr)
            return 7
    if args.snapshot:
        try:
            with OddsHistory(args.snapshot) as hist:
                n = hist.save_snapshot(result.odds)
                print(f"\nSnapshot de momios guardado en {args.snapshot} ({n} filas).")
                if args.movement:
                    section("MOVIMIENTO DE LÍNEA (mejor precio; negativo = el momio se acortó)")
                    show(hist.line_movement(list(result.predictions)).head(20).round(3),
                         "(se necesitan al menos dos capturas)")
        except OddsHistoryError as exc:
            print(f"Error en el histórico de momios: {exc}", file=sys.stderr)
            return 8
    if args.track:
        try:
            with BetTracker(args.track) as tracker:
                new = tracker.record_analysis(result.value_bets, result.parlays, result.bankroll)
                print(f"\nRegistro {args.track}: {new} apuesta(s) nueva(s) guardada(s).")
        except TrackerError as exc:
            print(f"Error en el registro de apuestas: {exc}", file=sys.stderr)
            return 6
    return 0


def run_weekly(args: argparse.Namespace) -> int:
    """Ciclo semanal automático (pensado para una tarea programada).

    1. Liquida las apuestas pendientes de los dos registros (sugerencias y
       reales) con los resultados de cada liga (gratis, football-data.co.uk).
    2. Analiza las ligas como una jornada multiliga (tope de exposición global).
    3. Registra las sugerencias en el registro de seguimiento.
    4. Envía por Telegram el resumen y los picks nuevos con el momio mínimo en tu casa.

    Args:
        args: Argumentos (``--leagues``, ``--live``, ``--track``, ``--real-track``...).

    Returns:
        Código de salida (0 = bien, aunque alguna liga falle).
    """
    leagues = [k.upper() for k in args.leagues] if args.leagues else selectable_leagues()
    track = args.track or "bets.sqlite"
    my_book = os.environ.get("MI_CASA", "Caliente")
    min_edge = ValueBetConfig().min_edge
    section(f"CICLO SEMANAL · {', '.join(leagues)}" + ("" if args.live else " [DATOS SIMULADOS]"))
    settled = 0
    errors: list[str] = []
    for key in leagues:  # 1. liquidar (sin gastar créditos)
        try:
            season = args.season if args.season and not args.live else season_for(key)
            results = make_fetcher(args.live, key, (season - 1, season), seed=args.seed,
                                   played_rounds=args.played_rounds).history_provider.fetch_matches(
                                       get_league(key), season)
            for path in (track, args.real_track):
                with BetTracker(path) as tracker:
                    settled += tracker.settle(results)
        except (DataFetchError, TrackerError, ConfigurationError) as exc:
            errors.append(f"{key}: no se pudo liquidar ({exc})")
    print(f"Liquidadas: {settled}")
    season = args.season if args.season else season_for(leagues[0])
    try:  # 2. analizar
        fetcher = make_fetcher(args.live, leagues[0], (season - 1, season), os.environ.get("ODDS_API_KEY", ""),
                               os.environ.get("API_FOOTBALL_KEY", ""), seed=args.seed, played_rounds=args.played_rounds)
        slate = analyze_slate(fetcher, leagues, PipelineConfig(seasons=(season - 1, season), bankroll=args.bankroll,
                                                               use_ml=not args.no_ml, max_singles=args.max_singles))
        errors += [f"{k}: {v}" for k, v in slate.errors.items()]
        print_slate(slate, not args.live)
    except (DataFetchError, AuthenticationError) as exc:
        print(f"Análisis no disponible: {exc}", file=sys.stderr)
        slate = None
        errors.append(f"análisis: {exc}")
    new_picks: list[Pick] = []
    if slate is not None:  # 3. registrar sugerencias
        with BetTracker(track) as tracker:
            n = tracker.record_analysis(slate.value_bets, slate.parlays, args.bankroll)
        print(f"Sugerencias registradas en {track}: {n}")
        new_picks = select_alerts(slate.value_bets, min_edge)
    with BetTracker(args.real_track) as real_t, BetTracker(track) as paper_t:
        real, paper = summarize(real_t.bets()), summarize(paper_t.bets())
    notifier = TelegramNotifier.from_env()  # 4. avisar
    if notifier is None:
        print("Telegram no configurado (TELEGRAM_BOT_TOKEN): el resumen solo se muestra aquí.")
        return 0
    try:
        notifier.send(format_weekly(new_picks, args.bankroll, my_book, min_edge, settled, real, paper, errors))
        mark_sent(new_picks)
        print(f"Telegram: resumen enviado ({len(new_picks)} pick(s) nuevos).")
    except NotifierError as exc:
        print(f"Telegram: {exc}", file=sys.stderr)
        return 9
    return 0


def run_tracker(args: argparse.Namespace) -> int:
    """Liquida las apuestas pendientes con los resultados disponibles y muestra el rendimiento.

    Args:
        args: Argumentos de línea de comandos.

    Returns:
        Código de salida.
    """
    try:
        league = get_league(args.league)
        fetcher = build_fetcher(args)
        results = fetcher.history_provider.fetch_matches(league, args.season)
        with BetTracker(args.track) as tracker:
            if args.snapshot:
                with OddsHistory(args.snapshot) as hist:
                    print(f"Cierre propio aplicado a {tracker.update_closing(hist.closing_odds())} apuesta(s)")
            settled = tracker.settle(results)
            section(f"REGISTRO DE APUESTAS — {args.track}")
            print(f"Liquidadas ahora: {settled}")
            for k, v in tracker.summary(initial_bankroll=args.bankroll).items():
                print(f"  {k:<20} {v:>12,.4f}")
            show(tracker.by_market().round(3), "(sin apuestas liquidadas)")
    except (ConfigurationError, DataFetchError, TrackerError, OddsHistoryError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    sys.exit(main())
