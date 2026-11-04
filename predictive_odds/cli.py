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

from predictive_odds.config import ConfigurationError, get_league
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
from predictive_odds.bet_tracker import BetTracker, TrackerError
from predictive_odds.backtester import BacktestConfig, Backtester, BacktestResult
from predictive_odds.parlay_builder import ParlayConfig
from predictive_odds.xg_sources import UnderstatProvider, XgEnrichedProvider
from predictive_odds.statistical_models import InsufficientDataError
from predictive_odds.risk import RiskConfig, risk_table
from predictive_odds.tuning import tune_ml_weight, tune_poisson
from predictive_odds.pipeline import AnalysisPipeline, AnalysisResult, PipelineConfig

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
    """Construye la fachada de datos según el modo (simulado o en vivo).

    Args:
        args: Argumentos de línea de comandos.

    Returns:
        ``DataFetcher`` configurado.

    Raises:
        AuthenticationError: Si faltan API keys en modo en vivo.
    """
    if not args.live:
        sim = SimulatedDataProvider(SimulationConfig(seed=args.seed, seasons=(args.season - 1, args.season),
                                                     played_rounds_current=args.played_rounds))
        return DataFetcher(history_provider=sim, odds_provider=sim)
    league = get_league(args.league)
    odds_provider = TheOddsApiProvider(os.environ.get("ODDS_API_KEY", ""), include_btts=args.btts)
    af_key = os.environ.get("API_FOOTBALL_KEY", "")
    api_football = ApiFootballProvider(af_key) if af_key else None
    history: ResultsProvider
    if league.football_data_code:
        history = FootballDataCoUkProvider()
        if args.understat:
            history = XgEnrichedProvider(history, UnderstatProvider())
    elif api_football is not None:
        history = api_football
    else:
        raise AuthenticationError(f"{league.name} requiere API_FOOTBALL_KEY para el histórico")
    # Sin API-Football el calendario se deriva de los eventos de The Odds API.
    fixtures = api_football if api_football is not None else _NoFixtures()
    return DataFetcher(history_provider=history, fixtures_provider=fixtures, odds_provider=odds_provider)


class _NoFixtures(ResultsProvider):
    """Proveedor nulo: fuerza a construir el calendario desde los momios."""

    name = "none"

    def fetch_matches(self, league, season):  # type: ignore[no-untyped-def]
        """Siempre falla para que el pipeline use los eventos de momios."""
        raise DataFetchError("Sin proveedor de calendario")


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
    p.add_argument("--league", default="EPL", help="EPL, UCL o LIGA_MX")
    p.add_argument("--season", type=int, default=2025, help="Temporada actual (año de inicio)")
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
    p.add_argument("--risk", action="store_true", help="Añadir simulación Monte Carlo de riesgo del bankroll")
    p.add_argument("--risk-rounds", type=int, default=38, help="Jornadas a simular en --risk")
    p.add_argument("--tune", action="store_true", help="Ajustar hiperparámetros con validación cronológica")
    p.add_argument("--backtest", action="store_true", help="Ejecutar un backtest walk-forward en vez del análisis")
    p.add_argument("--parlays", action="store_true", help="Backtest: incluir también los parlays")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada: ejecuta el flujo completo y devuelve el código de salida."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
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
        if args.backtest:
            print_backtest(run_backtest(args), league.key, simulated=not args.live)
            return 0
        fetcher = build_fetcher(args)
        config = PipelineConfig(
            league_key=league.key, seasons=(args.season - 1, args.season), bankroll=args.bankroll,
            use_ml=not args.no_ml, parlay=ParlayConfig(allow_same_match=args.same_match),
            model_path=args.model_path,
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
    if args.risk:
        print_risk(result, args.risk_rounds)
    if args.track:
        try:
            with BetTracker(args.track) as tracker:
                new = tracker.record_analysis(result.value_bets, result.parlays, result.bankroll)
                print(f"\nRegistro {args.track}: {new} apuesta(s) nueva(s) guardada(s).")
        except TrackerError as exc:
            print(f"Error en el registro de apuestas: {exc}", file=sys.stderr)
            return 6
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
            settled = tracker.settle(results)
            section(f"REGISTRO DE APUESTAS — {args.track}")
            print(f"Liquidadas ahora: {settled}")
            for k, v in tracker.summary(initial_bankroll=args.bankroll).items():
                print(f"  {k:<20} {v:>12,.4f}")
            show(tracker.by_market().round(3), "(sin apuestas liquidadas)")
    except (ConfigurationError, DataFetchError, TrackerError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    sys.exit(main())
