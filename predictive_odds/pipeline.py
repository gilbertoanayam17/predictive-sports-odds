"""Orquestación de extremo a extremo con una API pensada para la UI.

:class:`AnalysisPipeline` expone métodos puros que devuelven dataclasses y
DataFrames; una app de Streamlit/Dash solo tiene que llamar a
``pipeline.run()`` y pintar ``result.value_bets_frame()``, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import ProbabilityCalibrator
from .config import get_league
from .data_fetcher import DataFetcher, DataFetchError
from .domain import STATUS_FINISHED, MatchPrediction, Selection
from .injuries import InjuryAdjuster, InjuryImpactConfig
from .model_store import BundleMetadata, ModelBundle, ModelStoreError, load_bundle, save_bundle
from .parlay_builder import (
    Parlay,
    ParlayBuilder,
    ParlayConfig,
    Pick,
    ValueBetConfig,
    ValueBetFinder,
    apply_exposure_cap,
    apply_match_cap,
)
from .statistical_models import (
    EnsemblePredictor,
    FeatureEngineer,
    HeuristicAnalyzer,
    InsufficientDataError,
    MatchReport,
    MLOutcomeModel,
    PoissonModel,
    PoissonModelConfig,
    TeamAdjustment,
    evaluate_predictions,
    infer_team_leagues,
    NEUTRAL_ADJUSTMENT,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuración del análisis.

    Attributes:
        league_key: Liga a analizar.
        seasons: Temporadas de histórico (la última es la actual).
        bankroll: Bankroll en unidades monetarias.
        use_ml: Si se entrena el modelo de ML.
        ml_weight: Peso del ML en el ensamble.
        max_total_exposure: Exposición máxima simultánea del bankroll.
        holdout_fraction: Fracción final del histórico usada para evaluar.
        poisson: Configuración de Poisson.
        value: Configuración de valor.
        parlay: Configuración de parlays.
        model_path: Si se indica, se reutiliza el bundle guardado mientras no
            esté desactualizado y se guarda tras cada reentrenamiento.
        max_model_age_days: Antigüedad máxima del bundle respecto al último partido.
        extra_leagues: Ligas domésticas adicionales para el modelo de Poisson
            multiliga (útil en competiciones internacionales como la UCL).
        max_match_exposure: Exposición máxima de las sencillas de un mismo partido (``None`` = sin tope).
        max_singles: Máximo de sencillas sugeridas (las de mayor EV ajustado); ``None`` = sin límite.
        use_injuries: Si se ajustan las λ por lesiones/bajas reportadas.
        injury: Parámetros de la heurística de lesiones.
        calibrator: Calibrador opcional (ajustado, p. ej., sobre un backtest).
            Desactivado por defecto: en la validación simulada no mejoró el Brier.
    """

    league_key: str = "EPL"
    seasons: tuple[int, ...] = (2024, 2025)
    bankroll: float = 1000.0
    use_ml: bool = True
    ml_weight: float = 0.35
    max_total_exposure: float = 0.10
    holdout_fraction: float = 0.2
    poisson: PoissonModelConfig = field(default_factory=PoissonModelConfig)
    value: ValueBetConfig = field(default_factory=ValueBetConfig)
    parlay: ParlayConfig = field(default_factory=ParlayConfig)
    model_path: str | None = None
    max_model_age_days: int = 7
    calibrator: ProbabilityCalibrator | None = None
    use_injuries: bool = True
    extra_leagues: tuple[str, ...] = ()
    max_singles: int | None = None
    max_match_exposure: float | None = None
    injury: InjuryImpactConfig = field(default_factory=InjuryImpactConfig)


@dataclass
class AnalysisResult:
    """Resultado completo de un análisis.

    Attributes:
        league: Clave de la liga.
        bankroll: Bankroll usado para calcular stakes.
        standings: Tabla de posiciones.
        fixtures: Próximos partidos analizados.
        predictions: Predicciones por ``match_id``.
        reports: Informes heurísticos por ``match_id``.
        all_picks: Todas las selecciones con momio evaluadas.
        value_bets: Apuestas sencillas de valor (stakes tras tope de exposición).
        parlays: Parlays sugeridos (stakes tras tope de exposición).
        evaluation: Métricas de validación fuera de muestra por modelo.
        team_ratings: Ratings del modelo de Poisson.
        adjustments: Ajustes por lesiones aplicados (equipo → ajuste).
        odds: Momios usados (enlazados a los partidos), p. ej. para guardar snapshots.
    """

    league: str
    bankroll: float
    standings: pd.DataFrame
    fixtures: pd.DataFrame
    predictions: dict[str, MatchPrediction]
    reports: dict[str, MatchReport]
    all_picks: list[Pick]
    value_bets: list[Pick]
    parlays: list[Parlay]
    evaluation: dict[str, dict[str, float]]
    team_ratings: pd.DataFrame
    adjustments: dict[str, TeamAdjustment] = field(default_factory=dict)
    odds: pd.DataFrame = field(default_factory=pd.DataFrame)

    def predictions_frame(self) -> pd.DataFrame:
        """Tabla de predicciones por partido."""
        return pd.DataFrame([p.summary() for p in self.predictions.values()])

    def value_bets_frame(self) -> pd.DataFrame:
        """Tabla de apuestas de valor."""
        return pd.DataFrame([p.to_dict(self.bankroll) for p in self.value_bets])

    def parlays_frame(self) -> pd.DataFrame:
        """Tabla de parlays sugeridos."""
        return pd.DataFrame([p.to_dict(self.bankroll) for p in self.parlays])

    def evaluation_frame(self) -> pd.DataFrame:
        """Tabla de métricas de validación."""
        return pd.DataFrame(self.evaluation).T

    @property
    def total_stake(self) -> float:
        """Stake total sugerido (sencillas + parlays) en unidades monetarias."""
        frac = sum(p.stake_fraction for p in self.value_bets) + sum(p.stake_fraction for p in self.parlays)
        return frac * self.bankroll


class AnalysisPipeline:
    """Pipeline: datos → modelos → predicciones → valor → parlays → staking."""

    def __init__(self, fetcher: DataFetcher, config: PipelineConfig | None = None) -> None:
        """Crea el pipeline.

        Args:
            fetcher: Fachada de datos.
            config: Configuración.
        """
        self.fetcher = fetcher
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------ models
    def fit_models(self, history: pd.DataFrame) -> tuple[EnsemblePredictor, FeatureEngineer, pd.DataFrame]:
        """Entrena Poisson, features y (opcionalmente) ML sobre el histórico.

        Args:
            history: Partidos finalizados.

        Returns:
            ``(ensamble, feature_engineer, features_históricas)``.
        """
        poisson_model = self.fit_poisson(history)
        fe = FeatureEngineer()
        feats = fe.fit_transform(history)
        ml: MLOutcomeModel | None = None
        if self.config.use_ml:
            try:
                ml = MLOutcomeModel().fit(feats)
            except InsufficientDataError as exc:
                logger.warning("ML desactivado: %s", exc)
        return EnsemblePredictor(poisson_model, ml, fe, self.config.ml_weight), fe, feats

    def fit_poisson(self, history: pd.DataFrame) -> PoissonModel:
        """Entrena Poisson; en modo multiliga añade las ligas de ``extra_leagues``.

        Las ligas adicionales que no se puedan descargar se omiten con un aviso.

        Args:
            history: Partidos de la liga principal.

        Returns:
            Modelo de Poisson entrenado.
        """
        c = self.config
        if not c.extra_leagues:
            return PoissonModel(c.poisson).fit(history)
        frames = [history]
        for key in c.extra_leagues:
            try:
                frames.append(self.fetcher.get_history(key, c.seasons))
            except DataFetchError as exc:
                logger.warning("Liga adicional %s omitida: %s", key, exc)
        combined = pd.concat(frames, ignore_index=True)
        intl = [k for k in combined["league"].dropna().unique() if get_league(str(k)).is_international]
        team_leagues = infer_team_leagues(combined, intl)
        logger.info("Poisson multiliga: %d partidos de %d competiciones", len(combined), len(frames))
        return PoissonModel(c.poisson).fit(combined, reference_date=history["date"].max(), team_leagues=team_leagues)

    def get_models(self, history: pd.DataFrame) -> EnsemblePredictor:
        """Devuelve modelos listos: carga el bundle vigente o reentrena (y guarda).

        Args:
            history: Partidos finalizados.

        Returns:
            Ensamble entrenado.
        """
        c = self.config
        latest = history["date"].max()
        if c.model_path and Path(c.model_path).exists():
            try:
                bundle = load_bundle(c.model_path, expected_league=c.league_key)
                if not bundle.is_stale(latest, c.max_model_age_days):
                    logger.info("Usando modelos guardados (%s)", c.model_path)
                    return bundle.ensemble
                logger.info("Bundle desactualizado; se reentrena")
            except ModelStoreError as exc:
                logger.warning("No se pudo usar el bundle guardado: %s; se reentrena", exc)
        ensemble, fe, _feats = self.fit_models(history)
        if c.model_path:
            meta = BundleMetadata(league=c.league_key, trained_until=str(latest), n_matches=len(history))
            try:
                save_bundle(ModelBundle(ensemble, fe, meta, c.calibrator), c.model_path)
            except ModelStoreError as exc:
                logger.warning("%s", exc)
        return ensemble

    def evaluate_models(self, history: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Validación fuera de muestra (cronológica) del mercado 1X2.

        Entrena con el primer ``1 − holdout`` del histórico y evalúa en el
        resto. Incluye una referencia ingenua (frecuencias históricas).

        Args:
            history: Partidos finalizados.

        Returns:
            Métricas por modelo (``log_loss``, ``brier``, ``accuracy``, ``n``).
        """
        hist = history[history["status"] == STATUS_FINISHED].sort_values("date").reset_index(drop=True)
        cut = int(len(hist) * (1 - self.config.holdout_fraction))
        train, test = hist.iloc[:cut], hist.iloc[cut:]
        if len(test) < 20:
            return {}
        outcomes = np.where(test["home_goals"] > test["away_goals"], 0,
                            np.where(test["home_goals"] == test["away_goals"], 1, 2))
        result: dict[str, dict[str, float]] = {}
        freq = np.bincount(np.where(train["home_goals"] > train["away_goals"], 0,
                                    np.where(train["home_goals"] == train["away_goals"], 1, 2)), minlength=3)
        result["baseline_frecuencias"] = evaluate_predictions(np.tile(freq / freq.sum(), (len(test), 1)), outcomes)

        poisson_model = PoissonModel(self.config.poisson).fit(train)
        sels = [Selection.home(), Selection.draw(), Selection.away()]
        p_pois = np.array([[poisson_model.predict(h, a).matrix_probability(s) for s in sels]
                           for h, a in zip(test["home_team"], test["away_team"])])
        result["poisson_dixon_coles"] = evaluate_predictions(p_pois, outcomes)

        if self.config.use_ml:
            feats = FeatureEngineer().fit_transform(hist).set_index("match_id")
            try:
                ml = MLOutcomeModel().fit(feats.loc[train["match_id"]].reset_index())
                p_ml = np.array([[ml.predict_proba(feats.loc[[mid]])[s] for s in sels] for mid in test["match_id"]])
                result["ml_logistica"] = evaluate_predictions(p_ml, outcomes)
                w = self.config.ml_weight
                p_ens = (1 - w) * p_pois + w * p_ml
                result["ensamble"] = evaluate_predictions(p_ens / p_ens.sum(axis=1, keepdims=True), outcomes)
            except InsufficientDataError as exc:
                logger.warning("Evaluación ML omitida: %s", exc)
        return result

    # --------------------------------------------------------------------- run
    def predict_fixtures(self, fixtures: pd.DataFrame, ensemble: EnsemblePredictor,
                         adjustments: dict[str, TeamAdjustment] | None = None) -> dict[str, MatchPrediction]:
        """Predice cada partido del calendario.

        Args:
            fixtures: Partidos programados.
            ensemble: Ensamble entrenado.
            adjustments: Ajustes de contexto por equipo (lesiones).

        Returns:
            Predicciones por ``match_id``.
        """
        adj = adjustments or {}
        preds: dict[str, MatchPrediction] = {}
        for _, f in fixtures.iterrows():
            season = int(f["season"]) if pd.notna(f["season"]) else None
            preds[str(f["match_id"])] = ensemble.predict(
                f["home_team"], f["away_team"], str(f["match_id"]), str(f["league"]), f["date"], season,
                home_adj=adj.get(f["home_team"], NEUTRAL_ADJUSTMENT),
                away_adj=adj.get(f["away_team"], NEUTRAL_ADJUSTMENT))
        return preds

    def injury_adjustments(self, known_teams: list[str]) -> dict[str, TeamAdjustment]:
        """Descarga las bajas y las convierte en ajustes (vacío si no hay datos).

        Args:
            known_teams: Equipos del histórico (para alinear nombres).

        Returns:
            Ajustes por equipo.
        """
        c = self.config
        if not c.use_injuries:
            return {}
        try:
            injuries = self.fetcher.get_injuries(c.league_key, max(c.seasons))
        except DataFetchError as exc:
            logger.warning("No se pudieron obtener lesiones: %s", exc)
            return {}
        return InjuryAdjuster(c.injury, self.fetcher.normalizer).adjustments(injuries, known_teams)

    def run(self, evaluate: bool = True) -> AnalysisResult:
        """Ejecuta el análisis completo.

        Args:
            evaluate: Si se ejecuta la validación fuera de muestra.

        Returns:
            :class:`AnalysisResult`.

        Raises:
            DataFetchError: Si no se pueden obtener los datos necesarios.
        """
        c = self.config
        history = self.fetcher.get_history(c.league_key, c.seasons)
        logger.info("Histórico: %d partidos", len(history))
        try:
            fixtures = self.fetcher.get_fixtures(c.league_key, max(c.seasons))
        except DataFetchError as exc:
            logger.warning("Sin calendario del proveedor (%s); se usará el de los momios", exc)
            fixtures = pd.DataFrame()
        odds = self.fetcher.get_odds(c.league_key) if self.fetcher.odds_provider else pd.DataFrame()
        known = pd.unique(history[["home_team", "away_team"]].values.ravel())
        if fixtures.empty and not odds.empty:
            fixtures = self.fetcher.fixtures_from_odds(odds, known)
        if fixtures.empty:
            raise DataFetchError("No hay partidos próximos que analizar")
        if not odds.empty:
            odds = self.fetcher.link_odds_to_fixtures(odds, fixtures)

        ensemble = self.get_models(history)
        adjustments = self.injury_adjustments([str(t) for t in known])
        predictions = self.predict_fixtures(fixtures, ensemble, adjustments)
        if c.calibrator is not None:
            predictions = {mid: c.calibrator.calibrate_prediction(p) for mid, p in predictions.items()}

        analyzer = HeuristicAnalyzer(history)
        reports = {mid: analyzer.match_report(p.home_team, p.away_team) for mid, p in predictions.items()}

        finder = ValueBetFinder(c.value)
        all_picks = finder.evaluate(predictions, odds)
        value_bets = sorted([p for p in all_picks if finder.is_value(p)], key=lambda p: p.adjusted_edge, reverse=True)
        value_bets = value_bets[:c.max_singles] if c.max_singles is not None else value_bets
        value_bets = apply_match_cap(value_bets, c.max_match_exposure)
        parlays = ParlayBuilder(c.parlay).build(all_picks, predictions)
        value_bets, parlays, _ = apply_exposure_cap(value_bets, parlays, c.max_total_exposure)

        ratings = ensemble.poisson.team_ratings()
        return AnalysisResult(
            league=c.league_key, bankroll=c.bankroll, standings=analyzer.standings(), fixtures=fixtures,
            predictions=predictions, reports=reports, all_picks=all_picks, value_bets=value_bets,
            parlays=parlays, evaluation=self.evaluate_models(history) if evaluate else {},
            team_ratings=ratings[ratings["team"].isin(set(known))].reset_index(drop=True), adjustments=adjustments,
            odds=odds,
        )


@dataclass
class SlateResult:
    """Resultado de analizar varias ligas como una sola jornada.

    Attributes:
        bankroll: Bankroll común.
        results: Resultado completo por liga (las que se pudieron analizar).
        errors: Ligas que fallaron y el motivo.
        value_bets: Sencillas de valor de todas las ligas (tope de exposición global).
        parlays: Parlays construidos con picks de cualquier liga.
    """

    bankroll: float
    results: dict[str, AnalysisResult]
    errors: dict[str, str]
    value_bets: list[Pick]
    parlays: list[Parlay]

    @property
    def total_stake(self) -> float:
        """Stake total sugerido en unidades monetarias."""
        frac = sum(p.stake_fraction for p in self.value_bets) + sum(p.stake_fraction for p in self.parlays)
        return frac * self.bankroll

    def value_bets_frame(self) -> pd.DataFrame:
        """Tabla de sencillas con la liga de cada pick."""
        return pd.DataFrame([{"liga": p.league, **p.to_dict(self.bankroll)} for p in self.value_bets])

    def parlays_frame(self) -> pd.DataFrame:
        """Tabla de parlays con las ligas implicadas."""
        return pd.DataFrame([{"ligas": "+".join(sorted({l.league for l in p.legs})), **p.to_dict(self.bankroll)}
                             for p in self.parlays])


def analyze_slate(fetcher: DataFetcher, league_keys: list[str], base: PipelineConfig | None = None,
                  evaluate: bool = False) -> SlateResult:
    """Analiza varias ligas y combina sus picks en una sola jornada.

    Cada liga se analiza con :class:`AnalysisPipeline` (una liga que falla se
    registra en ``errors`` y no detiene al resto). Después se vuelven a
    seleccionar las sencillas de valor de todas las ligas, se construyen
    parlays que pueden cruzar ligas y se aplica el tope de exposición GLOBAL.

    Args:
        fetcher: Fachada de datos.
        league_keys: Ligas a analizar.
        base: Configuración común (su ``league_key`` se ignora).
        evaluate: Si se ejecuta la validación fuera de muestra por liga.

    Returns:
        :class:`SlateResult`.

    Raises:
        DataFetchError: Si ninguna liga pudo analizarse.
    """
    base = base or PipelineConfig()
    results: dict[str, AnalysisResult] = {}
    errors: dict[str, str] = {}
    for key in league_keys:
        try:
            cfg = replace(base, league_key=key,
                          extra_leagues=base.extra_leagues or (("EPL",) if get_league(key).is_international else ()))
            results[key] = AnalysisPipeline(fetcher, cfg).run(evaluate=evaluate)
        except (DataFetchError, InsufficientDataError) as exc:
            logger.warning("Liga %s omitida: %s", key, exc)
            errors[key] = str(exc)
    if not results:
        raise DataFetchError(f"Ninguna liga pudo analizarse: {errors}")
    all_picks = [p for r in results.values() for p in r.all_picks]
    predictions = {mid: pred for r in results.values() for mid, pred in r.predictions.items()}
    finder = ValueBetFinder(base.value)
    singles = sorted([p for p in all_picks if finder.is_value(p)], key=lambda p: p.adjusted_edge, reverse=True)
    singles = singles[:base.max_singles] if base.max_singles is not None else singles
    singles = apply_match_cap(singles, base.max_match_exposure)
    parlays = ParlayBuilder(base.parlay).build(all_picks, predictions)
    singles, parlays, _ = apply_exposure_cap(singles, parlays, base.max_total_exposure)
    return SlateResult(base.bankroll, results, errors, singles, parlays)
