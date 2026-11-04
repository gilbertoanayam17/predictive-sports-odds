"""Persistencia de modelos entrenados (guardar y cargar sin reentrenar).

Un "bundle" contiene el ensamble (Poisson + ML), el generador de features con
su estado (Elo, historiales) y, opcionalmente, un calibrador, junto con los
metadatos necesarios para detectar artefactos obsoletos o incompatibles.

Seguridad: los archivos se serializan con ``joblib`` (pickle). **Solo deben
cargarse bundles generados por uno mismo**, nunca archivos de terceros.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from . import __version__
from .calibration import ProbabilityCalibrator
from .statistical_models import EnsemblePredictor, FeatureEngineer

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = 1


class ModelStoreError(Exception):
    """Error al guardar o cargar un bundle de modelos."""


@dataclass
class BundleMetadata:
    """Metadatos de un bundle.

    Attributes:
        league: Liga para la que se entrenó.
        trained_until: Fecha del último partido de entrenamiento (ISO).
        n_matches: Partidos de entrenamiento.
        package_version: Versión del paquete que lo generó.
        bundle_format: Versión del formato del bundle.
        extra: Información adicional libre (p. ej. métricas de validación).
    """

    league: str
    trained_until: str
    n_matches: int
    package_version: str = __version__
    bundle_format: int = BUNDLE_FORMAT
    extra: dict[str, float | str] = field(default_factory=dict)


@dataclass
class ModelBundle:
    """Modelos entrenados listos para predecir.

    Attributes:
        ensemble: Ensamble Poisson + ML.
        feature_engineer: Generador de features con el estado al final del entrenamiento.
        metadata: Metadatos.
        calibrator: Calibrador opcional.
    """

    ensemble: EnsemblePredictor
    feature_engineer: FeatureEngineer
    metadata: BundleMetadata
    calibrator: ProbabilityCalibrator | None = None

    def is_stale(self, latest_match_date: pd.Timestamp, max_age_days: int = 7) -> bool:
        """Indica si el bundle está desactualizado respecto a los datos disponibles.

        Args:
            latest_match_date: Fecha del último partido disponible.
            max_age_days: Antigüedad máxima tolerada.

        Returns:
            ``True`` si hay datos más de ``max_age_days`` días posteriores al entrenamiento.
        """
        trained = pd.Timestamp(self.metadata.trained_until)
        if trained.tzinfo is None and latest_match_date.tzinfo is not None:
            trained = trained.tz_localize("UTC")
        return bool((latest_match_date - trained).days > max_age_days)


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    """Guarda un bundle en disco (``<path>`` + ``<path>.json`` con metadatos legibles).

    Args:
        bundle: Bundle a guardar.
        path: Ruta del archivo (se recomienda extensión ``.joblib``).

    Returns:
        Ruta escrita.

    Raises:
        ModelStoreError: Si no se puede escribir.
    """
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        joblib.dump(bundle, tmp, compress=3)
        tmp.replace(target)  # escritura atómica
        target.with_suffix(target.suffix + ".json").write_text(
            json.dumps(asdict(bundle.metadata), indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise ModelStoreError(f"No se pudo guardar el bundle en {target}: {exc}") from exc
    logger.info("Bundle guardado en %s", target)
    return target


def load_bundle(path: str | Path, expected_league: str | None = None) -> ModelBundle:
    """Carga un bundle validando formato, tipo y liga.

    Args:
        path: Ruta del archivo.
        expected_league: Si se indica, exige que coincida con la liga del bundle.

    Returns:
        El bundle cargado.

    Raises:
        ModelStoreError: Si el archivo no existe, está corrupto o es incompatible.
    """
    source = Path(path)
    if not source.exists():
        raise ModelStoreError(f"No existe el bundle {source}")
    try:
        obj = joblib.load(source)
    except Exception as exc:  # joblib/pickle pueden lanzar casi cualquier excepción
        raise ModelStoreError(f"Bundle corrupto o ilegible ({source}): {exc}") from exc
    if not isinstance(obj, ModelBundle):
        raise ModelStoreError(f"{source} no contiene un ModelBundle (tipo {type(obj).__name__})")
    if obj.metadata.bundle_format != BUNDLE_FORMAT:
        raise ModelStoreError(f"Formato de bundle {obj.metadata.bundle_format} incompatible (se espera {BUNDLE_FORMAT})")
    if expected_league is not None and obj.metadata.league != expected_league:
        raise ModelStoreError(f"El bundle es de {obj.metadata.league}, no de {expected_league}")
    if obj.metadata.package_version != __version__:
        logger.warning("Bundle generado con la versión %s (actual %s)", obj.metadata.package_version, __version__)
    return obj


__all__ = ["BundleMetadata", "ModelBundle", "ModelStoreError", "load_bundle", "save_bundle"]
