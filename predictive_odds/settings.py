"""Carga de configuración local desde un archivo ``.env`` (sin dependencias externas).

El archivo ``.env`` vive en la raíz del proyecto, está en ``.gitignore`` y
nunca se sube al repositorio. Formato (una variable por línea)::

    ODDS_API_KEY=tu_clave
    API_FOOTBALL_KEY=          # opcional
    # comentario

Las variables ya definidas en el sistema tienen prioridad: el ``.env`` solo
rellena las que faltan. Con ``PREDICTIVE_ODDS_NO_DOTENV=1`` no se carga (los
tests lo activan para no gastar créditos con claves reales).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

#: Raíz del proyecto (carpeta que contiene el paquete ``predictive_odds``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_env(text: str) -> dict[str, str]:
    """Interpreta el contenido de un ``.env``.

    Admite comentarios (``#``), líneas en blanco, el prefijo ``export``,
    comillas simples/dobles y comentarios al final de valores sin comillas.

    Args:
        text: Contenido del archivo.

    Returns:
        Mapa ``variable -> valor``.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key.replace("_", "").isalnum():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key] = value
    return values


def find_env_file(start: Path | None = None) -> Path | None:
    """Busca ``.env`` en la carpeta actual y luego en la raíz del proyecto.

    Args:
        start: Carpeta desde la que buscar (por defecto, la actual).

    Returns:
        Ruta del ``.env`` encontrado o ``None``.
    """
    for folder in (start or Path.cwd(), PROJECT_ROOT):
        candidate = Path(folder) / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env(path: str | Path | None = None, override: bool = False) -> list[str]:
    """Carga las variables del ``.env`` en ``os.environ``.

    Args:
        path: Ruta explícita (por defecto, :func:`find_env_file`).
        override: Si ``True``, el ``.env`` sobrescribe variables ya definidas.

    Returns:
        Nombres de las variables cargadas (sin sus valores, para no exponer claves).
    """
    if os.environ.get("PREDICTIVE_ODDS_NO_DOTENV") == "1":  # p. ej. en los tests: nunca usar claves reales
        return []
    env_path = Path(path) if path is not None else find_env_file()
    if env_path is None or not env_path.is_file():
        return []
    try:
        raw = env_path.read_bytes()
        # PowerShell 5 ("echo ... > .env") guarda en UTF-16 con BOM; el resto, en UTF-8.
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
        values = parse_env(text)
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("No se pudo leer %s: %s", env_path, exc)
        return []
    loaded = []
    for key, value in values.items():
        if value and (override or not os.environ.get(key)):
            os.environ[key] = value
            loaded.append(key)
    if loaded:
        logger.info("Variables cargadas desde %s: %s", env_path, ", ".join(loaded))
    return loaded


__all__ = ["PROJECT_ROOT", "find_env_file", "load_env", "parse_env"]
