"""Tests de la carga del archivo .env."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from predictive_odds.settings import find_env_file, load_env, parse_env


def test_parse_env_formats() -> None:
    text = """﻿# comentario
ODDS_API_KEY=abc123
export API_FOOTBALL_KEY = "con espacios"
QUOTED='x # no es comentario'
INLINE=valor # comentario
VACIA=
mal formada
"""
    values = parse_env(text)
    assert values == {"ODDS_API_KEY": "abc123", "API_FOOTBALL_KEY": "con espacios",
                      "QUOTED": "x # no es comentario", "INLINE": "valor", "VACIA": ""}


def test_load_env_respects_system_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PREDICTIVE_ODDS_NO_DOTENV")
    env = tmp_path / ".env"
    env.write_text("ODDS_API_KEY=desde_env\nAPI_FOOTBALL_KEY=af\nVACIA=\n", encoding="utf-8")
    monkeypatch.setenv("API_FOOTBALL_KEY", "del_sistema")
    loaded = load_env(env)
    assert loaded == ["ODDS_API_KEY"]  # no expone valores; las vacías se ignoran
    assert os.environ["ODDS_API_KEY"] == "desde_env"
    assert os.environ["API_FOOTBALL_KEY"] == "del_sistema"  # el sistema tiene prioridad
    assert load_env(env, override=True) == ["ODDS_API_KEY", "API_FOOTBALL_KEY"]


def test_disabled_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text("ODDS_API_KEY=x\n", encoding="utf-8")
    assert load_env(env) == []  # PREDICTIVE_ODDS_NO_DOTENV=1 (conftest)
    monkeypatch.delenv("PREDICTIVE_ODDS_NO_DOTENV")
    assert load_env(tmp_path / "no_existe.env") == []
    assert find_env_file(tmp_path) == env


def test_example_file_is_committed_and_env_is_ignored() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / ".env.example").is_file()
    assert ".env" in (root / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_utf16_file_from_powershell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PREDICTIVE_ODDS_NO_DOTENV")
    env = tmp_path / ".env"
    env.write_bytes("ODDS_API_KEY=ps_clave\r\n".encode("utf-16"))  # como "echo ... > .env" en PowerShell 5
    assert load_env(env) == ["ODDS_API_KEY"] and os.environ["ODDS_API_KEY"] == "ps_clave"
