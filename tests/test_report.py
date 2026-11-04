"""Tests de exportación de informes."""

from __future__ import annotations

from pathlib import Path

import pytest

from predictive_odds.data_fetcher import DataFetcher, SimulatedDataProvider
from predictive_odds.pipeline import AnalysisPipeline, AnalysisResult, PipelineConfig
from predictive_odds.report import render_html, render_markdown, save_report


@pytest.fixture(scope="module")
def result() -> AnalysisResult:
    sim = SimulatedDataProvider()
    return AnalysisPipeline(DataFetcher(sim, odds_provider=sim), PipelineConfig(use_ml=False)).run(evaluate=False)


def test_markdown_and_html(result: AnalysisResult, tmp_path: Path) -> None:
    md = render_markdown(result, simulated=True)
    assert md.startswith("# Informe de apuestas — Premier League")
    assert "## Apuestas de valor" in md and "datos simulados" in md
    assert md.count("\n| ") >= len(result.value_bets)
    page = render_html(result)
    assert page.startswith("<!doctype html>") and "prefers-color-scheme: dark" in page
    assert "<script" not in page and "http" not in page.split("<body>")[1]  # autocontenido
    saved = save_report(result, tmp_path / "out" / "r.html").read_text(encoding="utf-8")
    assert saved.startswith("<!doctype html>") and "Apuestas de valor" in saved
    assert save_report(result, tmp_path / "r.md").read_text(encoding="utf-8").startswith("# Informe")
    with pytest.raises(ValueError):
        save_report(result, tmp_path / "r.pdf")


def test_html_escapes_team_names(result: AnalysisResult) -> None:
    pred = next(iter(result.predictions.values()))
    pred.home_team = "<script>alert(1)</script>"
    assert "<script>alert" not in render_html(result)
