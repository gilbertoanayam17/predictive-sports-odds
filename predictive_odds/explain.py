"""Explicaciones breves de cada pick ("¿por qué este pick?").

No añaden información nueva al modelo: resumen en lenguaje llano de dónde sale
el valor y qué dato de fútbol lo respalda, para que el usuario pueda descartar
picks que no tienen sentido (p. ej. por una baja que el modelo no conoce).
"""

from __future__ import annotations

from typing import Mapping

from .domain import Market, MatchPrediction, Outcome
from .parlay_builder import Pick
from .statistical_models import FormSummary, MatchReport, TeamAdjustment

#: Diferencia modelo−mercado (puntos porcentuales) a partir de la cual se avisa.
BIG_GAP = 0.10


def _form(f: FormSummary) -> str:
    """'WWDLW (2.2 pts/p)' o '—' si no hay partidos."""
    return f"{f.results} ({f.ppg:.1f} pts/p)" if f.matches else "—"


def explain_pick(pick: Pick, pred: MatchPrediction | None = None, report: MatchReport | None = None,
                 adjustments: Mapping[str, TeamAdjustment] | None = None,
                 unknown_teams: set[str] | frozenset[str] = frozenset(), max_lines: int = 3) -> list[str]:
    """Hasta ``max_lines`` frases que explican el pick.

    Args:
        pick: Pick a explicar.
        pred: Predicción del partido (goles esperados).
        report: Informe heurístico (forma por sede).
        adjustments: Ajustes por lesiones por equipo.
        unknown_teams: Equipos sin historial (fuerza promedio asumida).
        max_lines: Máximo de frases.

    Returns:
        Frases en español, de la más a la menos importante.
    """
    lines: list[str] = []
    sel = pick.selection
    if pick.market_prob is not None:
        gap = pick.model_prob - pick.market_prob
        lines.append(f"Modelo {pick.model_prob:.0%} frente a mercado {pick.market_prob:.0%} ({100 * gap:+.0f} pp).")
    else:
        lines.append(f"Modelo {pick.model_prob:.0%} frente a {pick.implied_prob:.0%} implícito en el momio.")

    if pred is not None and sel.market in (Market.OVER_UNDER, Market.BTTS):
        total = pred.lambda_home + pred.lambda_away
        lines.append(f"Goles esperados: {pred.lambda_home:.1f} + {pred.lambda_away:.1f} = {total:.1f}.")
    elif report is not None and sel.outcome in (Outcome.HOME, Outcome.AWAY, Outcome.DRAW):
        lines.append(f"Forma: {pick.home_team} en casa {_form(report.home.form_venue)}; "
                     f"{pick.away_team} fuera {_form(report.away.form_venue)}.")
    elif pred is not None:
        lines.append(f"Goles esperados: {pick.home_team} {pred.lambda_home:.1f}, {pick.away_team} {pred.lambda_away:.1f}.")

    warnings: list[str] = []
    missing = [t for t in (pick.home_team, pick.away_team) if t in unknown_teams]
    if missing:
        warnings.append(f"Ojo: sin historial de {', '.join(missing)} (fuerza promedio asumida).")
    if pick.market_prob is not None and abs(pick.model_prob - pick.market_prob) >= BIG_GAP:
        warnings.append("Ojo: diferencia grande con el mercado; revisa alineaciones y bajas antes de apostar.")
    hurt = [f"{t} ({a.reason.split(':')[0]})" for t, a in (adjustments or {}).items()
            if t in (pick.home_team, pick.away_team) and not a.is_neutral]
    if hurt:
        warnings.append(f"Bajas consideradas: {'; '.join(hurt)}.")
    # Los avisos tienen prioridad sobre el segundo dato si no caben todos.
    out = lines[:1] + warnings + lines[1:]
    return out[:max_lines]


__all__ = ["BIG_GAP", "explain_pick"]
