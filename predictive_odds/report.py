"""Exportación del análisis a Markdown o HTML autocontenido.

Pensado para leer los picks fuera de la terminal (móvil, correo, chat). El HTML
no carga recursos externos y respeta el modo claro/oscuro del sistema.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import get_league
from .pipeline import AnalysisResult

DISCLAIMER = ("Herramienta analítica: no coloca apuestas. Las probabilidades son estimaciones de un modelo; "
              "apostar implica riesgo de pérdida. Juega con responsabilidad.")


def _md_table(df: pd.DataFrame) -> str:
    """DataFrame → tabla Markdown (sin dependencias externas)."""
    if df.empty:
        return "_(sin resultados)_"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    """Formato compacto de una celda."""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".") if abs(value) < 1000 else f"{value:,.0f}"
    return str(value).replace("|", "/")


def _sections(result: AnalysisResult, simulated: bool) -> list[tuple[str, pd.DataFrame | str]]:
    """Secciones comunes a ambos formatos."""
    preds = result.predictions_frame().drop(columns="match_id")
    for c in ("P(1)", "P(X)", "P(2)", "P(O2.5)", "P(BTTS)"):
        preds[c] = (100 * preds[c]).round(1)
    vb = result.value_bets_frame()
    pf = result.parlays_frame()
    injuries = pd.DataFrame([{"equipo": t, "ataque": round(a.attack, 2), "goles_recibidos": round(a.defense, 2),
                              "motivo": a.reason} for t, a in result.adjustments.items()])
    warnings = [w for p in result.parlays for w in p.warnings]
    summary = (f"Partidos: **{len(result.predictions)}** · value bets: **{len(result.value_bets)}** · "
               f"parlays: **{len(result.parlays)}** · exposición sugerida: **{result.total_stake:,.2f}** "
               f"({100 * result.total_stake / result.bankroll:.1f}% de un bankroll de {result.bankroll:,.0f})"
               + (" · **datos simulados**" if simulated else ""))
    out: list[tuple[str, pd.DataFrame | str]] = [("Resumen", summary)]
    out.append(("Apuestas de valor (sencillas)", vb))
    out.append(("Parlays sugeridos", pf))
    if warnings:
        out.append(("Avisos de parlays", "\n".join(f"- {w}" for w in warnings)))
    out.append(("Predicciones (probabilidades en %)", preds))
    if not injuries.empty:
        out.append(("Ajustes por lesiones", injuries))
    return out


def render_markdown(result: AnalysisResult, simulated: bool = False) -> str:
    """Genera el informe en Markdown.

    Args:
        result: Resultado del análisis.
        simulated: Si los datos son simulados (se indica en el informe).

    Returns:
        Texto Markdown.
    """
    league = get_league(result.league).name
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [f"# Informe de apuestas — {league}", f"_Generado: {now}_", ""]
    for title, content in _sections(result, simulated):
        parts += [f"## {title}", "", content if isinstance(content, str) else _md_table(content), ""]
    parts.append(f"> {DISCLAIMER}")
    return "\n".join(parts) + "\n"


_CSS = """
:root { --bg:#ffffff; --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; --head:#f6f8fa; --accent:#0969da; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --line:#3d444d; --head:#151b23; --accent:#4493f8; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:1.6rem; margin:0 0 4px; } h2 { font-size:1.15rem; margin:28px 0 8px; color:var(--accent); }
.muted { color:var(--muted); font-size:.9rem; }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:8px; }
table { border-collapse:collapse; width:100%; font-size:.9rem; font-variant-numeric:tabular-nums; }
th, td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
th { background:var(--head); font-weight:600; }
tr:last-child td { border-bottom:none; }
blockquote { margin:32px 0 0; padding:8px 12px; border-left:3px solid var(--line); color:var(--muted); }
"""


def _html_table(df: pd.DataFrame) -> str:
    """DataFrame → tabla HTML escapada."""
    if df.empty:
        return '<p class="muted">(sin resultados)</p>'
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(_fmt(v))}</td>" for v in row) + "</tr>"
                   for row in df.itertuples(index=False))
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _md_inline(text: str) -> str:
    """Convierte el poco Markdown usado (negritas y listas) a HTML escapado."""
    lines = []
    for line in html.escape(text).split("\n"):
        parts = line.split("**")
        line = "".join(f"<strong>{p}</strong>" if i % 2 else p for i, p in enumerate(parts))
        lines.append(f"<li>{line[2:]}</li>" if line.startswith("- ") else f"<p>{line}</p>")
    joined = "".join(lines)
    return f"<ul>{joined}</ul>" if joined.startswith("<li>") else joined


def render_html(result: AnalysisResult, simulated: bool = False) -> str:
    """Genera el informe como página HTML autocontenida.

    Args:
        result: Resultado del análisis.
        simulated: Si los datos son simulados.

    Returns:
        Documento HTML.
    """
    league = get_league(result.league).name
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blocks = []
    for title, content in _sections(result, simulated):
        inner = _md_inline(content) if isinstance(content, str) else _html_table(content)
        blocks.append(f"<h2>{html.escape(title)}</h2>{inner}")
    return (
        "<!doctype html><html lang=\"es\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>Informe {html.escape(league)}</title><style>{_CSS}</style></head><body><main>"
        f"<h1>Informe de apuestas — {html.escape(league)}</h1><p class=\"muted\">Generado: {now}</p>"
        + "".join(blocks)
        + f"<blockquote>{html.escape(DISCLAIMER)}</blockquote></main></body></html>\n"
    )


def save_report(result: AnalysisResult, path: str | Path, simulated: bool = False) -> Path:
    """Guarda el informe; el formato se deduce de la extensión (``.md`` o ``.html``).

    Args:
        result: Resultado del análisis.
        path: Ruta de salida.
        simulated: Si los datos son simulados.

    Returns:
        Ruta escrita.

    Raises:
        ValueError: Si la extensión no es ``.md`` ni ``.html``.
        OSError: Si no se puede escribir.
    """
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix not in (".md", ".html", ".htm"):
        raise ValueError("El informe debe tener extensión .md o .html")
    content = render_markdown(result, simulated) if suffix == ".md" else render_html(result, simulated)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


__all__ = ["render_html", "render_markdown", "save_report"]
