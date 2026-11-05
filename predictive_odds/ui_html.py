"""Componentes visuales de la app (HTML + CSS), sin dependencia de Streamlit.

Identidad "pizarra táctica": fondo verde pizarra casi negro, texto color tiza,
un único acento ámbar de marcador y tipografía de estadio (Barlow Condensed)
para títulos y cifras; IBM Plex para texto y números tabulares.

Colores de datos (validados con el validador de la skill de visualización
sobre la superficie #141c18): local azul ``#3987e5``, empate gris neutro
``#8a928c`` (punto medio de una escala divergente) y visitante naranja
``#e0682f``. Cada segmento lleva su porcentaje escrito: el color nunca es la
única pista. El ámbar de la marca se reserva para la interfaz, nunca para datos.

Todo el texto dinámico se escapa con :func:`html.escape`.
"""

from __future__ import annotations

import html
from typing import Iterable, Sequence

import pandas as pd

from .domain import MatchPrediction, Selection
from .parlay_builder import Parlay, Pick

HOME, DRAW, AWAY = "#3987e5", "#8a928c", "#e0682f"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{--ink:#0f1613;--slate:#141c18;--slate-2:#18221d;--line:#26332c;--chalk:#e9ece6;--chalk-2:#a9b3ac;
--chalk-3:#7c877f;--amber:#e8a93b;--amber-ink:#1a1408;--home:#3987e5;--draw:#8a928c;--away:#e0682f;
--good:#58b26c;--bad:#e06a5f}
html,body,[class*="css"],.stMarkdown,.stText,p,li,label,input,textarea,button{font-family:'IBM Plex Sans',system-ui,sans-serif}
h1,h2,h3,h4{font-family:'Barlow Condensed',sans-serif!important;letter-spacing:.01em;font-weight:600!important}
h2{font-size:1.65rem!important}h3{font-size:1.3rem!important}
[data-testid="stAppViewContainer"]{background:radial-gradient(1200px 500px at 85% -10%,#1b2a22 0,transparent 60%),var(--ink)}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:var(--slate);border-right:1px solid var(--line)}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{font-size:.95rem!important;text-transform:uppercase;
letter-spacing:.12em;color:var(--chalk-2)}
.block-container{padding-top:1.4rem;max-width:1240px}
[data-testid="stMetric"]{background:var(--slate);border:1px solid var(--line);border-radius:6px;padding:.7rem .9rem}
[data-testid="stMetricLabel"] p{font-size:.72rem!important;text-transform:uppercase;letter-spacing:.12em;color:var(--chalk-3)}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] p{font-family:'Barlow Condensed',sans-serif!important;
font-weight:600;font-size:2.1rem}
.stTabs [data-baseweb="tab-list"],.stTabs [role="tablist"]{gap:.2rem;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"],.stTabs [data-testid="stTab"],.stTabs [role="tab"]{padding:.35rem .7rem;color:var(--chalk-2)}
.stTabs [role="tab"] p,.stTabs [data-testid="stTab"] p,.stTabs [data-baseweb="tab"] p{
font-family:'Barlow Condensed',sans-serif!important;font-size:1.05rem;letter-spacing:.06em;text-transform:uppercase}
.stTabs [role="tab"][aria-selected="true"]{color:var(--chalk);box-shadow:inset 0 -3px 0 var(--amber)}
.stTabs [data-baseweb="tab-highlight"]{background:var(--amber);height:3px}
.stButton>button,.stDownloadButton>button{border-radius:4px;font-weight:600;letter-spacing:.02em}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:6px}
code{color:var(--amber)!important;background:transparent!important}
.mono,.odds{font-family:'IBM Plex Mono',ui-monospace,monospace;font-variant-numeric:tabular-nums}
.mast{display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;flex-wrap:wrap;
border-bottom:1px solid var(--line);padding:.2rem 0 .9rem;margin-bottom:1rem}
.mast .brand{display:flex;align-items:center;gap:.7rem}
.mast .word{font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:2.1rem;line-height:1;
letter-spacing:.02em;color:var(--chalk);text-transform:uppercase}
.mast .word span{color:var(--amber)}
.mast .sub{font-size:.78rem;color:var(--chalk-3);letter-spacing:.14em;text-transform:uppercase;margin-top:.25rem}
.mast .meta{text-align:right;font-size:.85rem;color:var(--chalk-2)}
.chip{display:inline-block;font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;padding:.15rem .5rem;
border-radius:3px;border:1px solid var(--line);color:var(--chalk-2);margin-left:.4rem}
.chip.live{border-color:var(--amber);color:var(--amber)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(285px,1fr));gap:.75rem;margin:.4rem 0 1rem}
.card{background:var(--slate);border:1px solid var(--line);border-radius:6px;padding:.8rem .9rem;position:relative}
.card .when{font-size:.7rem;color:var(--chalk-3);letter-spacing:.1em;text-transform:uppercase}
.teams{display:grid;grid-template-columns:1fr auto;row-gap:.15rem;margin:.35rem 0 .55rem}
.team{font-family:'Barlow Condensed',sans-serif;font-size:1.25rem;font-weight:600;color:var(--chalk);
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.xg{font-family:'IBM Plex Mono',monospace;font-size:.8rem;color:var(--chalk-2);align-self:center;text-align:right}
.bar{display:flex;gap:2px;height:22px;border-radius:4px;overflow:hidden}
.bar div{display:flex;align-items:center;justify-content:center;font-family:'IBM Plex Mono',monospace;font-size:.72rem;
color:#0b0f0d;font-weight:500;min-width:0}
.legend{display:flex;justify-content:space-between;font-size:.66rem;color:var(--chalk-3);letter-spacing:.1em;
text-transform:uppercase;margin-top:.25rem}
.foot{display:flex;justify-content:space-between;gap:.5rem;margin-top:.55rem;font-size:.78rem;color:var(--chalk-2)}
.foot b{color:var(--chalk);font-weight:500}
.badge{font-family:'IBM Plex Mono',monospace;font-size:.72rem;padding:.08rem .4rem;border-radius:3px;
background:var(--amber);color:var(--amber-ink);font-weight:500}
.ticket{border-left:3px solid var(--amber)}
.ticket .sel{font-family:'Barlow Condensed',sans-serif;font-size:1.3rem;font-weight:600;color:var(--chalk);margin:.15rem 0}
.ticket .fx{font-size:.82rem;color:var(--chalk-2)}
.ticket .row{display:flex;justify-content:space-between;align-items:baseline;margin-top:.55rem}
.ticket .price{font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:700;color:var(--chalk);line-height:1}
.ticket .price small{font-family:'IBM Plex Sans',sans-serif;font-size:.7rem;color:var(--chalk-3);letter-spacing:.08em;
text-transform:uppercase;margin-left:.35rem;font-weight:400}
.ticket .stake{text-align:right;font-size:.75rem;color:var(--chalk-3);text-transform:uppercase;letter-spacing:.08em}
.ticket .stake b{display:block;font-family:'IBM Plex Mono',monospace;font-size:1rem;color:var(--chalk);
text-transform:none;letter-spacing:0;font-weight:500}
.legs{margin:.3rem 0 0;padding:0;list-style:none}
.legs li{font-size:.82rem;color:var(--chalk-2);padding:.18rem 0;border-bottom:1px dashed var(--line)}
.legs li b{color:var(--chalk);font-weight:500}
.note{font-size:.8rem;color:var(--chalk-3);margin:.2rem 0 .6rem}
.mine{margin-top:.55rem;padding:.35rem .5rem;border:1px dashed var(--amber);border-radius:4px;font-size:.8rem;
color:var(--chalk-2)}.mine b{font-family:'IBM Plex Mono',monospace;color:var(--amber);font-weight:500}
.why{margin:.5rem 0 0!important;padding-left:1rem!important;color:var(--chalk-2)}
.why li{margin:.1rem 0!important;padding:0!important;font-size:.78rem!important;line-height:1.35!important}
.empty{border:1px dashed var(--line);border-radius:6px;padding:1.2rem;color:var(--chalk-2);text-align:center}
@media (max-width:640px){
.mast .word{font-size:1.55rem}.mast .sub{letter-spacing:.06em;font-size:.68rem}.mast .meta{text-align:left}
.mast svg{width:30px;height:30px}.team{font-size:1.1rem}
/* Métricas y columnas en 2×2 en vez de una pantalla por métrica */
[data-testid="stHorizontalBlock"]{flex-direction:row!important;flex-wrap:wrap!important;gap:.5rem!important}
[data-testid="stHorizontalBlock"]>[data-testid="stColumn"],[data-testid="stHorizontalBlock"]>[data-testid="column"]{
width:calc(50% - .25rem)!important;flex:1 1 calc(50% - .25rem)!important;min-width:calc(50% - .25rem)!important}
[data-testid="stMetric"]{padding:.5rem .6rem}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] p{font-size:1.6rem!important}
[data-testid="stMetricLabel"] p{font-size:.62rem!important;letter-spacing:.08em}
.block-container{padding-left:.8rem;padding-right:.8rem}
}
</style>
"""

LOGO = ('<svg width="38" height="38" viewBox="0 0 38 38" aria-hidden="true"><rect x="1" y="1" width="36" height="36" '
        'rx="5" fill="none" stroke="#26332c" stroke-width="2"/><line x1="19" y1="2" x2="19" y2="36" stroke="#26332c" '
        'stroke-width="2"/><circle cx="19" cy="19" r="5.5" fill="none" stroke="#26332c" stroke-width="2"/>'
        '<path d="M8 27 C13 15, 23 11, 30 10" fill="none" stroke="#e8a93b" stroke-width="2.6" stroke-linecap="round"/>'
        '<path d="M25.5 7.5 L30.5 10 L27.5 14.5" fill="none" stroke="#e8a93b" stroke-width="2.6" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>')

e = html.escape


DIAS = ("lun", "mar", "mié", "jue", "vie", "sáb", "dom")
MESES = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")


def fecha(value: object) -> str:
    """Fecha corta en español ("sáb 08 nov · 15:00"); cadena vacía si no hay fecha válida."""
    try:
        ts = pd.Timestamp(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return ""
    if pd.isna(ts):
        return ""
    return f"{DIAS[ts.weekday()]} {ts.day:02d} {MESES[ts.month - 1]} · {ts.hour:02d}:{ts.minute:02d}"


def _pct(p: float) -> str:
    """Probabilidad como porcentaje entero."""
    return f"{100 * p:.0f}%"


def masthead(league: str, live: bool, season: int | None = None) -> str:
    """Cabecera con la marca, la liga y el origen de los datos."""
    chip = '<span class="chip live">Datos reales</span>' if live else '<span class="chip">Simulación</span>'
    temporada = f"Temporada {season}/{(season + 1) % 100:02d}" if season else ""
    return (f'<div class="mast"><div class="brand">{LOGO}<div><div class="word">Predictive<span>·</span>Sports Odds</div>'
            f'<div class="sub">Pizarra de valor · modelo Dixon-Coles + mercado</div></div></div>'
            f'<div class="meta"><b>{e(league)}</b>{chip}<br>{e(temporada)}</div></div>')


def prob_bar(p_home: float, p_draw: float, p_away: float) -> str:
    """Barra 1/X/2 con porcentajes escritos (identidad nunca solo por color)."""
    seg = "".join(
        f'<div style="flex:{max(p, 0.001):.4f};background:{c}" title="{label} {_pct(p)}">{_pct(p) if p >= 0.12 else ""}</div>'
        for p, c, label in ((p_home, HOME, "Local"), (p_draw, DRAW, "Empate"), (p_away, AWAY, "Visitante")))
    return (f'<div class="bar" role="img" aria-label="Local {_pct(p_home)}, empate {_pct(p_draw)}, '
            f'visitante {_pct(p_away)}">{seg}</div>'
            '<div class="legend"><span>1 Local</span><span>X Empate</span><span>2 Visitante</span></div>')


def match_card(pred: MatchPrediction, value_count: int = 0) -> str:
    """Tarjeta tipo marcador de un partido."""
    when = fecha(pred.kickoff)
    top = pred.most_likely_scores(1)[0]
    badge = f'<span class="badge">{value_count} valor</span>' if value_count else ""
    p1, px, p2 = (pred.probability(s) for s in (Selection.home(), Selection.draw(), Selection.away()))
    return (f'<div class="card"><div class="when">{e(when)}</div><div class="teams">'
            f'<div class="team">{e(pred.home_team)}</div><div class="xg">xG {pred.lambda_home:.2f}</div>'
            f'<div class="team">{e(pred.away_team)}</div><div class="xg">xG {pred.lambda_away:.2f}</div></div>'
            f'{prob_bar(p1, px, p2)}<div class="foot"><span>Más de 2.5 <b>{_pct(pred.probability(Selection.over(2.5)))}</b>'
            f' · Ambos marcan <b>{_pct(pred.probability(Selection.btts(True)))}</b></span>'
            f'<span>{badge} <b class="mono">{top[0]}-{top[1]}</b></span></div></div>')


def pick_ticket(pick: Pick, bankroll: float, my_book: str = "", min_edge: float = 0.04, why: Sequence[str] = ()) -> str:
    """Boleto de una apuesta de valor.

    Args:
        pick: Pick.
        bankroll: Bankroll para el stake en dinero.
        my_book: Casa del usuario sin momios automáticos (p. ej. Caliente): se muestra el momio mínimo.
        min_edge: EV ajustado mínimo para calcular ese momio.
        why: Líneas de explicación ("¿por qué este pick?").
    """
    mine = (f'<div class="mine">En {e(my_book)}: apuesta si paga <b>≥ {pick.minimum_odds(min_edge):.2f}</b></div>'
            if my_book else "")
    reasons = (f'<ul class="why">{"".join(f"<li>{e(r)}</li>" for r in why)}</ul>' if why else "")
    return (f'<div class="card ticket"><div class="when">{e(pick.league)} · {e(pick.bookmaker)}</div>'
            f'<div class="sel">{e(pick.selection.label)}</div><div class="fx">{e(pick.fixture)}</div>'
            f'<div class="row"><div class="price">{pick.odds:.2f}<small>momio</small></div>'
            f'<div><span class="badge">EV {pick.adjusted_edge:+.1%}</span></div>'
            f'<div class="stake">Stake<b>{pick.stake_fraction * bankroll:,.2f}</b></div></div>'
            f'<div class="foot"><span>Modelo <b>{_pct(pick.model_prob)}</b></span>'
            f'<span>Mercado <b>{_pct(pick.market_prob) if pick.market_prob is not None else "—"}</b></span></div>'
            f'{mine}{reasons}</div>')


def parlay_ticket(parlay: Parlay, bankroll: float) -> str:
    """Boleto de un parlay."""
    legs = "".join(f"<li>{e(l.fixture)} · <b>{e(l.selection.label)}</b> <span class='mono'>{l.odds:.2f}</span></li>"
                   for l in parlay.legs)
    return (f'<div class="card ticket"><div class="when">Parlay · {len(parlay.legs)} patas · riesgo '
            f'{e(parlay.risk_level.lower())}</div><ul class="legs">{legs}</ul>'
            f'<div class="row"><div class="price">{parlay.combined_odds:.2f}<small>momio</small></div>'
            f'<div><span class="badge">EV {parlay.expected_value:+.1%}</span></div>'
            f'<div class="stake">Stake<b>{parlay.stake_fraction * bankroll:,.2f}</b></div></div>'
            f'<div class="foot"><span>Probabilidad conjunta <b>{_pct(parlay.joint_prob)}</b></span></div></div>')


def grid(cards: Iterable[str]) -> str:
    """Rejilla adaptable (2-4 columnas en escritorio, 1 en el celular)."""
    return f'<div class="grid">{"".join(cards)}</div>'


def empty(message: str) -> str:
    """Estado vacío con mensaje."""
    return f'<div class="empty">{e(message)}</div>'


def value_counts(picks: Sequence[Pick]) -> dict[str, int]:
    """Número de apuestas de valor por partido."""
    out: dict[str, int] = {}
    for p in picks:
        out[p.match_id] = out.get(p.match_id, 0) + 1
    return out


__all__ = ["AWAY", "CSS", "DRAW", "HOME", "empty", "fecha", "grid", "masthead", "match_card", "parlay_ticket",
           "pick_ticket", "prob_bar", "value_counts"]
