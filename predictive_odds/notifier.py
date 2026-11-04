"""Alertas de value bets por Telegram.

Configuración (una vez):

1. En Telegram, habla con **@BotFather** → ``/newbot`` → copia el *token*.
2. Escribe cualquier mensaje a tu bot nuevo.
3. Pon ``TELEGRAM_BOT_TOKEN=...`` en tu ``.env`` y ejecuta
   ``python main.py --telegram-setup``: detecta tu ``chat_id``, te dice qué
   añadir al ``.env`` y envía un mensaje de prueba.

Seguridad: el token forma parte de la URL de la API de Telegram, así que los
errores se reescriben para que **nunca** aparezca en mensajes ni registros.
Los avisos ya enviados se recuerdan (``.cache/alerts_sent.json``) para no
repetirlos en cada ejecución programada.
"""

from __future__ import annotations

import html
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests

from .parlay_builder import Parlay, Pick

logger = logging.getLogger(__name__)

API_URL = "https://api.telegram.org/bot{token}/{method}"
SENT_FILE = Path(".cache/alerts_sent.json")
MAX_MESSAGE = 4000  # Telegram admite 4096 caracteres


class NotifierError(Exception):
    """Error al enviar una alerta (sin datos sensibles en el mensaje)."""


@dataclass
class TelegramNotifier:
    """Cliente mínimo de la API de bots de Telegram.

    Attributes:
        token: Token del bot (de @BotFather).
        chat_id: Chat al que se envían los avisos.
        session: Sesión ``requests`` (inyectable en tests).
        timeout: Timeout por petición en segundos.
    """

    token: str
    chat_id: str = ""
    session: Any = None
    timeout: float = 15.0

    @classmethod
    def from_env(cls) -> TelegramNotifier | None:
        """Crea el notificador desde ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID`` (``None`` si falta el token)."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        return cls(token, os.environ.get("TELEGRAM_CHAT_ID", "").strip()) if token else None

    def _safe(self, text: str) -> str:
        """Elimina el token de cualquier texto."""
        return text.replace(self.token, "<TOKEN>") if self.token else text

    def _call(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        """Llama a un método de la API.

        Raises:
            NotifierError: Error de red, token inválido o respuesta no válida (sin el token).
        """
        if not self.token:
            raise NotifierError("Falta TELEGRAM_BOT_TOKEN")
        session = self.session or requests.Session()
        try:
            resp = session.post(API_URL.format(token=self.token, method=method), json=payload or {},
                                timeout=self.timeout)
        except requests.RequestException as exc:
            raise NotifierError(self._safe(f"Sin conexión con Telegram: {exc}")) from None
        try:
            data = resp.json()
        except ValueError:
            raise NotifierError(f"Respuesta no válida de Telegram (HTTP {resp.status_code})") from None
        if resp.status_code == 401:
            raise NotifierError("Token de Telegram inválido (revisa TELEGRAM_BOT_TOKEN)")
        if not data.get("ok"):
            raise NotifierError(self._safe(f"Telegram rechazó la petición: {data.get('description', '?')}"))
        return data.get("result")

    def bot_name(self) -> str:
        """Nombre de usuario del bot (``getMe``): comprueba que el token funciona."""
        return str((self._call("getMe") or {}).get("username", "?"))

    def discover_chat_ids(self) -> list[tuple[str, str]]:
        """Chats que han escrito al bot recientemente (``getUpdates``): ``[(chat_id, nombre)]``."""
        found: dict[str, str] = {}
        for upd in self._call("getUpdates") or []:
            chat = (upd.get("message") or upd.get("channel_post") or {}).get("chat") or {}
            if "id" in chat:
                name = chat.get("username") or chat.get("title") or chat.get("first_name") or ""
                found[str(chat["id"])] = str(name)
        return list(found.items())

    def send(self, text: str) -> None:
        """Envía un mensaje (HTML) al ``chat_id`` configurado, partiéndolo si es largo.

        Raises:
            NotifierError: Si falta el chat o falla el envío.
        """
        if not self.chat_id:
            raise NotifierError("Falta TELEGRAM_CHAT_ID (usa --telegram-setup para encontrarlo)")
        for start in range(0, len(text), MAX_MESSAGE):
            self._call("sendMessage", {"chat_id": self.chat_id, "text": text[start:start + MAX_MESSAGE],
                                       "parse_mode": "HTML", "disable_web_page_preview": True})


def pick_key(pick: Pick) -> str:
    """Clave estable de un pick para no repetir avisos."""
    return f"{pick.match_id}|{pick.selection.market.value}|{pick.selection.outcome.value}|{pick.selection.line}"


def _load_sent(path: Path) -> set[str]:
    """Claves ya avisadas."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data) if isinstance(data, list) else set()
    except (OSError, ValueError):
        return set()


def select_alerts(picks: Sequence[Pick], min_edge: float = 0.05, sent_file: Path | None = SENT_FILE,
                  max_items: int = 10) -> list[Pick]:
    """Picks que merecen aviso: EV ajustado ≥ ``min_edge`` y no avisados antes.

    Args:
        picks: Sencillas de valor (ya con stake).
        min_edge: EV ajustado mínimo.
        sent_file: Registro de avisos enviados (``None`` = no deduplicar).
        max_items: Máximo de picks por aviso.

    Returns:
        Picks ordenados por EV ajustado.
    """
    sent = _load_sent(sent_file) if sent_file else set()
    chosen = [p for p in picks if p.adjusted_edge >= min_edge and pick_key(p) not in sent]
    return sorted(chosen, key=lambda p: p.adjusted_edge, reverse=True)[:max_items]


def mark_sent(picks: Sequence[Pick], sent_file: Path = SENT_FILE, keep: int = 2000) -> None:
    """Recuerda los picks avisados (conserva los ``keep`` más recientes)."""
    sent = list(_load_sent(sent_file) | {pick_key(p) for p in picks})[-keep:]
    try:
        sent_file.parent.mkdir(parents=True, exist_ok=True)
        sent_file.write_text(json.dumps(sent), encoding="utf-8")
    except OSError as exc:
        logger.warning("No se pudo guardar el registro de avisos: %s", exc)


def format_alert(picks: Sequence[Pick], bankroll: float, parlays: Sequence[Parlay] = (), title: str = "") -> str:
    """Mensaje HTML con los picks (y opcionalmente parlays).

    Args:
        picks: Picks a avisar.
        bankroll: Bankroll para mostrar el stake en dinero.
        parlays: Parlays a incluir.
        title: Encabezado (p. ej. la liga).

    Returns:
        Texto con formato HTML de Telegram (contenido escapado).
    """
    e = html.escape
    lines = [f"<b>⚽ {e(title or 'Value bets')}</b>"]
    for p in picks:
        kick = pd.Timestamp(p.kickoff) if p.kickoff is not None else pd.NaT
        when = "" if pd.isna(kick) else kick.strftime("%d/%m %H:%M")
        lines.append(f"\n<b>{e(p.fixture)}</b> {e(when)}\n{e(p.selection.label)} @ <b>{p.odds:.2f}</b> "
                     f"({e(p.bookmaker)})\nEV ajustado {p.adjusted_edge:+.1%} · stake {p.stake_fraction * bankroll:,.2f}")
    for par in parlays:
        legs = " + ".join(f"{leg.fixture}: {leg.selection.label}" for leg in par.legs)
        lines.append(f"\n🎯 <b>Parlay</b> @ {par.combined_odds:.2f}: {e(legs)}\n"
                     f"P {par.joint_prob:.0%} · EV {par.expected_value:+.1%} · stake {par.stake_fraction * bankroll:,.2f}")
    lines.append("\n<i>Herramienta analítica; no es asesoría. Apuesta con responsabilidad.</i>")
    return "\n".join(lines)


def notify_value_bets(notifier: TelegramNotifier, picks: Sequence[Pick], bankroll: float, title: str = "",
                      min_edge: float = 0.05, sent_file: Path = SENT_FILE) -> int:
    """Selecciona, envía y recuerda los avisos nuevos.

    Returns:
        Número de picks avisados (0 si no había nada nuevo).

    Raises:
        NotifierError: Si falla el envío (en ese caso no se marcan como enviados).
    """
    chosen = select_alerts(picks, min_edge, sent_file)
    if not chosen:
        return 0
    notifier.send(format_alert(chosen, bankroll, title=title))
    mark_sent(chosen, sent_file)
    return len(chosen)


__all__ = ["NotifierError", "TelegramNotifier", "format_alert", "mark_sent", "notify_value_bets", "pick_key",
           "select_alerts"]
