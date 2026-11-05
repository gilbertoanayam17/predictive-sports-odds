# CLAUDE.md

Contexto para Claude en este repositorio.

## Usuario y preferencias
- Responde **en español**. Código, docstrings, mensajes de la app y commits también en español.
- El usuario usa **Windows con PowerShell**. Los comandos que le des deben funcionar ahí (`python -m pip`, rutas con `\`, `Get-Content`, nada de `&&` de bash).
- Su casa de apuestas es **Caliente** (México). The Odds API no la cubre: la app muestra el **momio mínimo** que debe pagar Caliente (`MI_CASA` en `.env`). No hagas scraping de Caliente.
- **Ahorra créditos de The Odds API** (plan gratuito, 500/mes): ~2 créditos por liga y consulta. El ciclo automático es semanal a propósito. Avisa del costo antes de añadir consultas.
- API-Football gratuito solo cubre las temporadas 2022–2024, así que la Champions (UCL) está oculta. Los datos históricos vienen de football-data.co.uk (gratis).
- Las claves van en `.env` (ver `.env.example`). Nunca las subas, ni las muestres, ni las envíes al navegador o a los logs.

## Dónde está el contexto
- `README.md`: arquitectura y uso.
- `docs/PLAN_AUTOESCALAMIENTO.md`: **historial de todo lo hecho** (tabla de tareas y bitácora). Añade una fila al terminar algo relevante.
- `docs/GUIA_USO.md`: operación diaria del usuario. `docs/DESPLIEGUE_CELULAR.md`: Streamlit Cloud.
- Paquete `predictive_odds/`. `app.py` es la interfaz Streamlit (sin lógica de negocio) y `main.py` / `predictive_odds/cli.py` el CLI.
- Automatización: `python main.py --live --weekly ...`, programado con `scripts/automatizar_windows.ps1` (viernes 10:00, Programador de tareas, `StartWhenAvailable`). Registro en `data\ciclo.log`. El usuario ya la instaló en su PC.
- Registros SQLite: `bets.sqlite` (sugerencias) y `mis_apuestas.sqlite` (apuestas reales). No los borres ni cambies su esquema sin migración.

## Antes de subir cambios
Ejecuta lo mismo que el CI (Python 3.10 y 3.12):
```
pip install -r requirements-dev.txt
ruff check --select F predictive_odds app.py main.py tests
mypy predictive_odds main.py --ignore-missing-imports
python -m pytest -q
python main.py --no-ml
```
- Confirma que el CI de GitHub queda en verde después de cada push; no lo des por hecho.
- Los tests no deben usar red ni el `.env` real (`tests/conftest.py` lo bloquea).
- Streamlit: `magicEnabled=false`. No uses expresiones sueltas como sentencias; da `key=` única a los widgets repetidos; en los tests de la app selecciona los controles por etiqueta, no por índice.
- La interfaz tiene identidad propia ("pizarra táctica": `predictive_odds/ui_html.py` y `.streamlit/config.toml`). Respétala y escapa el HTML.

## Advertencias conocidas
- `scripts/automatizar_windows.ps1` solo se ha probado en la PC del usuario.
- Los nombres de equipos de Ligue 1, Eredivisie, Championship, Primeira Liga y MLS no se verificaron en vivo. Si sale "sin historial", añade un alias en `TeamNameNormalizer` (`predictive_odds/data_fetcher.py`).
