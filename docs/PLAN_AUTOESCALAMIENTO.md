# Plan de auto-escalamiento

Plan para que el desarrollo continúe de forma autónoma: al terminar una tarea, se toma la siguiente pendiente de este archivo. **Este archivo es la fuente de verdad del progreso.**

## Protocolo por tarea
1. Tomar la primera tarea `[ ]` de la tabla (respetando dependencias) y marcarla `[~]`.
2. Implementar con type hints, docstrings y manejo de errores (mismas reglas que el resto del código).
3. Añadir tests. **Criterio de "hecho":** `python -m pytest` en verde, `mypy predictive_odds main.py` sin errores y `python main.py` funcionando.
4. Commit con un mensaje descriptivo y push a `claude/ecstatic-wright-6nlr98`.
5. Marcar `[x]`, anotar el commit y las notas en el registro, y pasar a la siguiente tarea.

## Reglas de seguridad
- Solo datos simulados o mocks: **no se consumen cuotas de API** ni se usan keys reales.
- Nunca se commitean secretos (`.env` y `.cache/` están en `.gitignore`).
- El sistema solo analiza y sugiere: **nunca coloca apuestas**.
- No se reescribe el historial, no se fuerza el push y no se abren PRs sin permiso.
- Si una tarea se bloquea (dependencia o red), se marca `[!]` con el motivo y se pasa a la siguiente.
- No se cambian los valores por defecto del modelo sin una mejora medida en la validación.

## Backlog priorizado

| # | Estado | Tarea | Valor | Depende de |
|---|---|---|---|---|
| 1 | [x] | **Backtester walk-forward** (`backtester.py`): reentrenar por jornada, apostar con las reglas del motor, medir ROI, yield, drawdown, CLV y la calibración por tramos | Indispensable: sin backtest no se sabe si el sistema gana dinero | — |
| 2 | [x] | **Calibración y persistencia de modelos**: diagrama de fiabilidad, calibración isotónica/Platt opcional, guardar y cargar modelos (`joblib`) | Evita reentrenar; probabilidades más fiables | 1 |
| 3 | [x] | **Ajuste por lesiones**: impacto por jugador ausente sobre λ (heurística por titularidad/posición) e integración con `fetch_injuries` | Mayor fuente de error del modelo | — |
| 4 | [x] | **Modelo multiliga para UCL**: coeficiente de fuerza por liga, entrenamiento conjunto con varias ligas | La UCL hoy trata a los equipos como de fuerza promedio | — |
| 5 | [x] | **CI en GitHub Actions**: pytest + mypy en cada push | Calidad continua | — |
| 6 | [x] | **Registro de apuestas y bankroll** (SQLite): guardar picks, liquidarlos con resultados y llevar el P&L | Seguimiento real del rendimiento | 1 |
| 7 | [x] | **App Streamlit** (`app.py`): predicciones, value bets, parlays, backtest y ajustes | Fase de UI | 1, 6 |
| 8 | [~] | **Proveedor de xG de Understat** (opcional, con caché y tolerancia a fallos) | Mejor señal en ligas top | — |

### Fase 2 (generada automáticamente al terminar el backlog inicial)

| # | Estado | Tarea | Valor | Depende de |
|---|---|---|---|---|
| 9 | [x] | **Empaquetado** (`pyproject.toml`, `pip install -e .`, comando `predictive-odds`) | Instalación y reutilización triviales | — |
| 10 | [x] | **Ajuste de hiperparámetros** con validación cronológica (vida media, peso del xG, prior, peso del ML); solo se cambian los valores por defecto si mejoran fuera de muestra | Mejor modelo, medido | 1 |
| 11 | [x] | **Simulador de riesgo Monte Carlo**: distribución de bankroll, drawdown y riesgo de ruina según la fracción de Kelly y el tope de exposición | Elegir el staking con datos | 1 |
| 12 | [x] | **Informe diario exportable** (Markdown/HTML) con picks, parlays, stakes y avisos | Consumo fácil fuera de la terminal | — |
| 13 | [x] | **Mercados extra**: Draw No Bet y hándicap asiático (con push/medio push) | Más oportunidades de valor | — |
| 14 | [x] | **Histórico de momios** (snapshots en SQLite) para medir el movimiento de línea y el CLV en vivo | CLV real sin depender del cierre de terceros | 6 |

### Fase 3 (generada automáticamente al terminar la fase 2)

| # | Estado | Tarea | Valor | Depende de |
|---|---|---|---|---|
| 15 | [x] | **Pasada de robustez**: casos límite del pipeline como tests de regresión | Menos sorpresas en producción | — |
| 16 | [x] | **Jornada multiliga**: analizar varias ligas a la vez y construir parlays con picks de todas, con un tope de exposición global | Más diversificación y combinaciones | — |
| 17 | [x] | **Operación diaria**: script listo para cron (snapshot + análisis + informe + liquidación) y guía de uso en `docs/` | Usar el sistema día a día | 12, 14 |
| 18 | [x] | **Rendimiento del backtest**: reentreno cada N fechas y medición del impacto en las métricas | Backtests más rápidos para iterar | 1 |

### Fase 4 (generada en el check-in de las 09:38 UTC)

| # | Estado | Tarea | Valor | Depende de |
|---|---|---|---|---|
| 19 | [ ] | **App al día**: pestañas/acciones para jornada multiliga, riesgo Monte Carlo y descarga del informe | La UI cubre lo añadido después de la tarea 7 | 7, 11, 12, 16 |
| 20 | [x] | **Cobertura de tests**: medirla y cubrir huecos en los módulos críticos (dinero/liquidación) | Menos riesgo de errores silenciosos | — |

## Registro de ejecución
| Fecha (UTC) | Tarea | Commit | Notas |
|---|---|---|---|
| 2026-11-04 | Base del sistema (Fases 1-5) | inicial | 26 tests, mypy limpio |
| 2026-11-04 | #1 Backtester walk-forward | ver git log | ROI/CLV/drawdown/calibración. Hallazgo: EV del modelo sobreestimado (≈18% esperado vs 3.8% real). `value_model_weight=0.5` → EV real 5.3%, CLV 4.0% (4 semillas). Momios históricos simulados con línea de cierre; cierre de Pinnacle en football-data |
| 2026-11-04 | #5 CI GitHub Actions | ver git log | pytest + mypy + demo en Python 3.10 y 3.12 |
| 2026-11-04 | #2 Calibración y persistencia | ver git log | `calibration.py` (isotónico/Platt, ECE, fiabilidad) y `model_store.py` (bundle joblib con validación de liga/formato/antigüedad). Calibración medida: no mejora el Brier en la simulación → desactivada por defecto. `--model-path` en main |
| 2026-11-04 | Fix CI | ver git log | mypy fallaba en Python 3.10 (numpy 2.2/pandas-stubs 2.3); reproducido con esas versiones y corregido |
| 2026-11-04 | #3 Ajuste por lesiones | ver git log | `injuries.py` + `TeamAdjustment` en Poisson/ensamble (ML corregido por razón). Bajas simuladas que las casas descuentan. EV real de los value bets 6.3%→6.5% (20 semillas); en partidos con bajas 5.7%→6.1%. Activado por defecto |
| 2026-11-04 | #4 Modelo multiliga UCL | ver git log | Coeficiente de liga en Poisson (gradiente verificado numéricamente), simulación coherente entre competiciones, `extra_leagues`/`--extra-leagues`. UCL escasa: error de λ 0.242→0.215 al combinar con la EPL; el coeficiente de liga ayuda si la diferencia entre ligas es grande (0.229→0.220) |
| 2026-11-04 | #6 Registro de apuestas | ver git log | `bet_tracker.py` (SQLite idempotente, liquidación de sencillas/parlays, anulación, CLV, P&L por mercado, curva de bankroll); CLI `--track`, `--settle`, `--played-rounds`. También corrige una incoherencia de la simulación: las bajas ahora dependen de temporada/jornada y afectan a resultados y momios por igual (efecto de lesiones re-medido: EV 6.2%→6.4%) |
| 2026-11-04 | #7 App Streamlit | ver git log | `app.py` con 6 pestañas (predicciones, value bets, parlays, tabla/ratings, backtest, registro), memoizada; prueba de humo con AppTest en el CI (3.12) y verificación visual con Playwright |
| 2026-11-04 | #8 xG Understat (parcial) | ver git log | `xg_sources.py`: `merge_xg` verificado (nombres, fecha ±2 días, valida marcador) y `XgEnrichedProvider` tolerante a fallos; `--understat`. **Lector de Understat NO verificado en vivo** (red bloqueada: CONNECT 403): queda `[~]` hasta probarlo en local |
| 2026-11-04 | Re-medición w | — | Con la simulación corregida (bajas por jornada): w=0.5 frente a 1.0 → EV real 6.2% vs 4.6%, CLV 4.1% vs 2.6%, crecimiento +35% vs +27%. Decisión confirmada |
| 2026-11-04 | #9 Empaquetado | ver git log | `pyproject.toml` (extras `ui` y `dev`), CLI movido a `predictive_odds/cli.py`, comando `predictive-odds` verificado con instalación editable fuera del repo; versión 0.2.0 |
| 2026-11-04 | #10 Ajuste de hiperparámetros | ver git log | `tuning.py` (rolling origin, rejilla de Poisson y peso del ML) + `--tune`. En la simulación la mejor configuración varía con la semilla (mejora ≈0.3%) y el ML no aporta (datos generados con Poisson) → valores por defecto SIN cambios; decidir con datos reales |
| 2026-11-04 | #11 Riesgo Monte Carlo | ver git log | `risk.py` + `--risk`: fracción de Kelly × recorte del edge. Con ¼ de Kelly y el 10% de exposición, P(ruina) = 0% incluso con el modelo en contra; con Kelly completo, ≈36% → respalda los valores por defecto |
| 2026-11-04 | #12 Informe exportable | ver git log | `report.py` + `--report archivo.md/.html`: HTML autocontenido (sin recursos externos, modo claro/oscuro, nombres escapados); verificado en móvil (390 px, sin desbordamiento horizontal) |
| 2026-11-04 | #13 DNB y hándicap asiático | ver git log | Rejilla de resultados graduados, `p_eff`/`action_share` (EV exacto, Kelly exacto con push), liquidación con push y medios resultados en backtest y registro, emparejamiento de las caras del hándicap al quitar el margen; Odds API `spreads`, columnas AH de football-data (+ cierre), simulación. Excluidos de parlays. 21 tests nuevos. Backtest simulado: hándicap asiático con ROI +11%, CLV +6% |
| 2026-11-04 | #14 Histórico de momios | ver git log | `odds_history.py`: snapshots SQLite, movimiento del mejor precio, cierre propio (última captura pre-partido; ignora capturas en juego y la pseudo-casa de cierre) que alimenta el CLV del registro. CLI `--snapshot`, `--movement` |
| 2026-11-04 | #15 Robustez | ver git log | 5 regresiones: sin proveedor de momios, temporada terminada, momios corruptos, inicio de temporada (ML y evaluación se desactivan solos), histórico insuficiente |
| 2026-11-04 | #16 Jornada multiliga | ver git log | `analyze_slate` + `SlateResult` (ligas fallidas → `errors`), parlays entre ligas, tope global, `max_singles`; CLI `--leagues`, `--max-singles`. Detectado: concentración de sencillas correlacionadas en un mismo partido → se mide un tope por partido |
| 2026-11-04 | #17 Operación diaria | ver git log | `scripts/daily_run.sh` (cron; modo `SIMULATE=1` verificado de punta a punta; reparte el bankroll entre ligas para respetar el tope conjunto) y `docs/GUIA_USO.md` (claves, flujo, lectura de columnas, validación con datos reales, cuotas de API, problemas frecuentes) |
| 2026-11-04 | #18 Rendimiento | ver git log | Perfilado: `market_consensus` consumía el 73% del tiempo → vectorizado con numpy (bisección vectorizada para el devig *power*; resultados idénticos a 1e-13). Selecciones mapeadas por combinación única y *warm start* de Poisson entre fechas. Backtest de ~56 s (perfilado) / 13 s → 6 s con métricas equivalentes |
| 2026-11-04 | Tope por partido (medición) | — | Backtest de 6 semillas: tope 2%/3% vs sin tope → drawdown medio 13.8%/13.4% vs 12.7%, crecimiento +41%/+42% vs +41%. Sin mejora de riesgo medible → `max_match_exposure=None` por defecto (disponible como opción) |
| 2026-11-04 | Fase 3 completada | — | Backlog 1–18 terminado salvo #8 (Understat sin verificar en vivo por bloqueo de red) |
| 2026-11-04 | #20 Cobertura | ver git log | 82% → 90% (CLI 0% → 69%, data_fetcher 77% → 87%). **Bug real encontrado y corregido:** en el formato "extra" de football-data (Liga MX) solo hay momios de cierre, así que el backtest real de Liga MX no habría tenido precios donde apostar; ahora usa el cierre de cada casa (backtest conservador). Tests con CSV en los formatos publicados, xG/lesiones de API-Football y humo del CLI |
