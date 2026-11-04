# Plan de auto-escalamiento

Plan para que el desarrollo continúe de forma autónoma: al terminar una tarea, se toma la siguiente pendiente de este archivo. **Este archivo es la fuente de verdad del progreso.**

## Protocolo por tarea
1. Tomar la primera tarea `[ ]` de la tabla (respetando dependencias) y marcarla `[~]`.
2. Implementar con type hints, docstrings y manejo de errores (mismas reglas que el resto del código).
3. Añadir tests. **Criterio de "hecho":** `python -m pytest` en verde, `mypy predictive_odds main.py` sin errores (también con las versiones del CI en Python 3.10: numpy 2.2 y pandas-stubs 2.3, cuya verificación de tipos es más estricta) y `python main.py` funcionando. Tras cada push, confirmar que el CI quedó en verde.
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
| 19 | [x] | **App al día**: pestañas/acciones para jornada multiliga, riesgo Monte Carlo y descarga del informe | La UI cubre lo añadido después de la tarea 7 | 7, 11, 12, 16 |
| 20 | [x] | **Cobertura de tests**: medirla y cubrir huecos en los módulos críticos (dinero/liquidación) | Menos riesgo de errores silenciosos | — |

### Fase 5 (aprobada por el usuario: orden 21 → 22 → 23 → 24 → 25; 26 si sobran créditos)

| # | Estado | Tarea | Valor |
|---|---|---|---|
| 21 | [x] | **Ahorro de créditos de The Odds API**: endpoint gratuito de eventos antes de pedir momios, horizonte de partidos, 1 región por defecto, contador de créditos | Rinde más el plan gratuito |
| 22 | [x] | **Estudio por liga y mercado**: backtest multi-temporada → configuración recomendada por liga que la app aplica | Apostar solo donde hay ventaja |
| 23 | [x] | **Panel de rendimiento real**: CLV/ROI por liga y mercado, real vs esperado, rachas | Saber si se gana por habilidad o por suerte |
| 24 | [x] | **Alertas por Telegram** de value bets con EV alto (tarea programada de Windows) | Avisos en el celular |
| 25 | [x] | **Revisión de código y limpieza**: auditoría de errores y división de `app.py` | Menos errores silenciosos |
| 26 | [x] | **App en el celular**: preparación para Streamlit Community Cloud con *secrets* | Uso desde cualquier lugar |

### Fase 6 (aprobada por el usuario: A → B → C → D → F; usa Caliente, que The Odds API no cubre)

| # | Estado | Tarea | Valor |
|---|---|---|---|
| 27 | [x] | **Caliente y apuestas reales**: momio mínimo por selección para comparar en Caliente; registro aparte de las apuestas reales (momio y monto tomados) | Valor que de verdad puedes tomar; rendimiento real |
| 28 | [ ] | **"¿Por qué este pick?"**: explicación breve en cada boleto | Descartar picks absurdos |
| 29 | [ ] | **Más ligas**: Ligue 1, Eredivisie, Championship, Portugal, MLS | Más oportunidades en mercados menos eficientes |
| 30 | [ ] | **Ciclo semanal automático**: liquidar, analizar, registrar sugerencias y resumen + picks por Telegram (tarea programada semanal) | Sin depender de acordarse |
| 31 | [ ] | **Cierre formal**: pull request, capturas en el README, versión 1.0 | Punto estable |

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
| 2026-11-04 | #19 App al día | ver git log | Descarga del informe (HTML/Markdown) en la barra lateral, pestaña **Riesgo** (Monte Carlo) y pestaña **Multiliga**; pruebas AppTest de ambas y verificación visual con Playwright. Fase 4 completada |
| 2026-11-04 | Check-in 10:53 · CI roto | ver git log | El CI fallaba desde la corrida 19 (solo mypy en Python 3.10, por la tipificación de numpy en el consenso vectorizado); los tests pasaban. No se detectó antes porque no repetí la verificación de mypy con las versiones del CI. Corregido y regla añadida al protocolo |
| 2026-11-04 | Cierre del ciclo (13:15 UTC) | ver git log | 19 de 20 tareas completadas; #8 (Understat) sigue `[~]`, bloqueada por red. Tests en verde y CI en verde (corrida 23). Pendiente de decisión del usuario: validar con datos reales y abrir PR |
| 2026-11-04 | Petición del usuario: datos reales + UX web | ver git log | `connectivity.py` (comprobación de conexión sin gastar créditos), la app en modo Real: enlaces para conseguir las claves, clave de API-Football, botón *Probar conexión*, guía de bienvenida y **backtest con datos reales sin clave** (football-data.co.uk). Guía de primeros pasos en `docs/GUIA_USO.md` |
| 2026-11-04 | Feedback del primer uso real | ver git log | (1) La liquidación en modo real nunca emparejaba: IDs de The Odds API ≠ IDs de football-data → ahora por equipos + fecha (±2 días), guardando equipos/fecha en cada pata. (2) Nombres de recién ascendidos ("Hull City" ↔ "Hull") → emparejamiento por palabras contenidas, con candidato único. (3) Temporada por defecto = temporada en curso (`current_season`); aviso de equipos sin historial. (4) Botón *Liquidar con resultados* en la app |
| 2026-11-04 | Petición del usuario: sin UCL; + La Liga, Serie A, Bundesliga | ver git log | El plan gratuito de API-Football solo cubre 2022–2024 → UCL inutilizable gratis: se oculta en app/CLI (`selectable_leagues`), el código multiliga se conserva. Nuevas ligas con football-data (SP1, I1, D1) y The Odds API; 58 pares de nombres football-data ↔ Odds API verificados por test (alias añadidos: Ath Bilbao, M'gladbach, Ein Frankfurt, Hamburg...). Error de plan de API-Football explicado y aviso en *Probar conexión* |
| 2026-11-04 | #21 Ahorro de créditos | ver git log | `/events` gratuito antes de `/odds`: 0 créditos si no hay partidos en 96 h; región única `eu` (2 créditos por liga en vez de 4); horizonte redondeado a la hora para que acierte la caché; créditos restantes persistidos y visibles en la app. 4 tests sin red |
| 2026-11-04 | #22 Estudio por liga | ver git log | `league_study.py`: backtest multi-temporada por liga → estadísticas por mercado → veredicto por CLV (ROI con el doble de muestra si el CLV no es comparable, p. ej. Liga MX) → `league_profiles.json`, aplicado automáticamente (`ValueBetConfig.allowed_markets`, también en parlays). CLI `--study`, panel en la app. Los tests corren en una carpeta temporal para no tocar archivos del usuario |
| 2026-11-04 | #23 Panel de rendimiento | ver git log | `performance.py` (funciones puras): beneficio real vs esperado según el cierre (si no, según el modelo), suerte en desviaciones típicas, veredicto, rachas, desglose por liga/mercado y curva acumulada; pestaña *Rendimiento* en la app |
| 2026-11-04 | #24 Alertas Telegram | ver git log | `notifier.py`: cliente propio (el token nunca aparece en errores, verificado por test), avisos sin repetir, mensajes HTML escapados y partidos en varios mensajes si son largos; CLI `--alert`, `--alert-min-ev`, `--telegram-setup` (encuentra el chat_id); `scripts/alertas_windows.ps1` para el Programador de tareas (por defecto cada 24 h ≈ 300 créditos/mes). El script .ps1 NO se pudo probar aquí (sin PowerShell) |
| 2026-11-04 | #25 Revisión de código | ver git log | 7 hallazgos, todos corregidos y con regresión: (1) **grave**: la columna "Max" de football-data (mejor precio de TODAS las casas) se usaba como casa apostable → inflaba ROI/CLV del backtest real y del estudio; excluida. (2) aviso de .env perdido tras recargar; (3) NaT rompía alertas; (4) clave gratuita de API-Football repetía llamadas rechazadas; (5) modelo guardado reutilizado pese a cambiar la configuración (huella de configuración); (6) caché ineficaz con líneas NaN; (7) estudio ocultaba AttributeError. Además: lógica de fuentes unificada en `sources.py` (la app y el CLI divergían) y ruff (pyflakes) en el CI. `main()` de la app no se dividió más: el beneficio frente al riesgo de regresiones en la UI no compensaba |
| 2026-11-04 | #26 App en el celular | ver git log | Preparada para Streamlit Community Cloud: `requirements.txt` de ejecución (con Streamlit) y `requirements-dev.txt`; **las claves del servidor ya no se envían al navegador** (antes el campo se rellenaba con el valor del .env, lo que en la nube expondría la clave a cualquiera); lectura de *secrets*; copia de seguridad/restauración validada del registro (el disco de la nube no persiste); guía `docs/DESPLIEGUE_CELULAR.md` con el aviso de hacer la app privada. El despliegue lo hace el usuario (requiere su cuenta) |
| 2026-11-04 | Rediseño de la interfaz (petición del usuario) | ver git log | Identidad "pizarra táctica": tema oscuro verde pizarra + tiza + acento ámbar, Barlow Condensed/IBM Plex, logo propio, sin emojis. Tarjetas de partido tipo marcador con barra 1/X/2 (paleta validada con el validador de la skill de visualización; empate gris = punto medio divergente; % siempre escritos), boletos de value bets y parlays (tablas en desplegables), mapa de calor de marcadores secuencial, gráficas Altair con tooltip y meses en español, métricas 2×2 en el celular sin desbordamiento horizontal (verificado con capturas a 390 px y 1440 px). `ui_html.py` con HTML escapado y tests |
| 2026-11-04 | #27 Caliente | ver git log | The Odds API no cubre Caliente (ni casas mexicanas); no se hace scraping. `Pick.minimum_odds` (exacto también con push) y `ValueBetFinder.reprice`; boletos con "En Caliente: apuesta si paga ≥ X"; pestaña *Mis apuestas* (momios mínimos de toda la jornada + registro real en `mis_apuestas.sqlite`, separado de las sugerencias); *Rendimiento* elige registro. Arreglado: widgets duplicados del panel de copias |
