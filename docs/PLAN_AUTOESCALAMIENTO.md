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
| 2 | [ ] | **Calibración y persistencia de modelos**: diagrama de fiabilidad, calibración isotónica/Platt opcional, guardar y cargar modelos (`joblib`) | Evita reentrenar; probabilidades más fiables | 1 |
| 3 | [ ] | **Ajuste por lesiones**: impacto por jugador ausente sobre λ (heurística por titularidad/posición) e integración con `fetch_injuries` | Mayor fuente de error del modelo | — |
| 4 | [ ] | **Modelo multiliga para UCL**: coeficiente de fuerza por liga, entrenamiento conjunto con varias ligas | La UCL hoy trata a los equipos como de fuerza promedio | — |
| 5 | [x] | **CI en GitHub Actions**: pytest + mypy en cada push | Calidad continua | — |
| 6 | [ ] | **Registro de apuestas y bankroll** (SQLite): guardar picks, liquidarlos con resultados y llevar el P&L | Seguimiento real del rendimiento | 1 |
| 7 | [ ] | **App Streamlit** (`app.py`): predicciones, value bets, parlays, backtest y ajustes | Fase de UI | 1, 6 |
| 8 | [ ] | **Proveedor de xG de Understat** (opcional, con caché y tolerancia a fallos) | Mejor señal en ligas top | — |

## Registro de ejecución
| Fecha (UTC) | Tarea | Commit | Notas |
|---|---|---|---|
| 2026-11-04 | Base del sistema (Fases 1-5) | inicial | 26 tests, mypy limpio |
| 2026-11-04 | #1 Backtester walk-forward | ver git log | ROI/CLV/drawdown/calibración. Hallazgo: EV del modelo sobreestimado (≈18% esperado vs 3.8% real). `value_model_weight=0.5` → EV real 5.3%, CLV 4.0% (4 semillas). Momios históricos simulados con línea de cierre; cierre de Pinnacle en football-data |
| 2026-11-04 | #5 CI GitHub Actions | ver git log | pytest + mypy + demo en Python 3.10 y 3.12 |
