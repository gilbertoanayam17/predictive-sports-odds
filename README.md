# predictive-sports-odds

Backend en Python que analiza datos de fútbol para detectar **apuestas de valor** y construir **parlays** con gestión de riesgo (Kelly modificado). Empieza con Premier League, Champions League y Liga MX; la arquitectura no depende de ninguna liga ni deporte.

> La reflexión de la Fase 1 (fuentes de datos y features) está en [`docs/fase1_reflexion.md`](docs/fase1_reflexion.md).

## Inicio rápido

```bash
pip install -e ".[dev]"          # o: pip install -r requirements.txt
predictive-odds --help           # comando instalado (equivale a python main.py)
python main.py                                  # EPL simulada: no necesita red ni API keys
python main.py --league LIGA_MX --bankroll 5000
python main.py --same-match                     # permite parlays con varias patas del mismo partido
python main.py --backtest                       # backtest walk-forward: ROI, CLV, drawdown, calibración
python main.py --track bets.sqlite              # guarda las apuestas sugeridas (idempotente)
python main.py --track bets.sqlite --settle --played-rounds 29   # liquida (en la simulación, juega la jornada 29)
python -m pytest
```

Con datos reales (histórico de football-data.co.uk + momios de The Odds API):

```bash
export ODDS_API_KEY=...          # obligatoria
export API_FOOTBALL_KEY=...      # opcional: calendario, lesiones, UCL
python main.py --live --league EPL
```

## Arquitectura

```
predictive_odds/
├── config.py              # Registro de ligas (IDs de cada proveedor) → agregar una liga = 1 entrada
├── domain.py              # Selection (mercado como máscara de marcadores), MatchPrediction, esquemas
├── data_fetcher.py        # HttpClient robusto + proveedores intercambiables + DataFetcher (fachada)
├── statistical_models.py  # Heurísticas, Elo, features sin fuga, Dixon-Coles, ML, ensamble
├── parlay_builder.py      # Value bets, conflictos, parlays, Kelly modificado, tope de exposición
├── backtester.py          # Backtest walk-forward con CLV contra la línea de cierre
├── calibration.py         # Calibración isotónica/Platt, ECE y tabla de fiabilidad
├── model_store.py         # Guardar/cargar modelos entrenados (joblib + metadatos)
├── injuries.py            # Bajas → multiplicadores de ataque/defensa sobre λ
├── bet_tracker.py         # Registro SQLite: apuestas, liquidación, P&L y CLV
├── xg_sources.py          # merge_xg (fusión de xG de cualquier fuente) + Understat (experimental)
├── tuning.py              # Ajuste de hiperparámetros con validación cronológica
├── risk.py                # Monte Carlo del bankroll: ruina, drawdown, fracción de Kelly
└── pipeline.py            # AnalysisPipeline.run() → AnalysisResult (DataFrames listos para UI)
├── cli.py                 # CLI (también como comando `predictive-odds`)
main.py                    # Demo de punta a punta en la terminal (envoltorio del CLI)
```

**Idea central:** cada `Selection` (1X2, doble oportunidad, Over/Under x.5, BTTS, marcador exacto) se representa como una máscara booleana sobre la matriz de marcadores `P(local=i, visitante=j)`. De ahí sale todo:
- la probabilidad de cualquier mercado, a partir de una sola matriz de Poisson;
- la probabilidad conjunta **exacta** de varias patas del mismo partido (correlación);
- la detección de contradicciones (sin ningún marcador en común; p. ej. *Menos de 1.5 + Ambos marcan*) y de redundancias (*Más de 2.5* ya implica *Más de 1.5*).

### Datos (`data_fetcher.py`)
| Proveedor | Uso |
|---|---|
| `FootballDataCoUkProvider` | Histórico gratuito + momios de cierre (backtesting) |
| `ApiFootballProvider` | Calendario, resultados, xG por partido (con presupuesto de llamadas), lesiones |
| `TheOddsApiProvider` | Momios en vivo de muchas casas (1X2, totals, BTTS opcional) |
| `SimulatedDataProvider` | Liga sintética con "verdad" conocida, para demos y tests |

Manejo de errores: timeouts, reintentos con backoff exponencial, `Retry-After` en 429, sin reintentos en 401/403, errores de API-Football que llegan con HTTP 200, JSON inválido, caché en disco (las API keys nunca se guardan) y validación de esquema que descarta filas corruptas. Excepciones tipadas: `AuthenticationError`, `RateLimitError`, `DataValidationError`, `ProviderNotSupportedError`. Los nombres de equipos se emparejan entre fuentes con alias y coincidencia aproximada (`Man Utd` → `Manchester United`, `U.N.A.M.` → `Pumas UNAM`).

### Modelos (`statistical_models.py`)
- **HeuristicAnalyzer:** tabla, forma de los últimos 5 (general, local y visitante), H2H, fuerza del calendario (PPG de los rivales) y diferencial de goles ajustado por rival → índice y tendencia.
- **PoissonModel (Dixon-Coles):** ataque/defensa por equipo, ventaja de local, decaimiento temporal, mezcla goles/xG, prior gaussiano y corrección ρ de marcadores bajos. Se ajusta con L-BFGS y gradiente analítico (<0.1 s).
- **MLOutcomeModel:** regresión logística (o Random Forest) para 1X2, OU 2.5 y BTTS sobre 20 features pre-partido (Elo, forma, xG, calendario, H2H) calculadas en orden cronológico.
- **EnsemblePredictor:** mezcla Poisson + ML y conserva la matriz para las correlaciones.
- **Validación:** partición cronológica con log-loss, Brier y accuracy contra una referencia de frecuencias.

### Motor de apuestas (`parlay_builder.py`)
1. **Valor:** se toma el mejor precio entre casas y se quita el margen a cada casa (método *power*) para obtener la probabilidad justa de consenso. La decisión usa la probabilidad del modelo encogida al 50% hacia el mercado (`EV_ajustado ≥ 4%`), lo que corrige la "maldición del ganador" (ver Backtesting). Se descartan los picks en los que el modelo discrepa del mercado por más de 15 pp, porque suele significar información que el modelo no tiene, como lesiones.
2. **Parlays:** las patas candidatas son "bankers" con probabilidad ≥ 55% y EV ≥ 0. Se combinan de 2 a 3 patas, se descartan las contradictorias y redundantes, y se calcula la probabilidad conjunta (independencia entre partidos y matriz exacta dentro del mismo partido). Luego se filtra por probabilidad mínima, momio máximo y EV, y se ordena por crecimiento logarítmico esperado. Una misma pata no puede aparecer en más de 2 parlays.
3. **Kelly modificado:** ¼ de Kelly; la probabilidad del modelo se acerca a la del mercado (confianza 0.6); cada pata extra multiplica el stake por 0.8; tope de 3% por sencilla y 1.5% por parlay.
4. **Exposición:** si la suma de stakes supera el 10% del bankroll, se escalan todos.

### Backtesting (`backtester.py`)
Walk-forward fecha a fecha: reentrena solo con el pasado, apuesta con los momios pre-partido, liquida con el resultado real y mide ROI, drawdown, acierto, calibración por tramos y **CLV** contra la línea de cierre (Pinnacle en football-data.co.uk; una línea "sharp" en la simulación). La línea de cierre nunca se usa para decidir.

Hallazgo medido (simulación, 4 semillas, ≈1,400–1,800 apuestas por semilla): con el modelo puro (`value_model_weight=1.0`), el EV esperado medio de los picks es ≈18% y el real solo 4.6%. Es la "maldición del ganador": el filtro elige justo donde el modelo más se equivoca. Encoger la probabilidad al 50% hacia el mercado sube el EV real a 6.2%, el CLV de 2.6% a 4.1% y el crecimiento del bankroll de +27% a +35%; limitar los momios a ≤ 4 empeora. Por eso el valor por defecto es `value_model_weight=0.5`. Los parlays del backtest rinden peor que las sencillas, como era de esperar por la varianza y el error compuesto del modelo.

### Modelo multiliga (UCL)
En competiciones internacionales, Poisson se entrena junto con las ligas domésticas (`extra_leagues`, por defecto la EPL para la UCL) y estima un coeficiente de fuerza por liga `L`: ataque + L y defensa − L. Dentro de una liga, L se cancela; en los cruces entre ligas es lo que permite comparar a los equipos. Medido con UCL escasa (8 jornadas de entrenamiento, 8 semillas): combinar datos baja el error de λ de 0.242 a 0.215 y el log-loss de 1.039 a 1.025. El coeficiente de liga aporta además cuando las ligas difieren mucho (0.229 → 0.220; log-loss 1.007 → 1.001).

### Lesiones (`injuries.py`)
Las bajas (API-Football o simuladas) se traducen en multiplicadores de ataque y defensa que modifican las λ de Poisson. Como el ML no conoce ese contexto, sus probabilidades se corrigen en la misma proporción. Medido en la simulación (20 jornadas con bajas que las casas sí descuentan), el EV real de los value bets sube de 6.3% a 6.5%, y de 5.7% a 6.1% en los partidos con bajas. La mejora es moderada porque el encogimiento hacia el mercado ya absorbía parte del efecto. Activado por defecto (`use_injuries=True`).

### Riesgo Monte Carlo (`risk.py`, `--risk`)
Repite la jornada analizada N veces y cruza la fracción de Kelly con supuestos sobre el edge real: el modelo acierta, el edge real es la mitad, el modelo no aporta o el modelo se equivoca en contra. Resultado típico en la simulación (38 jornadas): con **¼ de Kelly y el 10% de exposición** (valores por defecto), la probabilidad de ruina es del 0% incluso en el peor escenario, y la de un drawdown mayor al 30%, del ≈5%. Con Kelly completo, en el peor escenario la probabilidad de ruina sube al ≈36%. Esto respalda los valores por defecto.

### Ajuste de hiperparámetros (`tuning.py`, `--tune`)
Validación *rolling origin*: en varios cortes temporales se entrena con el pasado y se evalúa en las 4 semanas siguientes (log-loss medio de 1X2, OU 2.5 y BTTS). Hace una búsqueda en rejilla de vida media, peso del xG y prior de Poisson, y del peso del ML en el ensamble. En la simulación, la "mejor" configuración cambia según la semilla y mejora solo un ≈0.3%, y el ML no aporta porque los datos se generan con Poisson. Por eso **los valores por defecto no se cambiaron**: hay que decidirlos con `python main.py --tune --live` sobre datos reales.

### Registro de apuestas (`bet_tracker.py`)
SQLite con restricciones de integridad. Guarda sencillas y parlays (sin duplicar si se reejecuta el análisis), liquida cuando hay resultados (un parlay, cuando terminaron todas sus patas), permite anular apuestas, guarda la probabilidad de cierre para el CLV y devuelve el P&L, el ROI por mercado y la curva de bankroll. Solo registra; nunca coloca apuestas.

### Calibración y persistencia
- `python main.py --model-path models/epl.joblib` reutiliza los modelos guardados mientras no tengan más de 7 días de antigüedad respecto al último partido, y los guarda al reentrenar (escritura atómica, metadatos en `.json`). Solo se deben cargar bundles propios, porque joblib usa pickle.
- `ProbabilityCalibrator` (isotónico o Platt, por mercado) se ajusta sobre las selecciones evaluadas fuera de muestra en un backtest. **Está desactivado por defecto:** en la validación simulada no mejoró el Brier (0.2054 → 0.2059–0.2070), porque el modelo ya está bien calibrado. Con datos reales hay que medirlo con `calibration_report(backtest.evaluated)`.

## Interfaz web (Streamlit)

```bash
pip install -r requirements-ui.txt
streamlit run app.py
```

La barra lateral permite elegir la fuente de datos (simulada o real con `ODDS_API_KEY`), la liga, el bankroll, el ML y los umbrales de valor, parlays y exposición. Pestañas:
- **Predicciones:** probabilidades por partido, análisis heurístico, ajustes por lesiones y matriz de marcadores.
- **Value bets** y **Parlays**, con sus stakes.
- **Tabla y ratings**, más la validación fuera de muestra.
- **Backtest:** ROI, CLV, curva de bankroll y calibración.
- **Registro:** guarda las apuestas sugeridas y muestra el P&L.

La app no contiene lógica de negocio: solo llama a `AnalysisPipeline`, `Backtester` y `BetTracker`. Tiene una prueba de humo con `streamlit.testing` que corre en el CI.

## Limitaciones y próximos pasos
- Las **lesiones** se convierten en ajustes con una heurística (3% de ataque y 2% de defensa por baja; las dudas cuentan a medias; tope del 15%). Para mejorarlo hacen falta datos de importancia por jugador (`PlayerImpact`), como minutos o goles+asistencias.
- **Multiliga:** solo la EPL está configurada como liga doméstica de los participantes de la UCL. Para la UCL real hay que registrar LaLiga, Serie A, Bundesliga y Ligue 1 (`register_league`) y pasarlas en `--extra-leagues`.
- Los momios de un parlay del mismo partido se calculan como producto; un *Bet Builder* real los ajusta por correlación (el sistema lo avisa).
- **Understat** (`--live --understat`): el lector está implementado y probado contra el formato documentado del sitio, pero **no se ha podido verificar en vivo** porque la red del entorno de desarrollo bloquea understat.com. Hay que probarlo en local. Si falla, el sistema sigue funcionando sin xG. La fusión `merge_xg` (nombres normalizados, fecha ±2 días y verificación del marcador) sí está verificada.

*Herramienta analítica. Apostar implica riesgo de pérdida; juega con responsabilidad.*
