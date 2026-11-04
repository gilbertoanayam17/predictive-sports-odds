# predictive-sports-odds

Backend en Python que analiza datos de fútbol para detectar **apuestas de valor** y construir **parlays** con gestión de riesgo (Kelly modificado). Empieza con Premier League, Champions League y Liga MX; la arquitectura no depende de ninguna liga ni deporte.

> La reflexión de la Fase 1 (fuentes de datos y features) está en [`docs/fase1_reflexion.md`](docs/fase1_reflexion.md).

## Inicio rápido

```bash
pip install -r requirements.txt
python main.py                                  # EPL simulada: no necesita red ni API keys
python main.py --league LIGA_MX --bankroll 5000
python main.py --same-match                     # permite parlays con varias patas del mismo partido
python main.py --backtest                       # backtest walk-forward: ROI, CLV, drawdown, calibración
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
└── pipeline.py            # AnalysisPipeline.run() → AnalysisResult (DataFrames listos para UI)
main.py                    # Demo de punta a punta en la terminal
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

Hallazgo medido (simulación, 4 semillas): con el modelo puro, el EV esperado medio de los picks era ≈18% y el real ≈3.8%. Encoger la probabilidad al 50% hacia el mercado sube el EV real a 5.3% y el CLV de 2.6% a 4.0%; limitar los momios a ≤ 4 empeora. Por eso el valor por defecto es `value_model_weight=0.5`. Los parlays del backtest rinden peor que las sencillas, como era de esperar por la varianza y el error compuesto del modelo.

## Conectar una UI (siguiente fase)

```python
import streamlit as st
from predictive_odds.data_fetcher import DataFetcher, SimulatedDataProvider
from predictive_odds.pipeline import AnalysisPipeline, PipelineConfig

sim = SimulatedDataProvider()
result = AnalysisPipeline(DataFetcher(sim, odds_provider=sim), PipelineConfig(league_key="EPL")).run()
st.dataframe(result.predictions_frame())
st.dataframe(result.value_bets_frame())
st.dataframe(result.parlays_frame())
```

## Limitaciones y próximos pasos
- Las **lesiones** se descargan pero todavía no ajustan las λ; por ahora la protección es el filtro de discrepancia con el mercado.
- **UCL:** los equipos de ligas distintas no tienen historial compartido. Falta un modelo multiliga con un coeficiente de fuerza por liga.
- Los momios de un parlay del mismo partido se calculan como producto; un *Bet Builder* real los ajusta por correlación (el sistema lo avisa).
- Scrapers de xG (Understat) como un `ResultsProvider` más.

*Herramienta analítica. Apostar implica riesgo de pérdida; juega con responsabilidad.*
