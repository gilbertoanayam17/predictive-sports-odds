# Fase 1 — Reflexión e investigación inicial

## 1. Fuentes de datos

### Necesidades
| Necesidad | Frecuencia | Criticidad |
|---|---|---|
| Resultados históricos (2-5 temporadas) | Diaria/semanal | Alta (entrena los modelos) |
| Calendario próximo | Diaria | Alta |
| Momios actuales de varias casas | Minutos/horas antes del partido | Crítica (define el valor) |
| xG por partido | Semanal | Media (mejora la señal, no es imprescindible) |
| Lesiones/alineaciones | Horas antes del partido | Media-alta (principal fuente de error del modelo) |

### Opciones comparadas

| Fuente | Tipo | Ventajas | Desventajas |
|---|---|---|---|
| **The Odds API** | API freemium (500 créditos/mes) | Momios de decenas de casas en JSON normalizado; EPL, UCL y Liga MX; legal y estable | Cuota pequeña; BTTS solo por evento (más caro); sin histórico en el plan gratuito |
| **API-Football (api-sports)** | API freemium (100 req/día) | Calendario, resultados, estadísticas por partido (incluye `expected_goals` en ligas grandes), lesiones; cubre Liga MX y UCL | xG requiere 1 petición por partido; errores devueltos con HTTP 200 |
| **football-data.co.uk** | CSV gratuito | 20+ años de resultados **con momios de cierre** (Pinnacle, Bet365, promedio) — perfecto para *backtesting*; sin API key; formato muy estable | Sin xG, sin calendario futuro; UCL no disponible |
| **FBref / Understat** | Scraping | xG/xGA de calidad, gratis | Frágil (cambios de HTML, Cloudflare, límites agresivos), zona gris de términos de uso; FBref ha recortado sus métricas avanzadas de proveedor; Understat no cubre Liga MX |

### Decisión (corto plazo)
**APIs + CSV, sin scrapers en la ruta crítica.**

1. **Histórico:** `football-data.co.uk` (EPL, Liga MX) → gratis, estable y con momios de cierre para medir el *Closing Line Value*. API-Football como respaldo y para UCL.
2. **Calendario + lesiones + xG:** API-Football, con presupuesto de llamadas (`enrich_with_xg(max_calls=...)`) y caché en disco.
3. **Momios en vivo:** The Odds API (mercados `h2h` y `totals`; `btts` opcional). Si no hay API-Football, el calendario se **deriva de los propios eventos de momios** (`DataFetcher.fixtures_from_odds`), así el sistema funciona con una sola API key.
4. **Scrapers (Understat/FBref):** fuera de esta iteración. La interfaz `ResultsProvider` permite añadirlos después sin tocar los modelos. El modelo ya acepta xG faltante (usa goles).

Justificación: lo que más rompe un sistema de apuestas no es la falta de una métrica avanzada sino los datos que dejan de llegar a mitad de temporada. Las APIs ofrecen contratos estables, cuotas conocidas y errores tipados; el CSV da un histórico gratuito y verificable. Toda la ingesta pasa por un `HttpClient` con timeouts, reintentos con backoff, `Retry-After`, *throttling* y caché, y por validadores de esquema que descartan filas corruptas en vez de propagar basura a los modelos.

Para desarrollo y CI se incluye `SimulatedDataProvider`: ligas sintéticas con fuerzas latentes, xG ruidoso y casas con margen y error propio. Al conocer la "verdad" permite comprobar que los picks marcados como valor tienen EV real positivo.

## 2. Features clave por mercado

Evidencia general de la literatura (Maher 1982; Dixon & Coles 1997; Constantinou & Fenton; trabajos sobre xG y modelos de rating) y de la práctica de mercado:

| Feature | 1X2 | Over/Under | BTTS | Notas |
|---|---|---|---|---|
| **Fuerza ataque/defensa ajustada por rival** (Poisson/Dixon-Coles) | ★★★ | ★★★ | ★★★ | El núcleo: produce la distribución completa de marcadores |
| **xG / xGA** (por partido, media móvil) | ★★★ | ★★★ | ★★★ | Más estable que los goles: separa calidad de ocasiones del azar en la definición. Mezclado con goles (peso 0.4) |
| **Rating dinámico (Elo)** | ★★★ | ★ | ★ | Resume la fuerza con ajuste por rival y margen; base de la "fuerza del calendario" |
| **Ventaja de local** | ★★★ | ★★ | ★ | Se estima por liga (no es constante: varía mucho entre ligas) |
| **Forma reciente (5 partidos, general y por sede)** | ★★ | ★★ | ★★ | Útil pero ruidosa; el mercado la sobrevalora. Se usa con regularización |
| **Fuerza del calendario (nivel del rival)** | ★★ | ★ | ★ | Corrige la forma: 3 victorias contra el fondo de la tabla ≠ contra el top 4 |
| **Posición/PPG en la tabla** | ★★ | ★ | ★ | Redundante con ratings, aporta contexto de motivación |
| **H2H** | ★ | ★ | ★ | Débil (muestras pequeñas, plantillas cambian). Se usa con poco peso |
| **Decaimiento temporal** | ★★★ | ★★★ | ★★★ | Partidos recientes pesan más (vida media 240 días) |
| **Lesiones/alineaciones** | ★★★ | ★★ | ★★ | Muy predictiva pero difícil de cuantificar; en esta iteración se **protege** contra ella: si el modelo discrepa del mercado > 15 pp, el pick se descarta (el mercado probablemente sabe algo) |
| **Momios del mercado** | ★★★ | ★★★ | ★★★ | El mejor predictor individual. Se usan para quitar margen, como prior en el Kelly modificado y como referencia de calibración |

Conclusiones aplicadas al diseño:
- **Poisson / Dixon-Coles** como modelo principal: con una sola matriz de marcadores se derivan 1X2, doble oportunidad, Over/Under de cualquier línea, BTTS y marcador exacto, **y las correlaciones entre ellos** (clave para parlays del mismo partido).
- **Regresión logística** (bien calibrada por construcción) sobre features de contexto (Elo, forma por sede, xG, calendario, H2H) para 1X2, OU 2.5 y BTTS; se mezcla con Poisson (peso 0.35). Random Forest disponible como alternativa.
- **Validación cronológica** (sin fuga de información) contra una referencia ingenua: un modelo que no bate a las frecuencias históricas no debe apostar.
