# Guía de uso

Guía práctica para operar el sistema día a día. La arquitectura y las decisiones de diseño están en el [README](../README.md).

> **Importante:** el sistema analiza y sugiere; **nunca coloca apuestas**. Antes de arriesgar dinero real, conviene un periodo de *paper trading* (registrar las sugerencias sin apostar) y comprobar que el **CLV** es positivo de forma sostenida.

## 1. Instalación

```bash
pip install -e ".[dev]"        # núcleo + pruebas
pip install -e ".[ui]"         # opcional: interfaz web (Streamlit)
python -m pytest               # verificación
```

## 2. Claves y fuentes de datos

| Variable | Obligatoria | Para qué |
|---|---|---|
| `ODDS_API_KEY` | Sí (modo real) | Momios en vivo ([the-odds-api.com](https://the-odds-api.com), 500 créditos/mes gratis) |
| `API_FOOTBALL_KEY` | No | Calendario, lesiones y la UCL ([api-football](https://www.api-football.com), 100 peticiones/día gratis) |

- El histórico de la EPL y la Liga MX viene de football-data.co.uk (gratis, sin clave).
- Sin `API_FOOTBALL_KEY`, el calendario se deduce de los eventos de momios y no hay ajuste por lesiones.
- `--understat` añade xG de Understat (experimental, solo ligas europeas; no se verificó en vivo).

Todas las opciones funcionan **sin red** con datos simulados: basta con omitir `--live`.

## 3. Flujo diario

```bash
# Análisis de una liga con informe HTML
python main.py --live --league EPL --report informe.html

# Varias ligas como una sola jornada (parlays entre ligas y tope de exposición global)
python main.py --live --leagues EPL LIGA_MX --max-singles 12

# Riesgo del staking si se repitiera esta jornada toda una temporada
python main.py --live --league EPL --risk
```

### Automatizado (cron)

`scripts/daily_run.sh` hace, por cada liga, lo siguiente:
- guarda una captura de momios;
- reutiliza o reentrena los modelos;
- genera el informe HTML;
- opcionalmente (`TRACK=1`), registra las sugerencias y liquida las terminadas usando la última captura como línea de cierre.

```bash
SIMULATE=1 TRACK=1 scripts/daily_run.sh EPL LIGA_MX        # ensayo sin red ni claves
# cron, cada 6 horas:
7 */6 * * * cd /ruta/predictive-sports-odds && ODDS_API_KEY=xxx TRACK=1 scripts/daily_run.sh EPL >> data/cron.log 2>&1
```

El `BANKROLL` del script se reparte entre las ligas, para que la exposición conjunta no supere el tope. **Cuota de The Odds API:** cada llamada cuesta *mercados × regiones* créditos. Con la configuración por defecto (`h2h,totals` × `eu,uk`) son 4 créditos, así que una liga cada 6 horas gasta ≈ 480 créditos al mes, casi todo el plan gratuito (500). Con el plan gratuito conviene ejecutar cada 12 horas o usar una sola región (`TheOddsApiProvider(regions="eu")`).

## 4. Cómo leer los resultados

| Columna | Significado |
|---|---|
| `p_modelo` | Probabilidad del modelo (en DNB/hándicap asiático: probabilidad de ganar sin contar el push) |
| `p_mercado` | Probabilidad justa del mercado: consenso entre casas, sin margen |
| `EV_modelo` | Valor esperado con la probabilidad del modelo puro. **Es optimista**: el filtro elige justo donde el modelo más se equivoca |
| `EV_ajustado` | EV con la probabilidad encogida al 50% hacia el mercado. **Es la cifra a mirar** |
| `stake_%` / `stake` | ¼ de Kelly con encogimiento hacia el mercado, topes por apuesta y exposición total ≤ 10% |
| `riesgo` (parlays) | BAJO ≥ 40% de probabilidad conjunta, MEDIO ≥ 22%, ALTO por debajo |

Reglas prácticas:
- Un `EV_ajustado` del 3–8% es razonable. Valores muy altos casi siempre indican información que el modelo no tiene, como lesiones o rotaciones; por eso los picks que discrepan del mercado en más de 15 puntos se descartan.
- Los parlays tienen más varianza y en el backtest rinden peor que las sencillas. Úsalos con stakes pequeños (ya vienen penalizados).
- El **CLV** (apostar mejor que el cierre) es la mejor señal de que el edge es real. El ROI a corto plazo es casi puro ruido: incluso con un CLV de +5%, 1,000 apuestas pueden dar ROI negativo.

## 5. Validar con datos reales antes de apostar

```bash
python main.py --live --league EPL --backtest    # ROI, CLV contra el cierre de Pinnacle, drawdown y calibración
python main.py --live --league EPL --tune        # hiperparámetros con validación cronológica
```

1. El backtest debe mostrar **CLV medio > 0** y la mayoría de apuestas con CLV positivo.
2. Si `--tune` mejora de forma consistente el log-loss (> 0.5%) en varias temporadas, se pueden cambiar los valores por defecto en `PoissonModelConfig` o en el peso del ML.
3. La calibración (`calibration_report`) solo debe activarse si mejora el Brier fuera de muestra.

## 6. Registro de apuestas

```bash
python main.py --live --league EPL --track data/bets.sqlite                    # registrar sugerencias
python main.py --live --league EPL --track data/bets.sqlite --settle \\
               --snapshot data/odds.sqlite                                     # liquidar + CLV propio
```

También se puede consultar desde la pestaña **Registro** de la app (`streamlit run app.py`).

## 7. Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `Error de autenticación` | Falta la clave o no es válida (`ODDS_API_KEY`, `API_FOOTBALL_KEY`) |
| `Límite de peticiones excedido` | Cuota agotada; las respuestas se guardan en `.cache/` para no repetir llamadas |
| `Equipo sin emparejar con el histórico` | Nombre distinto entre fuentes: añade un alias en `DEFAULT_TEAM_ALIASES` (`data_fetcher.py`) |
| `No hay partidos próximos que analizar` | No hay eventos con momios en este momento (parón, fin de temporada) |
| `Datos insuficientes` | Inicio de temporada: añade la temporada anterior (`--season`) |
