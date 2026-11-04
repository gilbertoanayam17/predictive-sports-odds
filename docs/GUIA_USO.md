# Guía de uso

Guía práctica para operar el sistema día a día. La arquitectura y las decisiones de diseño están en el [README](../README.md).

> **Importante:** el sistema analiza y sugiere; **nunca coloca apuestas**. Antes de arriesgar dinero real, conviene un periodo de *paper trading* (registrar las sugerencias sin apostar) y comprobar que el **CLV** es positivo de forma sostenida.

## 1. Instalación

```bash
pip install -e ".[dev]"        # núcleo + pruebas
pip install -e ".[ui]"         # opcional: interfaz web (Streamlit)
python -m pytest               # verificación
```

## 2. Primeros pasos con datos reales (en tu computadora)

1. Instala Python 3.10 o superior y clona el repositorio:
   ```bash
   git clone https://github.com/gilbertoanayam17/predictive-sports-odds.git
   cd predictive-sports-odds
   git checkout claude/ecstatic-wright-6nlr98
   pip install -e ".[ui]"
   ```
2. Abre la app con `streamlit run app.py`; se abre en el navegador, en `http://localhost:8501`.
3. **Sin ninguna clave:** elige *Fuente → Real* y pulsa **Ejecutar backtest**. Valida el modelo con la Premier o la Liga MX usando resultados, momios y el cierre de Pinnacle reales de football-data.co.uk.
4. **Clave de momios:** regístrate gratis en [the-odds-api.com](https://the-odds-api.com/#get-access) (plan *Starter*, 500 créditos al mes; la clave llega por correo). Pégala en la barra lateral y pulsa **Probar conexión**, que no gasta créditos.
5. *(Opcional)* Para tener lesiones y la Champions League, crea una clave de [API-Football](https://dashboard.api-football.com/register) (100 peticiones/día gratis).

Para no tener que pegar las claves cada vez, define las variables de entorno `ODDS_API_KEY` y `API_FOOTBALL_KEY` antes de abrir la app. Nunca subas las claves al repositorio.

## 2b. Claves y fuentes de datos

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

**Desde la app** (pestaña *Registro*):
1. Tras analizar la jornada, pulsa **Registrar las apuestas sugeridas**. Se guardan en el archivo SQLite que indica la barra lateral (por defecto `bets.sqlite` en la carpeta del proyecto). Registrar no coloca apuestas y volver a registrar no duplica.
2. Cuando se hayan jugado los partidos, pulsa **Liquidar con resultados**. football-data.co.uk suele publicar los resultados 1-2 días después; con clave de API-Football llegan antes.
3. La misma pestaña muestra el beneficio, el ROI, el historial y la curva de bankroll.

Las apuestas se emparejan con los resultados aunque cada fuente use IDs y nombres distintos ("Hull City" en los momios, "Hull" en los resultados), por equipos y fecha de inicio.

**Desde la terminal:**

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
