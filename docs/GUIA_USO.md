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
5. *(Opcional, de pago)* Una clave de [API-Football](https://dashboard.api-football.com/register) activa el ajuste por lesiones. **Su plan gratuito solo cubre temporadas antiguas (2022–2024)**, así que no sirve para la temporada actual: si no tienes un plan de pago, deja ese campo vacío.

**Para no pegar las claves cada vez**, guárdalas en un archivo `.env` en la carpeta del proyecto (está en `.gitignore`, así que nunca se sube a GitHub):

```powershell
Copy-Item .env.example .env
notepad .env
```

En el archivo, escribe tu clave tras `ODDS_API_KEY=` (sin comillas ni espacios) y guárdalo. La app y el CLI lo leen al arrancar, y la barra lateral muestra "✓ Cargado desde `.env`". Si también defines la variable en el sistema, tiene prioridad sobre el `.env`.

## 2b. Claves y fuentes de datos

Ligas disponibles: **Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Eredivisie, Championship, Primeira Liga, Liga MX y MLS**. Todas tienen histórico gratuito en football-data.co.uk y momios en The Odds API. La Champions League no aparece en la app porque necesitaría API-Football de pago.


| Variable | Obligatoria | Para qué |
|---|---|---|
| `ODDS_API_KEY` | Sí (modo real) | Momios en vivo ([the-odds-api.com](https://the-odds-api.com), 500 créditos/mes gratis) |
| `API_FOOTBALL_KEY` | No (de pago) | Lesiones y calendario ([api-football](https://www.api-football.com)); el plan gratuito no cubre la temporada actual |

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

El `BANKROLL` del script se reparte entre las ligas, para que la exposición conjunta no supere el tope. **Cuota de The Odds API:** cada llamada a momios cuesta *mercados × regiones* créditos: con la configuración por defecto (`h2h,totals` × `eu`) son **2**. Antes se consulta el endpoint gratuito de eventos, y si la liga no tiene partidos en las próximas 96 horas no se gasta nada. Una liga cada 6 horas gasta como mucho ≈ 240 créditos al mes; en la práctica menos, por los parones. La barra lateral de la app muestra los créditos restantes.

### Alertas por Telegram

1. En Telegram, abre **@BotFather**, envía `/newbot` y copia el *token*.
2. Escribe cualquier mensaje a tu bot nuevo.
3. Añade `TELEGRAM_BOT_TOKEN=el_token` a tu `.env` y ejecuta `python main.py --telegram-setup`. El comando encuentra tu `chat_id`, te dice qué línea añadir al `.env` y te manda un mensaje de prueba.
4. Programa el análisis diario con avisos (desde la carpeta del proyecto, en PowerShell):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\alertas_windows.ps1
   ```
   Crea la tarea *PredictiveSportsOdds-Alertas* (cada 24 h, 5 ligas, ≈300 créditos al mes) y guarda la salida en `data\alertas.log`. Opciones: `-Horas 12`, `-Ligas "EPL LA_LIGA"`, `-EvMinimo 0.06`; para quitarla, `-Quitar`.

Solo se avisan los value bets **nuevos** con EV ajustado ≥ 5%; el mismo pick nunca se manda dos veces. El token no aparece en errores ni registros. También puedes avisar a mano: `python main.py --live --leagues EPL LA_LIGA --alert`.

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

## 5a. Apostar en Caliente (o en otra casa sin momios automáticos)

The Odds API no cubre las casas mexicanas como Caliente, así que la app te da el **momio mínimo** de cada selección: si Caliente paga **igual o más**, hay valor.
- Cada boleto de *Value bets* muestra "En Caliente: apuesta si paga ≥ X.XX".
- La pestaña **Mis apuestas** lista el momio mínimo de **todas** las selecciones de la jornada, no solo de los value bets: a veces Caliente paga más que las casas europeas en una selección que allí no tenía valor.
- Cuando apuestes, regístralo en **Mis apuestas → Registrar una apuesta real** con el momio y el monto que tomaste. Se guarda en `mis_apuestas.sqlite`, separado de las sugerencias, y la pestaña *Rendimiento* mide **tu** rendimiento real.
- La casa se configura en la barra lateral o con `MI_CASA=` en el `.env`.

## 5b. Estudio por liga y mercado (recomendado antes de apostar)

En la app (*Fuente → Real*, pestaña **Backtest → Estudio por liga y mercado**) o con `python main.py --live --study`:
- Ejecuta un backtest de varias temporadas en cada liga y muestra, por mercado (1X2, Over/Under, BTTS, DNB, hándicap asiático), el número de apuestas, el ROI y el **CLV**.
- Recomienda un mercado solo si su **CLV medio es positivo** con suficientes apuestas. El ROI de unos cientos de apuestas es casi ruido; el CLV no.
- Guarda el resultado en `league_profiles.json`, y a partir de ahí **la app solo sugiere apuestas en los mercados recomendados** de cada liga (lo indica con un aviso azul). Si una liga no tiene ningún mercado recomendado, no se sugiere nada en ella.
- **Liga MX:** football-data.co.uk solo publica sus momios de cierre, así que el CLV no es comparable; se usa el ROI con el doble de muestra mínima. Tómalo con más cautela.

Repite el estudio cada pocos meses. Para volver a ver todos los mercados, borra `league_profiles.json`.

## 6. Registro de apuestas

**Desde la app** (pestaña *Registro*):
1. Tras analizar la jornada, pulsa **Registrar las apuestas sugeridas**. Se guardan en el archivo SQLite que indica la barra lateral (por defecto `bets.sqlite` en la carpeta del proyecto). Registrar no coloca apuestas y volver a registrar no duplica.
2. Cuando se hayan jugado los partidos, pulsa **Liquidar con resultados**. football-data.co.uk suele publicar los resultados 1-2 días después; con un plan de pago de API-Football llegan antes.
3. La misma pestaña muestra el beneficio, el ROI, el historial y la curva de bankroll.

Las apuestas se emparejan con los resultados aunque cada fuente use IDs y nombres distintos ("Hull City" en los momios, "Hull" en los resultados), por equipos y fecha de inicio.

**Pestaña Rendimiento:** compara tu beneficio real con el **esperado según el cierre**. La diferencia es la **suerte**, medida en desviaciones típicas: entre −2 y +2 es variación normal. Incluye el desglose por liga y por mercado, las rachas y la curva real frente a la esperada. Si el ROI esperado es positivo, tienes ventaja aunque en el corto plazo pierdas. Si es negativo, ganar es suerte.

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
