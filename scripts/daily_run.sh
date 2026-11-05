#!/usr/bin/env bash
# Ejecución diaria (pensada para cron): captura de momios, análisis, informe y liquidación.
#
# Uso:   scripts/daily_run.sh [LIGA ...]          (por defecto: EPL LIGA_MX)
# Requiere: ODDS_API_KEY (y API_FOOTBALL_KEY para UCL o calendario/lesiones).
# Variables opcionales:
#   DATA_DIR=data      carpeta de bases de datos, modelos e informes
#   BANKROLL=1000      bankroll total; se reparte a partes iguales entre las ligas para que la
#                      exposición conjunta no supere el tope (cada análisis aplica su propio 10%)
#   TRACK=0            1 = registrar las sugerencias en el registro de apuestas (paper trading)
#   PYTHON=python      intérprete
#   SIMULATE=0         1 = ensayo con datos simulados (sin red ni API keys)
#
# Ejemplo de cron (cada 6 horas):
#   7 */6 * * * cd /ruta/predictive-sports-odds && ODDS_API_KEY=xxx scripts/daily_run.sh EPL >> data/cron.log 2>&1
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
DATA_DIR="${DATA_DIR:-data}"
BANKROLL="${BANKROLL:-1000}"
TRACK="${TRACK:-0}"
PYTHON="${PYTHON:-python}"
SIMULATE="${SIMULATE:-0}"
LIVE=(--live)
[ "$SIMULATE" = "1" ] && LIVE=()
LEAGUES=("$@")
[ ${#LEAGUES[@]} -eq 0 ] && LEAGUES=(EPL LIGA_MX)

if [ "$SIMULATE" != "1" ] && [ -z "${ODDS_API_KEY:-}" ]; then
  echo "ERROR: falta ODDS_API_KEY" >&2
  exit 3
fi

mkdir -p "$DATA_DIR/reports" "$DATA_DIR/models"
LEAGUE_BANKROLL="$("$PYTHON" -c "import sys; print(round(float(sys.argv[1]) / int(sys.argv[2]), 2))" "$BANKROLL" "${#LEAGUES[@]}")"
STAMP="$(date -u +%Y%m%d_%H%M)"
STATUS=0

for LEAGUE in "${LEAGUES[@]}"; do
  echo "== $(date -u +%FT%TZ) $LEAGUE =="
  ARGS=("${LIVE[@]}" --league "$LEAGUE" --bankroll "$LEAGUE_BANKROLL"
        --snapshot "$DATA_DIR/odds.sqlite"
        --model-path "$DATA_DIR/models/$LEAGUE.joblib"
        --report "$DATA_DIR/reports/${LEAGUE}_${STAMP}.html")
  [ "$TRACK" = "1" ] && ARGS+=(--track "$DATA_DIR/bets.sqlite")
  if ! "$PYTHON" main.py "${ARGS[@]}" > "$DATA_DIR/reports/${LEAGUE}_${STAMP}.txt"; then
    echo "AVISO: el análisis de $LEAGUE falló (ver ${LEAGUE}_${STAMP}.txt)" >&2
    STATUS=1
  fi
  if [ "$TRACK" = "1" ]; then
    "$PYTHON" main.py "${LIVE[@]}" --league "$LEAGUE" --bankroll "$BANKROLL" --track "$DATA_DIR/bets.sqlite" \
      --settle --snapshot "$DATA_DIR/odds.sqlite" || STATUS=1
  fi
done
exit $STATUS
