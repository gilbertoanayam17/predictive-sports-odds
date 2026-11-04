<#
.SYNOPSIS
  Programa (o quita) el análisis automático con alertas de Telegram en el
  Programador de tareas de Windows.

.DESCRIPTION
  Crea la tarea "PredictiveSportsOdds-Alertas", que cada N horas (24 por defecto) ejecuta:
      python main.py --live --leagues <ligas> --alert
  desde la carpeta del proyecto (lee las claves del .env) y guarda la salida en
  data\alertas.log. Solo se envían por Telegram los value bets NUEVOS.

.EXAMPLE
  # Desde la carpeta del proyecto, en PowerShell:
  powershell -ExecutionPolicy Bypass -File scripts\alertas_windows.ps1
  powershell -ExecutionPolicy Bypass -File scripts\alertas_windows.ps1 -Horas 12 -Ligas "EPL LA_LIGA"
  # 5 ligas cada 24 h ≈ 300 créditos/mes (plan gratuito: 500)
  powershell -ExecutionPolicy Bypass -File scripts\alertas_windows.ps1 -Quitar
#>
param(
    [int]$Horas = 24,
    [string]$Ligas = "EPL LA_LIGA SERIE_A BUNDESLIGA LIGA_MX",
    [double]$EvMinimo = 0.05,
    [switch]$Quitar
)

$ErrorActionPreference = "Stop"
$Nombre = "PredictiveSportsOdds-Alertas"
$Proyecto = Split-Path -Parent $PSScriptRoot

if ($Quitar) {
    Unregister-ScheduledTask -TaskName $Nombre -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Tarea '$Nombre' eliminada."
    exit 0
}

$Python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) { $Python = (Get-Command py -ErrorAction SilentlyContinue).Source }
if (-not $Python) { throw "No encuentro python ni py en el PATH." }

if (-not (Test-Path (Join-Path $Proyecto ".env"))) {
    throw "Falta el archivo .env en $Proyecto (copia .env.example y pon tus claves)."
}
New-Item -ItemType Directory -Force -Path (Join-Path $Proyecto "data") | Out-Null

$Log = Join-Path $Proyecto "data\alertas.log"
$Comando = "& '$Python' main.py --live --leagues $Ligas --alert --alert-min-ev $EvMinimo *>> '$Log'"
$Accion = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$Comando`"" -WorkingDirectory $Proyecto
$Disparador = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Hours $Horas)
$Ajustes = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries

Register-ScheduledTask -TaskName $Nombre -Action $Accion -Trigger $Disparador -Settings $Ajustes `
    -Description "Análisis de value bets y alertas por Telegram (predictive-sports-odds)" -Force | Out-Null

$Creditos = 2 * ($Ligas -split '\s+').Count * [math]::Ceiling(24 / $Horas) * 30
Write-Host "Tarea '$Nombre' creada: cada $Horas h, ligas: $Ligas. Registro: $Log"
Write-Host "Consumo máximo estimado de The Odds API: ~$Creditos créditos/mes (0 en ligas sin partidos próximos)."
if ($Creditos -gt 500) {
    Write-Warning "Supera los 500 créditos del plan gratuito: usa más horas (-Horas 24) o menos ligas."
}
Write-Host "Para quitarla: powershell -ExecutionPolicy Bypass -File scripts\alertas_windows.ps1 -Quitar"
