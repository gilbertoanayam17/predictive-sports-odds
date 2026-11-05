<#
.SYNOPSIS
  Programa (o quita) el ciclo automático en el Programador de tareas de Windows.

.DESCRIPTION
  Crea la tarea "PredictiveSportsOdds-Ciclo", que ejecuta desde la carpeta del
  proyecto (lee las claves del .env):
      python main.py --live --weekly --leagues <ligas>
  El ciclo liquida las apuestas pendientes (sugerencias y reales), analiza las
  ligas, registra las sugerencias y manda por Telegram el resumen y los picks
  con el momio mínimo en tu casa (MI_CASA, Caliente por defecto).
  La salida se guarda en data\ciclo.log.

  Créditos de The Odds API: ~2 por liga con partidos en los próximos 4 días.
  Semanal con 10 ligas ≈ 90 créditos/mes (plan gratuito: 500).

.EXAMPLE
  # Desde la carpeta del proyecto, en PowerShell:
  powershell -ExecutionPolicy Bypass -File scripts\automatizar_windows.ps1
  powershell -ExecutionPolicy Bypass -File scripts\automatizar_windows.ps1 -Dia Thursday -Hora 18:00
  powershell -ExecutionPolicy Bypass -File scripts\automatizar_windows.ps1 -Ligas "EPL LA_LIGA LIGA_MX"
  powershell -ExecutionPolicy Bypass -File scripts\automatizar_windows.ps1 -Frecuencia Diaria
  powershell -ExecutionPolicy Bypass -File scripts\automatizar_windows.ps1 -Quitar
#>
param(
    [ValidateSet("Semanal", "Diaria")][string]$Frecuencia = "Semanal",
    [ValidateSet("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")]
    [string]$Dia = "Friday",
    [string]$Hora = "10:00",
    [string]$Ligas = "EPL LA_LIGA SERIE_A BUNDESLIGA LIGUE_1 EREDIVISIE CHAMPIONSHIP PRIMEIRA_LIGA LIGA_MX MLS",
    [double]$Bankroll = 1000,
    [switch]$Quitar
)

$ErrorActionPreference = "Stop"
$Nombre = "PredictiveSportsOdds-Ciclo"
$Proyecto = Split-Path -Parent $PSScriptRoot

# La versión anterior del script creaba otra tarea: se elimina para no duplicar.
Unregister-ScheduledTask -TaskName "PredictiveSportsOdds-Alertas" -Confirm:$false -ErrorAction SilentlyContinue

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

$Log = Join-Path $Proyecto "data\ciclo.log"
$Comando = "& '$Python' main.py --live --weekly --leagues $Ligas --bankroll $Bankroll *>> '$Log'"
$Accion = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -Command `"$Comando`"" -WorkingDirectory $Proyecto
$Momento = [datetime]::ParseExact($Hora, "HH:mm", $null)
if ($Frecuencia -eq "Semanal") {
    $Disparador = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Dia -At $Momento
    $Veces = 4.3
} else {
    $Disparador = New-ScheduledTaskTrigger -Daily -At $Momento
    $Veces = 30
}
$Ajustes = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries

Register-ScheduledTask -TaskName $Nombre -Action $Accion -Trigger $Disparador -Settings $Ajustes `
    -Description "Ciclo de value bets: liquidar, analizar, registrar y avisar por Telegram" -Force | Out-Null

$Creditos = [math]::Round(2 * ($Ligas -split '\s+').Count * $Veces)
$Cuando = if ($Frecuencia -eq "Semanal") { "cada $Dia a las $Hora" } else { "cada día a las $Hora" }
Write-Host "Tarea '$Nombre' creada: $Cuando. Ligas: $Ligas"
Write-Host "Registro de ejecución: $Log"
Write-Host "Consumo máximo estimado de The Odds API: ~$Creditos créditos/mes (0 en ligas sin partidos próximos)."
if ($Creditos -gt 500) {
    Write-Warning "Supera los 500 créditos del plan gratuito: usa -Frecuencia Semanal o menos ligas."
}
Write-Host "Para probarla ya: Start-ScheduledTask -TaskName $Nombre"
Write-Host "Para quitarla:    powershell -ExecutionPolicy Bypass -File scripts\automatizar_windows.ps1 -Quitar"
