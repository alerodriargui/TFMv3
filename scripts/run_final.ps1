param(
    [string]$Python = "..\TFMv2\.tools\python-3.11.9\tools\python.exe",
    [string]$OutputRoot = "artifacts\experiments",
    [string]$CacheRoot = "artifacts\cache\chest_rsna",
    [int]$Epochs = 20
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "No se encuentra el intérprete Python: $Python"
}

$env:PYTHONPATH = (Resolve-Path "src").Path
& $Python scripts\build_cache.py --cache-root $CacheRoot --image-size 64
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo construir la caché de preprocesado"
}
foreach ($seed in 13, 42, 73) {
    foreach ($model in "ae", "vae", "ganomaly") {
        $metrics = Join-Path $OutputRoot "${model}_seed${seed}\metrics.json"
        if (Test-Path -LiteralPath $metrics) {
            $report = Get-Content -LiteralPath $metrics -Raw | ConvertFrom-Json
            if ($report.scientific_run -eq $true) {
                Write-Host "SKIP $model seed=${seed}: ejecución científica ya completa"
                continue
            }
        }
        & $Python scripts\run_experiment.py --models $model --epochs $Epochs `
            --seed $seed --output-root $OutputRoot --cache-root $CacheRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Falló $model con semilla $seed"
        }
    }
}

& $Python scripts\summarize_results.py --root $OutputRoot `
    --output reports\model_comparison.csv
if ($LASTEXITCODE -ne 0) {
    throw "No se pudo generar el resumen final"
}
