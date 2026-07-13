# TFM: detección de anomalías en radiografías

Comparación de tres modelos entrenados únicamente con radiografías normales:

- autoencoder convolucional (AE);
- autoencoder variacional (VAE);
- GANomaly.

El proyecto sigue un único flujo fácil de explicar:

```text
Chest-RSNA → redimensionado 64x64 → entrenamiento normal
           → umbral de validación → evaluación de test
```

## Resultado

| Modelo | AUROC medio |
|---|---:|
| AE | 0,4566 ± 0,0163 |
| VAE | 0,4506 ± 0,0016 |
| GANomaly | **0,5636 ± 0,0150** |

GANomaly supera AE y VAE, pero un control de gradiente sencillo obtiene 0,5981.
Por tanto, el mejor modelo neuronal todavía depende de señales simples de
textura o adquisición y no puede considerarse un detector clínico.

## Archivos imprescindibles

```text
src/tfm_anomaly/dataset.py     carga y particiones del dataset
src/tfm_anomaly/models.py      AE, VAE y GANomaly
src/tfm_anomaly/experiment.py  entrenamiento y evaluación
src/tfm_anomaly/metrics.py     AUROC, AUPRC y matriz de confusión
scripts/run_experiment.py      ejecución de los experimentos
scripts/evaluate_controls.py   control estadístico sencillo
scripts/audit_dataset.py       comprobación del dataset
scripts/analyze_errors.py      figura de errores
memoria/                       memoria y presentación
reports/                       resultados finales
```

## Dataset

Se utiliza Chest-RSNA/BMAD: 8.000 imágenes normales de entrenamiento, 1.490 de
validación y 17.194 de test. El código encuentra automáticamente la copia local
en el repositorio hermano `TFMv2`. También se puede indicar:

```powershell
$env:TFM_DATA_ROOT = 'D:\datasets\Chest-RSNA'
```

## Ejecución

Desde PowerShell:

```powershell
$python = '..\TFMv2\.tools\python-3.11.9\tools\python.exe'
$env:PYTHONPATH = 'src'

& $python scripts\audit_dataset.py
& $python scripts\run_experiment.py
& $python scripts\evaluate_controls.py
& $python scripts\analyze_errors.py
```

`run_experiment.py` ejecuta por defecto AE, VAE y GANomaly durante 20 épocas
con semillas 13, 42 y 73. Para una prueba rápida:

```powershell
& $python scripts\run_experiment.py --models ae --seeds 42 --epochs 1 `
  --max-train-images 64 --max-eval-images-per-class 32 `
  --output-root artifacts\smoke
```

Los pesos y puntuaciones individuales se guardan bajo `artifacts/`; las tablas
finales se encuentran en `reports/`.

## Alcance

La etiqueta anómala significa ``no normal según RSNA/BMAD''. El sistema no
diagnostica neumonía, no sustituye a profesionales sanitarios y no está
validado para uso clínico.
