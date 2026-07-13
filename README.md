# Detección de anomalías en radiografías mediante AE, VAE y GAN

Proyecto reproducible del Trabajo Fin de Máster del Máster Universitario en
Sistemas Inteligentes (UIB). El objetivo es comparar tres familias generativas
entrenadas **solo con radiografías normales**:

- autoencoder convolucional (AE);
- autoencoder variacional (VAE);
- GANomaly, una GAN con codificador--decodificador--codificador.

La evaluación se realiza a nivel de imagen sobre **Chest-RSNA**, la partición de
radiografías del benchmark BMAD. Los datos anómalos se usan únicamente para
seleccionar el umbral en validación y para la evaluación final, nunca para
ajustar los pesos de las redes.

## Estado y resultado principal

- Dataset localizado y auditado: 26.684 PNG de 1024x1024.
- Campaña final completada: tres modelos, 20 épocas y semillas 13, 42 y 73.
- GANomaly supera AE y VAE: AUROC medio `0.5636 ± 0.0150`, frente a
  `0.4566 ± 0.0163` y `0.4506 ± 0.0016`.
- Un control de gradiente alcanza AUROC `0.5981`; GANomaly es el mejor modelo
  neuronal estudiado, pero no supera una estadística simple de textura.
- Memoria, bibliografía, análisis de errores y resultados CSV preparados.

## Dataset

El código busca Chest-RSNA en este orden:

1. argumento `--data-root`;
2. variable de entorno `TFM_DATA_ROOT`;
3. `data/raw/rsna_bmad/Chest-RSNA`;
4. la copia ya disponible en el repositorio hermano `TFMv2`.

En este equipo ya está disponible la cuarta ruta, por lo que no se duplican
aproximadamente 9 GB. Véase [data/README.md](data/README.md) para obtenerlo desde
cero y para los conteos auditados.

## Ejecución

El intérprete Python portátil ya existente puede reutilizarse en PowerShell:

```powershell
$python = '..\TFMv2\.tools\python-3.11.9\tools\python.exe'
$env:PYTHONPATH = 'src'
& $python scripts\audit_dataset.py
& $python scripts\build_cache.py
.\scripts\run_final.ps1 -Python $python
```

Prueba rápida del pipeline:

```powershell
& $python scripts\run_experiment.py --models ae vae ganomaly --epochs 1 `
  --max-train-images 64 --max-eval-images-per-class 32 `
  --output-root artifacts/smoke
```

Cada modelo guarda configuración, pesos, curvas, puntuaciones de validación y
test, métricas y una cuadrícula de reconstrucciones. El umbral se fija una sola
vez en validación maximizando la exactitud balanceada y se congela para test.
El lanzador final reanuda por modelo y semilla: omite únicamente las ejecuciones
científicas ya completas. También construye una caché local `uint8` del
redimensionado determinista; reduce drásticamente la lectura sin cambiar los
valores que reciben los modelos.

Resultados finales versionables:

- `reports/model_comparison.csv`: nueve ejecuciones individuales;
- `reports/model_comparison_summary.csv`: medias y desviaciones;
- `reports/intensity_controls.csv`: controles simples;
- `reports/error_analysis.json` y `.png`: distribuciones y casos extremos.

## Estructura

```text
configs/                 configuración experimental documentada
src/tfm_anomaly/         datos, modelos, métricas y entrenamiento
scripts/                 auditoría, ejecución y resumen
tests/                   pruebas unitarias y de integración ligera
memoria/                 memoria del TFM en LaTeX
bibliografia/            referencias BibTeX verificadas
reports/                 auditoría y tablas finales versionables
artifacts/               pesos y salidas pesadas, ignorados por Git
```

## Alcance clínico

Este es un estudio metodológico retrospectivo sobre un benchmark público. El
sistema produce una puntuación de rareza respecto del conjunto normal; no
diagnostica neumonía, no sustituye a profesionales sanitarios y no está
validado para uso clínico.
