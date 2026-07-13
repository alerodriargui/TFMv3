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

## Estado

- Dataset localizado y auditado: 26.684 PNG de 1024x1024.
- Pipeline común de datos, entrenamiento y evaluación implementado.
- AE, VAE y GANomaly implementados en PyTorch sin depender de `torchvision`.
- Memoria LaTeX y bibliografía preparadas en `memoria/` y `bibliografia/`.
- Los resultados de humo sirven para verificar el software; los resultados
  científicos finales deben proceder de `scripts/run_experiment.py` sin límites
  de muestras y con las semillas indicadas en el protocolo.

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
científicas ya completas.

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
