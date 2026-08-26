# DAE para deteccion no supervisada de anomalias

Este proyecto entrena un unico modelo para detectar anomalias en cortes cerebrales
FLAIR de Brain-AD / BraTS2021: un **Denoising Autoencoder (DAE)**.

La implementacion esta basada en Kascenas, Pugeault y O'Neil,
*Denoising Autoencoders for Unsupervised Anomaly Detection in Brain MRI*
(MIDL 2022). El metodo es deliberadamente sencillo:

1. Entrena solo con imagenes normales.
2. Anade ruido gaussiano de baja resolucion al primer plano.
3. Reconstruye la imagen limpia con una red tipo U-Net y conexiones skip.
4. Usa el error absoluto de reconstruccion como puntuacion de anomalia.

Articulo: https://proceedings.mlr.press/v172/kascenas22a.html

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Datos

El dataset debe contener esta estructura:

```text
BraTS2021_slice/
|-- train/good/
|-- valid/good/
|-- valid/Ungood/
|-- test/good/
`-- test/Ungood/
```

La ruta se pasa con `--data-root`.

## Ejecucion

```powershell
python -m tfm_ae.train --data-root D:\datasets\BraTS2021_slice --epochs 100
```

Parametros principales:

```text
--image-size          Resolucion de entrada, multiplo de 16 (def: 224)
--batch-size          Tamano del lote (def: 16)
--seeds               Semillas del experimento (def: 42)
--dae-base-ch         Canales base del DAE (def: 64)
--noise-sigma         Desviacion del ruido (def: 0.2)
--noise-resolution    Resolucion del ruido antes de interpolar (def: 16)
```

Cada ejecucion guarda `model.pt`, `metrics.json`, los scores de validacion y
test, y una imagen con reconstrucciones en `results/experiments/dae_seed{N}/`.
Las unicas metricas finales son AUROC y balanced accuracy. El umbral se elige
con validacion y se guarda por separado para aplicarlo despues sobre test.

## Google Colab

El notebook [TFMv3_colab_brain.ipynb](notebooks/TFMv3_colab_brain.ipynb)
descarga los datos, valida la GPU, ejecuta una prueba reducida y entrena el DAE.
