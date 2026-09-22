# DAE para deteccion no supervisada de anomalias

Este proyecto entrena un unico modelo para detectar anomalias en cortes cerebrales
FLAIR de Brain-AD / BraTS2021: un **Denoising Autoencoder (DAE)**.

La implementacion esta basada en Kascenas, Pugeault y O'Neil,
*Denoising Autoencoders for Unsupervised Anomaly Detection in Brain MRI*
(MIDL 2022), pero adapta el metodo a deteccion binaria a nivel de imagen
utilizando unicamente la modalidad FLAIR.

El pipeline actual es:

1. Entrena solo con imagenes FLAIR normales.
2. Anade coarse Gaussian noise de baja resolucion al primer plano.
3. Reconstruye la imagen limpia con una U-Net de tres reducciones y conexiones skip.
4. En inferencia no se anade ruido.
5. Calcula el error absoluto entre original y reconstruccion.
6. Aplica mascara de foreground y filtro de mediana 5x5.
7. Reduce el mapa de error a un unico score mediante el maximo espacial.
8. Selecciona el threshold en validacion mediante Youden J.
9. Evalua en test con AUROC y balanced accuracy.

### Diferencias principales respecto a Kascenas et al.

- Nuestro modelo recibe unicamente FLAIR (1 canal), no las cuatro modalidades MRI combinadas.
- El decoder usa upsampling bilinear seguido de Conv2d.
- No se aplica weight standardization en las convoluciones.
- El optimizador es Adam con AMSGrad y weight_decay=1e-5.
- El scheduler actual es CosineAnnealingLR con T_max=100.
- El objetivo final es clasificar cada corte como normal o anomalo a nivel de imagen.
- El mapa de error se convierte en un score por imagen usando su maximo espacial.
- El threshold se obtiene exclusivamente sobre validacion mediante el estadistico de Youden.

Articulo: https://proceedings.mlr.press/v172/kascenas22a.html

## Guia visual del proyecto

La explicacion interactiva y visual del proyecto esta en
[`web/index.html`](web/index.html). Se puede abrir directamente en el navegador;
no requiere instalar dependencias ni ejecutar un servidor.

## Demo interactiva del modelo

La demo carga el checkpoint congelado, permite elegir entre seis cortes o subir
una imagen propia y muestra la reconstruccion, el mapa de residuo y el score:

```powershell
python -m tfm_ae.demo_server
```

Abre `http://127.0.0.1:8000`. Las imagenes subidas se procesan en memoria, no se
guardan, y los pesos permanecen en modo evaluacion durante toda la inferencia.

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
python -m tfm_ae.train --data-root D:\datasets\BraTS2021_slice
```

Parametros principales:

```text
--image-size          Resolucion de entrada, multiplo de 8 (def: 128)
--batch-size          Tamano del lote (def: 16)
--epochs              Numero de epocas (def: 50)
--seeds               Semillas del experimento (def: 42; una ejecucion)
--dae-base-ch         Canales base del DAE (def: 64)
--noise-sigma         Desviacion del ruido (def: 0.2)
--noise-resolution    Resolucion del ruido antes de interpolar (def: 16)
```

La configuracion actual usa batch 16, Adam con AMSGrad y weight_decay=`1e-5`,
learning rate inicial `1e-4`, CosineAnnealingLR con `T_max=100`, y ruido
gaussiano coarse de `16x16` con sigma `0.2`. La ejecucion predeterminada
entrena una unica semilla durante 50 epochs y conserva el checkpoint con menor
perdida de validacion sobre imagenes normales.

Cada ejecucion guarda `model.pt`, `metrics.json`, los scores de validacion y
test, y una imagen con reconstrucciones en `results/experiments/dae_seed{N}/`.
Las unicas metricas finales son AUROC y balanced accuracy. El umbral se elige
con validacion y se guarda por separado para aplicarlo despues sobre test.

## Google Colab

El notebook [TFMv3_colab_brain.ipynb](notebooks/TFMv3_colab_brain.ipynb)
descarga los datos, valida la GPU, ejecuta una prueba reducida y entrena el DAE.
