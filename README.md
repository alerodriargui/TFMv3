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
Las metricas por imagen son AUROC y balanced accuracy; la evaluacion espacial anade Dice, IoU y sensibilidad. El umbral se elige
con validacion y se guarda por separado para aplicarlo despues sobre test.

## Google Colab

El notebook [TFMv3_colab_brain.ipynb](notebooks/TFMv3_colab_brain.ipynb)
descarga los datos, valida la GPU, ejecuta una prueba reducida y entrena el DAE.

## Evaluacion de localizacion con mascaras

Validacion y test incluyen pares `good/img/<nombre>.png` / `good/label/<nombre>.png`
y `Ungood/img/<nombre>.png` / `Ungood/label/<nombre>.png`. Las mascaras de referencia
no se usan para entrenar el DAE. Se ignora su canal alfa y se considera lesion
cualquier pixel RGB mayor que cero; se redimensionan con vecino mas cercano.

Para evaluar el checkpoint existente sin reentrenar:

```powershell
python -m tfm_ae.localization --data-root data/raw/rsna_bmad/BraTS2021_slice
python -m tfm_ae.demo_server
```

En la demo, selecciona un corte en **Corte con referencia** y pulsa **Ejecutar modelo**.
Se muestran la referencia, la prediccion, su superposicion, Dice, IoU y sensibilidad.
Verde indica solo referencia, naranja solo prediccion y cian coincidencia.
Las imagenes propias sin referencia no reciben metricas de coincidencia.

La evaluacion elige un umbral por pixel maximizando Dice micro en toda la
validacion sobre una rejilla fija de paso 1/4096; despues lo aplica sin cambios
a test. Es independiente del umbral de clasificacion por imagen. Se evaluan todos
los pixeles a la resolucion del modelo, incluidos los del fondo y las lesiones
que el filtro de foreground pudiera suprimir. `localization.json` guarda el
umbral y los resultados; `valid_localization.csv` y `test_localization.csv`
guardan TP, FP, FN y metricas por corte. Tambien se ejecuta en nuevos entrenamientos.

Dice micro e IoU micro agregan los conteos de todos los cortes, incluidos los
normales para penalizar sus falsos positivos. Las medias por corte se calculan
solo sobre referencias con lesion. Dos mascaras vacias tienen Dice/IoU 1;
la sensibilidad sin lesion es indefinida (`null` en la demo, celda vacia en CSV).
Estos resultados son por corte/pixel a resolucion reducida, no por paciente ni
una evaluacion volumetrica. La procedencia de las anotaciones debe documentarse
antes de atribuir las referencias a expertos.

Las pruebas reducidas de entrenamiento respetan `--max-eval-images-per-class`
tambien en la evaluacion espacial y lo registran en el informe; sus resultados
no deben presentarse como evaluacion completa.

En el dataset local, seis cortes de test etiquetados como anomalos tienen
referencia vacia despues de reducir a 128x128. Se registran en
`anomalous_empty_reference_images`; siguen contando en las metricas micro,
pero no en las medias restringidas a referencias con lesion visible.

Para otra ubicacion de los datos, pasa tambien `--data-root RUTA` al servidor
de la demo. La calibracion se valida contra una huella de los pesos del modelo
para impedir mezclar umbrales y checkpoints diferentes.
