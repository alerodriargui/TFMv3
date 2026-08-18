# TFM: detección no supervisada de anomalías

Sistema no supervisado para detectar anomalías en imágenes médicas aprendiendo
únicamente con muestras normales. El modelo final es un autoencoder
convencional de 16.281 parámetros que reconstruye cortes cerebrales de
240×240 (BraTS2021). La puntuación combina el error de reconstrucción
calibrado con señales globales de la imagen (kurtosis, región brillante más
grande, gradiente medio, entropía), con un peso calibrado en validación.

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Estructura del proyecto

```text
TFMv3/
|-- tfm_ae/              Código ejecutable
|   |-- models.py        Arquitecturas (ConvAutoencoder)
|   |-- data.py          Carga determinista + data augmentation
|   |-- features.py      Señales globales de imagen
|   |-- metrics.py       AUROC, umbral y métricas
|   |-- experiment.py    Entrenamiento, calibración e inferencia
|   |-- train.py         Comando de entrenamiento
|   `-- demo_tfm.py      Demo visual para el TFM
|-- checkpoints/         Pesos congelados
|-- results/             Métricas, experimentos y figuras
|   |-- brain_v2_final/  Resultados (Colab GPU, 20 épocas + augmentation)
|   `-- brain_hybrid_full/ Resultados locales (3 épocas)
|-- notebooks/           Notebooks de Google Colab
|-- docs/                Memoria en LaTeX y recursos
|-- data/raw/rsna_bmad/  Datos locales (BraTS2021)
`-- requirements*.txt    Dependencias
```

## Datos

El código soporta el dataset **BraTS2021_slice** (cortes cerebrales FLAIR):
- 7.500 entrenamiento (normales)
- 39 normales + 44 anómalos validación
- 640 normales + 3.075 anómalos test

Para indicar su ubicación:

```powershell
$env:TFM_DATA_ROOT = 'D:\datasets\BraTS2021_slice'
```

## Modelos

| Modelo | Parámetros | Resolución | Uso |
|---|---:|---:|---|
| `ae` (ConvAutoencoder) | 16.281 | 64–256 | Modelo principal |

El AE simple tiene tres etapas convolucionales (1→8→16→32 canales) con
decodificador simétrico.

## Data augmentation

El entrenamiento aplica automáticamente transformaciones aleatorias para mejorar
la generalización:

- Flip horizontal (50%)
- Flip vertical (50%)
- Rotación 90/180/270° (aleatoria)

Definido en `data.py` como `RandomFlipRotate`.

## Puntuación

El experimento usa un único modo de puntuación híbrido:

- **hybrid**: `w × señales globales calibradas + (1-w) × MAE calibrado`

Las señales globales (`features.py`) son cuatro estadísticos calculados sobre
cada imagen: kurtosis, tamaño de la región brillante más grande, gradiente medio
y entropía. Cada señal se tipifica con la media y desviación de las normales de
entrenamiento; su signo se fija por la dirección del AUROC en validación. El
peso `w` se busca en validación con una rejilla de 21 valores entre 0 y 1.

## Ejecución

```powershell
python -m tfm_ae.train --model ae --image-size 240 --epochs 20 --score-mode hybrid
```

Argumentos principales:

```text
--data-root         Ruta al dataset
--model             ae
--image-size        Resolución (64, 240, 256, 512...)
--epochs            Número de épocas
--batch-size        Tamaño del lote (def: 32)
--score-mode        hybrid
--seeds             Semillas para la campaña (def: 13 42 73)
--bottleneck        Canales del cuello de botella (def: 32)
--max-train-images  Limita imágenes de entrenamiento (debug)
--max-eval-images-per-class  Limita imágenes de evaluación (debug)
```

El entrenamiento guarda en `results/experiments/ae_seed{N}/`:
- `model.pt` — pesos del modelo
- `metrics.json` — configuración, historial, calibración y métricas
- `validation_scores.csv` / `test_scores.csv` — puntuaciones por imagen
- `reconstructions.png` — reconstrucciones de ejemplo

### Demo visual para el TFM

```powershell
python -m tfm_ae.demo_tfm
```

Genera figuras con 6 paneles explicativos para cada imagen de prueba:

1. **Imagen original** — corte cerebral preprocesado
2. **Reconstrucción AE** — salida del autoencoder
3. **Mapa de error** — heatmap del error absoluto
4. **Regiones anómalas** — overlay con zonas de mayor error
5. **Señales globales** — valores de kurtosis, correlación, gradiente, entropía
6. **Gauge del score** — puntuación frente al umbral de decisión

## Resultados

### Cerebro (BraTS2021, score híbrido, 20 épocas + data augmentation)

| Métrica | Valor |
|---|---|
| AUROC | 0,9029 |
| Balanced accuracy | 0,8332 |
| Sensitivity | 75,5% |
| Specificity | 91,1% |
| Peso híbrido (w) | 0,95 |
| Épocas | 20 |
| Resolución | 240×240 |

El peso híbrido seleccionado en validación es `w = 0,95`: las señales globales
dominan la separación y el MAE actúa como señal complementaria.

## Google Colab

El notebook `notebooks/TFMv3_colab_brain.ipynb` reproduce el experimento en Colab
con GPU. Descarga BraTS2021, ejecuta una prueba reducida y lanza la campaña
completa con respaldo incremental en Drive.

1. Abre el notebook en Colab.
2. Selecciona `Runtime > Change runtime type > GPU`.
3. Ejecuta todas las celdas.

## Entregables

| Ruta | Función |
|---|---|
| `docs/memoria.pdf` | Memoria final (LaTeX) |
| `docs/assets/` | Diagramas y recursos |
| `notebooks/TFMv3_colab_brain.ipynb` | Reproducible en Colab |
| `results/brain_v2_final/` | Resultados + demos visuales |
| `tfm_ae/demo_tfm.py` | Script de demo para el TFM |
