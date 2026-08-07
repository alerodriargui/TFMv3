# TFM: detección no supervisada de anomalías

Sistema sencillo para detectar radiografías anómalas aprendiendo únicamente
con imágenes normales. El modelo final es un autoencoder tipo U-Net de 366.433
parámetros que reconstruye radiografías de 512×512 con conexiones de salto.

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Estructura del proyecto

```text
TFMv3/
|-- tfm_ae/            Código ejecutable del autoencoder
|   |-- models.py      Arquitectura de la red
|   |-- data.py        Lectura y preparación de radiografías
|   |-- experiment.py  Entrenamiento, calibración y evaluación
|   |-- metrics.py     Métricas y selección del umbral
|   |-- train.py       Comando de entrenamiento
|   `-- demo.py        Evaluación de una sola imagen
|-- checkpoints/       Pesos congelados del modelo
|-- results/           Métricas, experimentos y figuras generadas
|-- notebooks/         Notebook para Google Colab
|-- docs/              Memoria, bibliografía y recursos
|-- README.md          Guía del proyecto
`-- requirements*.txt Dependencias
```

La raíz contiene únicamente los archivos habituales de configuración. Cada
carpeta agrupa elementos con una responsabilidad concreta.

## Dataset

Se utiliza RSNA Pneumonia Detection Challenge con la reorganización de
[BMAD](https://github.com/DorisBao/BMAD). `Chest-AD.zip` se descarga desde el
Google Drive enlazado por BMAD, aceptando las condiciones de la
[fuente RSNA](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018).

```text
Chest-RSNA/
├── train/good/
├── valid/good/
├── valid/Ungood/
├── test/good/
└── test/Ungood/
```

La copia empleada contiene 8.000 imágenes de entrenamiento, 1.490 de
validación y 17.194 de test. Para indicar su ubicación:

```powershell
$env:TFM_DATA_ROOT = 'D:\datasets\Chest-RSNA'
```

## Modelo final

```text
radiografía 512x512
      ↓  Conv 3×3, s2 (1→16)
256x256x16 ─────────────┐
      ↓  Conv 3×3, s2   │ salto (concat)
128x128x32 ───────────┐ │
      ↓  Conv 3×3, s2  │ │
64x64x64 ───────────┐ │ │
      ↓  Conv 3×3, s2 │ │ │
32x32x128 (cuello de botella) ─→ ConvT 4×4, s2 + sigmoid → reconstrucción
```

El diagrama completo está en `docs/assets/unet_architecture.png`. Los pesos se
ajustan exclusivamente con las 8.000 radiografías normales. La puntuación
final promedia dos señales tipificadas:

```text
0,5 × error de reconstrucción calibrado
+ 0,5 × diferencia de intensidad centro–borde
```

Validación selecciona la dirección de ambas señales y el umbral; nunca se usa
para actualizar los pesos. El test se evalúa con modelo, calibración y umbral
congelados.

## Ejecución

```powershell
python -m tfm_ae.train
```

El orden interno del entrenamiento es:

```text
train.py
   │
   ├── data.py
   │     Carga train, valid y test
   │
   └── experiment.py
         │
         ├── models.py
         │     Construye y entrena el autoencoder
         │
         ├── data.py
         │     Entrega los lotes de imágenes
         │
         ├── metrics.py
         │     Calcula umbral, AUROC y balanced accuracy
         │
         └── Guarda resultados
```

El AE usa tres épocas, batch 32 y Adam con tasa `1e-3`.

### Ejecución en Google Colab con GPU

El notebook `notebooks/TFMv3_colab.ipynb` reproduce el experimento principal en una
runtime GPU de Google Colab. Incluye clonación del repositorio, instalación de
dependencias de apoyo, comprobación de CUDA/PyTorch, descarga y extracción del
dataset en el entorno de Colab, conteos del dataset, prueba reducida y ejecución
completa del AE U-Net con la semilla 42.

Para usarlo:

1. Abre `notebooks/TFMv3_colab.ipynb` en Colab.
2. Selecciona `Runtime > Change runtime type > GPU`.
3. Ejecuta todas las celdas. El dataset (`Chest-AD.zip`, ~9,6 GB) se descarga y
   extrae automáticamente en `/content`, sin necesidad de Google Drive. Los
   resultados se guardan en `/content/TFMv3_colab_outputs/`.

La campaña completa respalda de forma incremental en Google Drive
(`MyDrive/TFMv3_colab_backup/<fecha>/`): tras cada semilla copia resultados y
checkpoint. Si la sesión se desconecta por inactividad, los artefactos pueden
recuperarse desde esa carpeta sin volver a entrenar. Al terminar, la última
celda descarga un zip con todo.

Para evaluar una radiografía con el autoencoder congelado:

```powershell
python -m tfm_ae.demo D:\radiografias\ejemplo.png
```

La demo usa por defecto `checkpoints/modelo_autoencoder.pt` y guarda la figura
en `results/demo_resultado.png`. Ambas rutas se pueden cambiar con `--model` y
`--output`.

La consola conserva el error de reconstrucción, la puntuación, el umbral y la
clase. Además, el PNG generado contiene cuatro paneles:

1. imagen original preprocesada a 512×512;
2. reconstrucción producida por el autoencoder;
3. mapa de error absoluto por píxel, con escala de color y MAE;
4. puntuación de anomalía comparada con el umbral congelado de validación.

Puede añadirse `--show` para abrir la figura en un entorno gráfico. Sin esa
opción se usa un backend sin interfaz y el PNG se genera igualmente. La
reconstrucción es la aproximación aprendida por el autoencoder, no una
radiografía clínicamente normal garantizada. La demo es una herramienta
experimental y no constituye un diagnóstico.

## Resultados

| Modelo | AUROC | Balanced accuracy |
|---|---:|---:|
| Control de gradiente | 0,5981 | 0,5894 |
| AE mínimo 64×64 | 0,7608 | 0,6790 |
| **U-Net final 512×512** | **0,6574** | **0,6224** |

El resultado del modelo final corresponde a la semilla 42 con test congelado.
No se utiliza ningún clasificador supervisado, ensamble ni red preentrenada.

## Entregables

| Ruta | Función |
|---|---|
| `docs/memoria.pdf` | Memoria final |
| `docs/assets/` | Diagramas, imágenes y recursos de la memoria |
| `notebooks/TFMv3_colab.ipynb` | Ejecución reproducible en Colab |
| `checkpoints/modelo_autoencoder.pt` | Único modelo congelado (U-Net 512, semilla 42) |
| `results/resultados.csv` | Resultados de la campaña final |
