# TFM: detección no supervisada de anomalías

Sistema para detectar radiografías anómalas aprendiendo únicamente con
imágenes normales. El autoencoder procesa las imágenes originales a
1024×1024 para conservar estructuras pequeñas y textura radiográfica.

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
|   |-- scoring.py     Puntuaciones compartidas por entrenamiento y demo
|   |-- train.py       Comando de entrenamiento
|   `-- demo.py        Evaluación de una sola imagen
|-- checkpoints/       Pesos congelados del modelo
|-- results/           Métricas, experimentos y figuras generadas
|-- notebooks/         Notebook para Google Colab
|-- docs/              Documentación del experimento actual
|-- legacy/64x64/      Pesos, resultados y documentos anteriores
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
radiografía 1024x1024
      ↓
Conv 1→8→16→32→64→128→128
      ↓
representación 128×16×16
      ↓
ConvTranspose 128→128→64→32→16→8→1
      ↓
reconstrucción
```

Los seis bloques de bajada comprimen espacialmente 1024→16. La arquitectura
tiene 682.425 parámetros. Los pesos se ajustan exclusivamente con las 8.000
radiografías normales. La
puntuación final promedia dos señales tipificadas:

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
         │     Define la arquitectura del autoencoder
         │
         ├── data.py
         │     Entrega los lotes de imágenes
         │
         ├── scoring.py
         │     Calcula las puntuaciones de anomalía
         │
         ├── metrics.py
         │     Calcula umbral, AUROC y balanced accuracy
         │
         └── Guarda resultados
```

El AE usa tres épocas, batch 2 y Adam con tasa `1e-3`. El batch se reduce
porque las imágenes 1024×1024 requieren mucha más memoria que las de 64×64.
Se recomienda una GPU. Si aparece un error de memoria, puede ejecutarse con
`--batch-size 1`.

### Ejecución en Google Colab con GPU

El notebook `notebooks/TFMv3_colab.ipynb` reproduce el experimento principal en una
runtime GPU de Google Colab. Incluye clonación del repositorio, instalación de
dependencias de apoyo, comprobación de CUDA/PyTorch, montaje de Google Drive,
conteos del dataset, prueba reducida y ejecución completa del AE con semillas
13, 42 y 73.

Para usarlo:

1. Abre `notebooks/TFMv3_colab.ipynb` en Colab.
2. Selecciona `Runtime > Change runtime type > GPU`.
3. Coloca `Chest-RSNA` en Drive y ajusta `DATA_ROOT` si no está en
   `/content/drive/MyDrive/datasets/Chest-RSNA`.
4. Ejecuta todas las celdas. Los resultados se guardan también en
   `/content/drive/MyDrive/TFMv3_colab_outputs/`.

Para evaluar una radiografía con el autoencoder congelado:

```powershell
python -m tfm_ae.demo D:\radiografias\ejemplo.png
```

Después de entrenar, la demo usa por defecto
`checkpoints/modelo_autoencoder.pt` y guarda la figura
en `results/demo_resultado.png`. Ambas rutas se pueden cambiar con `--model` y
`--output`.

La consola conserva el error de reconstrucción, la puntuación, el umbral y la
clase. Además, el PNG generado contiene cuatro paneles:

1. imagen original preprocesada a 1024×1024;
2. reconstrucción producida por el autoencoder;
3. mapa de error absoluto por píxel, con escala de color y MAE;
4. puntuación de anomalía comparada con el umbral congelado de validación.

Puede añadirse `--show` para abrir la figura en un entorno gráfico. Sin esa
opción se usa un backend sin interfaz y el PNG se genera igualmente. La
reconstrucción es la aproximación aprendida por el autoencoder, no una
radiografía clínicamente normal garantizada. La demo es una herramienta
experimental y no constituye un diagnóstico.

## Resultados

Las métricas 1024×1024 están pendientes de volver a ejecutar las semillas 13,
42 y 73. `results/resultados.csv` se generará automáticamente al terminar.

El checkpoint, las métricas y los documentos del experimento anterior a
64×64 se conservan en `legacy/64x64/`; no deben confundirse con resultados
de la arquitectura actual.

## Entregables

| Ruta | Función |
|---|---|
| `docs/` | Documentación pendiente del experimento 1024×1024 |
| `notebooks/TFMv3_colab.ipynb` | Ejecución reproducible en Colab |
| `checkpoints/modelo_autoencoder.pt` | Modelo 1024×1024 generado al entrenar |
| `results/resultados.csv` | Resultados 1024×1024 generados al entrenar |
| `legacy/64x64/` | Entregables históricos del experimento anterior |
