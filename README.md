# TFM: detección no supervisada de anomalías

Sistema sencillo para detectar radiografías anómalas aprendiendo únicamente
con imágenes normales. El modelo final es un autoencoder convolucional de
16.281 parámetros y tres bloques de bajada/subida.

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

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
radiografía 64x64
      ↓
Conv 1→8 → Conv 8→16 → Conv 16→32
      ↓
ConvTranspose 32→16 → 16→8 → 8→1
      ↓
reconstrucción
```

Los pesos se ajustan exclusivamente con las 8.000 radiografías normales. La
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
python run.py       # AE, VAE y GANomaly; semillas 13, 42 y 73
python control.py   # controles estadísticos
```

El AE usa tres épocas, batch 32 y Adam con tasa `1e-3`. VAE y GANomaly se
conservan como comparaciones de la propuesta original y usan 20 épocas.

Para evaluar una radiografía con el autoencoder congelado:

```powershell
python demo.py D:\radiografias\ejemplo.png --output demo_resultado.png
```

La consola conserva el error de reconstrucción, la puntuación, el umbral y la
clase. Además, el PNG generado contiene cuatro paneles:

1. imagen original preprocesada a 64×64;
2. reconstrucción producida por el autoencoder;
3. mapa de error absoluto por píxel, con escala de color y MAE;
4. puntuación de anomalía comparada con el umbral congelado de validación.

Puede añadirse `--show` para abrir la figura en un entorno gráfico. Sin esa
opción se usa un backend sin interfaz y el PNG se genera igualmente. La
reconstrucción es la aproximación aprendida por el autoencoder, no una
radiografía clínicamente normal garantizada. La demo es una herramienta
experimental y no constituye un diagnóstico.

## Resultados

| Método no supervisado | AUROC | Balanced accuracy |
|---|---:|---:|
| VAE | 0,4506 | 0,5064 |
| GANomaly | 0,5636 | 0,5484 |
| Control de gradiente | 0,5981 | 0,5894 |
| **AE final calibrado** | **0,7608** | **0,6790** |

El resultado se repite en las tres semillas: 0,7608, 0,7603 y 0,7614. No se
utiliza ningún clasificador supervisado, ensamble ni red preentrenada.

## Entregables

| Archivo | Función |
|---|---|
| `memoria.pdf` | Memoria final |
| `presentacion.pdf` | Diapositivas |
| `demo.py` | Demostración con una imagen |
| `modelo_autoencoder.pt` | Único modelo congelado |
| `generate_architecture_diagram.py` | Regenera el diagrama del AE desde `models.py` |
| `artifacts/autoencoder_architecture.{svg,pdf,png}` | Diagrama vectorial/editable y versión de uso directo |
| `run.py` | Entrenamiento y evaluación |
| `resultados.csv` | Resultados de tres semillas |
| `referencias.bib` | Bibliografía |

El diagrama se regenera con:

```bash
python generate_architecture_diagram.py
```
