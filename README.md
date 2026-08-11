# TFM: detección no supervisada de anomalías

Sistema no supervisado para detectar anomalías en imágenes médicas aprendiendo
únicamente con muestras normales. El modelo final es un autoencoder
convencional de 16.281 parámetros que reconstruye cortes cerebrales de
240×240 (BraTS2021) y 256×256 (Chest-RSNA). La puntuación combina el error de
reconstrucción calibrado con señales globales de la imagen (kurtosis, región
brillante más grande, gradiente medio, entropía), con un peso calibrado en
validación.

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
|   |-- models.py        Arquitecturas (ConvAutoencoder, UNetAutoencoder)
|   |-- data.py          Carga determinista de imágenes
|   |-- features.py      Señales globales de imagen
|   |-- metrics.py       AUROC, umbral y métricas
|   |-- experiment.py    Entrenamiento, calibración y evaluación
|   |-- train.py         Comando de entrenamiento
|   |-- demo.py          Demo original (U-Net 512, Chest-RSNA)
|   `-- demo_hybrid.py   Demo del score híbrido (AE, BraTS2021)
|-- checkpoints/         Pesos congelados
|-- results/             Métricas, experimentos y figuras
|   |-- brain_v2_final/  Campaña Cerebro v2 (Colab GPU, 10 épocas)
|   `-- brain_hybrid_full/ Campaña híbrida local (256×256, 3 épocas)
|-- notebooks/           Notebooks de Google Colab
|-- docs/                Memoria en LaTeX y recursos
|-- data/raw/rsna_bmad/  Datos locales (Chest-RSNA y BraTS2021)
`-- requirements*.txt    Dependencias
```

## Datos

El código soporta dos datasets organizados por BMAD:

- **Chest-RSNA**: radiografías de tórax (8.000 train, ~1.490 valid, ~17.194 test)
- **BraTS2021_slice**: cortes cerebrales FLAIR (7.500 train, 39 normales + 44 anómalos valid, 640 normales + 3.075 anómalos test)

Para indicar su ubicación:

```powershell
$env:TFM_DATA_ROOT = 'D:\datasets\BraTS2021_slice'
```

El módulo `data.py` resuelve automáticamente la ruta entre variables de entorno,
argumentos o ubicaciones conocidas.

## Modelos

| Modelo | Parámetros | Resolución | Uso |
|---|---:|---:|---|
| `ae` (ConvAutoencoder) | 16.281 | 64–256 | Modelo principal |
| `unet` (UNetAutoencoder) | 366.433 | 512 | Demo original Chest-RSNA |

El AE simple tiene tres etapas convolucionales (1→8→16→32 canales) con
decodificador simétrico. El U-Net añade conexiones de salto en cuatro etapas
(1→16→32→64→128).

## Puntuación

El experimento admite dos modos de puntuación seleccionables con `--score-mode`:

- **ae_classic**: `0,5 × MAE calibrado + 0,5 × centro-borde calibrado`
- **hybrid**: `w × señales globales calibradas + (1-w) × MAE calibrado`

Las señales globales (`features.py`) son cuatro estadísticos calculados sobre
cada imagen: kurtosis, tamaño de la región brillante más grande, gradiente medio
y entropía. Cada señal se tipifica con la media y desviación de las normales de
entrenamiento; su signo se fija por la dirección del AUROC en validación. El
peso `w` se busca en validación con una rejilla de 21 valores entre 0 y 1.

## Ejecución

```powershell
python -m tfm_ae.train --model ae --image-size 240 --epochs 10 --score-mode hybrid
```

Argumentos principales:

```text
--data-root         Ruta al dataset
--model             ae | unet
--image-size        Resolución (64, 240, 256, 512...)
--epochs            Número de épocas
--batch-size        Tamaño del lote (def: 32)
--score-mode        ae_classic | hybrid
--seeds             Semillas para la campaña (def: 13 42 73)
--bottleneck        Canales del cuello de botella (def: 32)
--noise-std         Ruido gaussiano para denoising AE
--max-train-images  Limita imágenes de entrenamiento (debug)
--max-eval-images-per-class  Limita imágenes de evaluación (debug)
```

El entrenamiento guarda en `results/experiments/ae_seed{N}/`:
- `model.pt` — pesos del modelo
- `metrics.json` — configuración, historial, calibración y métricas
- `validation_scores.csv` / `test_scores.csv` — puntuaciones por imagen
- `reconstructions.png` — reconstrucciones de ejemplo

### Demo híbrida (cerebro)

```powershell
python -m tfm_ae.demo_hybrid D:\cortes\paciente_001.png
```

Usa por defecto `results/brain_hybrid_full/ae_seed42/metrics.json` y guarda la
figura en `results/demo_hybrid_resultado.png`. La figura contiene cuatro paneles:
original, reconstrucción, mapa de error absoluto y puntuación híbrida frente al
umbral.

### Demo original (tórax)

```powershell
python -m tfm_ae.demo D:\radiografias\ejemplo.png
```

Usa `checkpoints/modelo_autoencoder.pt` (U-Net 512, Chest-RSNA).

## Resultados

### Cerebro (BraTS2021, score híbrido)

| Ejecución | Resolución | Épocas | AUROC | Bal. acc. |
|---|---:|---:|---:|---:|
| Local (semilla 42) | 256×256 | 3 | 0,9085 | 0,8364 |
| Colab GPU (semilla 42) | 240×240 | 10 | 0,9030 | 0,8327 |

El peso híbrido seleccionado en validación es `w = 0,95`: las señales globales
dominan la separación y el MAE actúa como señal complementaria. La specificity
en test supera el 94% mientras la sensitivity se mantiene por encima del 72%.

### Tórax (Chest-RSNA, score clásico)

| Modelo | Parámetros | AUROC | Bal. acc. |
|---|---:|---:|---:|
| Control de gradiente | 0 | 0,5981 | 0,5894 |
| AE mínimo 64×64 | 16.281 | 0,7608 | 0,6790 |
| U-Net final 512×512 | 366.433 | 0,6574 | 0,6224 |

## Google Colab

El notebook `notebooks/TFMv3_colab_brain.ipynb` reproduce el experimento de
cerebro en Colab con GPU. Descarga BraTS2021, ejecuta una prueba reducida y
lanza la campaña completa con respaldo incremental en Drive.

1. Abre el notebook en Colab.
2. Selecciona `Runtime > Change runtime type > GPU`.
3. Ejecuta todas las celdas.

## Entregables

| Ruta | Función |
|---|---|
| `docs/memoria.pdf` | Memoria final (LaTeX) |
| `docs/assets/` | Diagramas y recursos |
| `notebooks/TFMv3_colab_brain.ipynb` | Reproducible en Colab |
| `checkpoints/modelo_autoencoder.pt` | Modelo congelado U-Net 512 (Chest-RSNA) |
| `results/brain_hybrid_full/ae_seed42/` | Resultados cerebrales híbridos |
| `results/brain_v2_final/` | Resultados campaña Colab GPU |
