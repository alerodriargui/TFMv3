# TFM: detección de anomalías en radiografías

Proyecto mínimo y reproducible para comparar un autoencoder (AE), un
autoencoder variacional (VAE) y GANomaly sobre Chest-RSNA. Incluye un único
clasificador supervisado como referencia práctica y una demostración de
inferencia sobre una radiografía.

## Instalación

Se recomienda Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset

Se utiliza RSNA Pneumonia Detection Challenge con la reorganización de
[BMAD](https://github.com/DorisBao/BMAD). `Chest-AD.zip` se descarga desde el
Google Drive enlazado por BMAD, aceptando previamente las condiciones de la
[fuente RSNA](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018).

Estructura esperada:

```text
Chest-RSNA/
├── train/good/
├── valid/good/
├── valid/Ungood/
├── test/good/
└── test/Ungood/
```

La copia auditada contiene 8.000 imágenes de entrenamiento, 1.490 de
validación y 17.194 de test. Para indicar su ubicación:

```powershell
$env:TFM_DATA_ROOT = 'D:\datasets\Chest-RSNA'
```

## Ejecución

Todo el estudio se reproduce con dos comandos:

```powershell
python run.py       # AE, VAE, GANomaly y clasificador; tres semillas
python control.py   # control estadístico de intensidad y textura
```

La ejecución completa usa 20 épocas y semillas 13, 42 y 73. Al ejecutar el
clasificador con la semilla 42 se genera `modelo_clasificador.pt`, el único
checkpoint conservado para la demostración.

Para clasificar una radiografía:

```powershell
python demo.py D:\radiografias\ejemplo.png
```

La salida muestra la probabilidad, el umbral fijado en validación y la clase
normal/anómala. Es una herramienta experimental, no un diagnóstico.

## Resultado principal

| Método | AUROC |
|---|---:|
| AE | 0,4566 ± 0,0163 |
| VAE | 0,4506 ± 0,0016 |
| GANomaly | 0,5636 ± 0,0150 |
| Control de gradiente | 0,5981 |
| Clasificador supervisado | **0,8538 ± 0,0070** |

El clasificador tiene 175.025 parámetros y reutiliza el codificador
convolucional. Se ajusta con 1.136 imágenes por clase, valida con 284 por clase
y mantiene intacto el test oficial. No se compara como si fuera un detector no
supervisado: cuantifica el valor aportado por las etiquetas.

## Entregables

| Archivo | Función |
|---|---|
| `memoria.pdf` | Memoria final |
| `presentacion.pdf` | Diapositivas de defensa |
| `demo.py` | Demostración con una imagen |
| `modelo_clasificador.pt` | Único modelo congelado |
| `run.py` | Entrenamiento y evaluación |
| `control.py` | Control estadístico |
| `resultados.csv` | Resultados agregados |
| `referencias.bib` | Bibliografía |

Los documentos se recompilan con:

```powershell
tectonic memoria.tex
tectonic presentacion.tex
```
