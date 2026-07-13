# TFM: detección de anomalías en radiografías

Proyecto mínimo para comparar un autoencoder (AE), un autoencoder variacional
(VAE) y GANomaly usando Chest-RSNA. Incluye un único clasificador supervisado
pequeño como referencia práctica.

## Flujo completo

```text
radiografías → modelo → puntuación de anomalía → métricas
```

Solo hay dos comandos:

```powershell
$python = '..\TFMv2\.tools\python-3.11.9\tools\python.exe'
& $python run.py       # entrena y evalúa los cuatro modelos
& $python control.py   # evalúa el control estadístico sencillo
```

`run.py` ejecuta 20 épocas con semillas 13, 42 y 73. Los datos se encuentran
automáticamente en la copia local de Chest-RSNA. Para usar otra ubicación:

```powershell
$env:TFM_DATA_ROOT = 'D:\datasets\Chest-RSNA'
```

## Archivos

| Archivo | Función |
|---|---|
| `run.py` | Ejecuta todos los experimentos |
| `control.py` | Ejecuta el control de intensidad/textura |
| `data.py` | Carga y redimensiona radiografías |
| `models.py` | Define AE, VAE, GANomaly y el clasificador |
| `experiment.py` | Entrena, selecciona umbral y evalúa |
| `metrics.py` | Calcula AUROC, AUPRC y matriz de confusión |
| `resultados.csv` | Resultados agregados de los modelos |
| `control.csv` | Resultados del control sencillo |
| `memoria.tex` | Memoria completa |
| `presentacion.tex` | Diapositivas de defensa |
| `referencias.bib` | Bibliografía |
| `errores.png` | Ejemplos de errores de GANomaly |

## Resultado principal

| Método | AUROC |
|---|---:|
| AE | 0,4566 ± 0,0163 |
| VAE | 0,4506 ± 0,0016 |
| GANomaly | **0,5636 ± 0,0150** |
| Control de gradiente | 0,5981 |
| Clasificador supervisado | **0,8538 ± 0,0070** |

La mejora se obtiene con una sola extensión: un clasificador de 175.025
parámetros que reutiliza el codificador convolucional. Se entrena con 1.136
imágenes por clase, valida con 284 por clase y nunca usa el test durante el
desarrollo. No es una comparación directa con el escenario no supervisado:
demuestra que las etiquetas aportan la señal que AE, VAE y GANomaly no captan.
