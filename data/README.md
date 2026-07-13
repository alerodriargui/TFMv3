# Dataset Chest-RSNA

No se versionan radiografías ni metadatos médicos en Git.

## Dataset seleccionado

Se usa la reorganización Chest-RSNA publicada con **BMAD: Benchmarks for
Medical Anomaly Detection**. Procede del RSNA Pneumonia Detection Challenge y
encaja con la propuesta porque es 2D, de rayos X, contiene una partición de
entrenamiento exclusivamente normal y proporciona validación y test comunes.

Fuentes oficiales:

- BMAD: <https://github.com/DorisBao/BMAD>
- Artículo BMAD: <https://openaccess.thecvf.com/content/CVPR2024W/VAND/html/Bao_BMAD_Benchmarks_for_Medical_Anomaly_Detection_CVPRW_2024_paper.html>
- Desafío RSNA: <https://www.kaggle.com/c/rsna-pneumonia-detection-challenge>

BMAD distribuye los conjuntos preparados mediante los enlaces de su README.
La descarga requiere aceptar las condiciones de la fuente. Descargar
`Chest-AD.zip`, extraerlo y proporcionar la raíz mediante:

```powershell
$env:TFM_DATA_ROOT = 'D:\datasets\Chest-RSNA'
```

No se incluye un descargador que eluda la aceptación de licencia o los permisos
de Google Drive/Kaggle.

## Copia disponible en este equipo

El dataset completo ya está en:

```text
C:\misarchivos\GitHub\TFMv2\data\raw\rsna_bmad\Chest-RSNA
```

El resolvedor del proyecto detecta esa ruta automáticamente. Esto evita una
copia redundante de aproximadamente 9 GB sin reducir la reproducibilidad.

## Auditoría realizada

| Partición | Normal | Anómala | Total |
|---|---:|---:|---:|
| Entrenamiento | 8.000 | 0 | 8.000 |
| Validación | 70 | 1.420 | 1.490 |
| Test | 781 | 16.413 | 17.194 |
| **Total** | **8.851** | **17.833** | **26.684** |

Son PNG en escala de grises de 1024x1024. La auditoría previa no encontró
imágenes ilegibles ni duplicados exactos entre particiones. El script de este
repositorio vuelve a comprobar estructura, conteos, legibilidad y una muestra
de dimensiones sin modificar los datos:

```powershell
& $python scripts\audit_dataset.py --strict
```

## Definición operacional

`good` se considera normal y `Ungood` anómala. Esta segunda clase agrega
`Lung Opacity` y `No Lung Opacity / Not Normal`. Por tanto, el TFM estudia
normalidad frente a no normalidad según RSNA/BMAD, no diagnóstico clínico ni
localización exhaustiva de patologías.
