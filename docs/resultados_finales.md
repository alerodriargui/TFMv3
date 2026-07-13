# Resultados finales

Se ejecutaron AE, VAE y GANomaly durante 20 épocas con semillas 13, 42 y 73
sobre las 8.000 radiografías normales de entrenamiento. El checkpoint se eligió
con validación normal y el umbral con validación etiquetada; test permaneció
congelado hasta el final.

| Modelo | AUROC | AUPRC | Exactitud balanceada | Tiempo |
|---|---:|---:|---:|---:|
| AE | 0,4566 ± 0,0163 | 0,9533 ± 0,0021 | 0,5105 ± 0,0086 | 205,6 ± 27,9 s |
| VAE | 0,4506 ± 0,0016 | 0,9526 ± 0,0002 | 0,5064 ± 0,0018 | 221,4 ± 17,4 s |
| GANomaly | **0,5636 ± 0,0150** | **0,9655 ± 0,0013** | **0,5484 ± 0,0126** | 572,3 ± 107,1 s |

GANomaly supera AE y VAE en todas las semillas, pero el control simple de
gradiente medio logra AUROC 0,5981 y exactitud balanceada 0,5894. La evidencia
no permite afirmar que GANomaly aprenda una señal clínica superior a factores
de textura o adquisición.

El análisis visual apoya esta cautela: los falsos positivos extremos incluyen
rotación, encuadre pediátrico, marcadores y proyecciones portátiles. La figura
`reports/error_analysis_ganomaly.png` reúne los casos representativos.

La prevalencia anómala de test es 95,46 %, por lo que AUPRC alrededor de 0,95 no
implica por sí sola un detector útil. AUROC, exactitud balanceada, sensibilidad
y especificidad sostienen la interpretación principal.
