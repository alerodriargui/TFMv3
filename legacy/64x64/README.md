# Experimento anterior a 64×64

Esta carpeta conserva los entregables del experimento original:

- `modelo_autoencoder.pt`: pesos incompatibles con la arquitectura actual;
- `resultados.csv`: métricas obtenidas reduciendo las imágenes a 64×64;
- `docs/`: memoria, presentación y diagramas que documentan ese experimento.

El pipeline actual trabaja a 1024×1024. Para generar su checkpoint hay que
volver a entrenar con:

```powershell
python -m tfm_ae.train
```
