# Protocolo experimental cerrado

## Pregunta

¿Cuál de AE, VAE y GANomaly ordena mejor radiografías Chest-RSNA normales y no
normales cuando todos se entrenan exclusivamente con imágenes normales?

## Reglas antes de mirar test

1. No modificar las particiones BMAD.
2. No entrenar pesos con etiquetas anómalas.
3. Seleccionar el checkpoint por puntuación media en validación normal.
4. Seleccionar un umbral por exactitud balanceada en validación completa.
5. Aplicar una sola vez el modelo y umbral congelados a test.
6. Ejecutar las semillas 13, 42 y 73.
7. Informar todas las ejecuciones, incluidas las fallidas, y no escoger semillas.

## Comandos finales

```powershell
$python = '..\TFMv2\.tools\python-3.11.9\tools\python.exe'
$env:PYTHONPATH = 'src'
.\scripts\run_final.ps1 -Python $python
```

## Criterio de interpretación

- Principal: AUROC de test, media y desviación entre semillas.
- Secundarias: AUPRC, exactitud balanceada, sensibilidad, especificidad y F1.
- Recursos: parámetros, tiempo y dispositivo.
- No declarar superioridad por una sola semilla o por F1 aislado.
- Comparar con controles simples antes de atribuir la señal a anatomía.

## Amenazas a la validez

- Redimensionado 1024 a 64 puede eliminar hallazgos pequeños.
- Las etiquetas RSNA no representan toda la patología torácica.
- El test está fuertemente desequilibrado hacia la clase anómala.
- El modelo puede explotar texto, dispositivos, contraste o protocolo.
- Un solo centro/benchmark no demuestra generalización clínica.
