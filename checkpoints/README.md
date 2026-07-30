# Checkpoints

`python -m tfm_ae.train` guarda aquí `modelo_autoencoder.pt` cuando completa
el entrenamiento 1024×1024 con la semilla 42.

El checkpoint contiene los pesos, el nombre de la puntuación
`reconstruction_mae` y el umbral obtenido como percentil 95 del MAE de las
radiografías normales de validación.

El modelo anterior de 64×64 está conservado en `legacy/64x64/`.
