# Guía técnica para defender el proyecto

Esta guía explica **lo que hace realmente el código actual**. No está pensada
para memorizar frases, sino para poder reconstruir el razonamiento durante una
pregunta.

> Regla principal: distingue siempre el experimento actual a 1024×1024 del
> histórico a 64×64. Las cifras 0,7608 de AUROC y 0,6790 de balanced accuracy
> pertenecen al segundo. Los resultados del modelo actual están pendientes.

## 1. El proyecto en un minuto

El objetivo es detectar radiografías que se apartan de la distribución normal.
Se entrena un autoencoder convolucional exclusivamente con 8.000 radiografías
normales. Aprende a comprimirlas y reconstruirlas. En inferencia se usa como
puntuación el error absoluto medio entre imagen y reconstrucción.

El umbral se fija como el percentil 95 del error sobre las radiografías normales
de validación. Después, modelo y umbral se congelan para test.

La descripción metodológica exacta es:

> **Detección no supervisada de anomalías mediante error de reconstrucción.**

Las etiquetas de validación y test solo se usan para informar AUROC, sensibilidad,
especificidad y balanced accuracy. No intervienen en los pesos, la puntuación ni
el umbral.

## 2. Recorrido completo de una imagen

1. `data.py` abre la imagen y la convierte a escala de grises.
2. Si hace falta, la redimensiona a 1024×1024 con Lanczos.
3. Convierte los píxeles 0–255 a números reales en `[0,1]`.
4. Añade el canal: forma `(1, 1024, 1024)`.
5. El `DataLoader` agrupa dos imágenes: `(2, 1, 1024, 1024)`.
6. El encoder reduce la resolución seis veces.
7. El decoder recupera la resolución original.
8. En entrenamiento, se calcula L1 y se propaga el gradiente.
9. En evaluación, el MAE de reconstrucción es la puntuación.
10. Se declara anómala si `puntuación >= umbral`.

## 3. Arquitectura capa por capa

El tamaño de salida de una convolución es:

```text
salida = floor((entrada + 2·padding - kernel) / stride) + 1
```

Con kernel 3, stride 2 y padding 1, cada convolución reduce a la mitad. En el
decoder, kernel 4, stride 2 y padding 1 duplica exactamente el tamaño.

| Etapa | Operación | Salida | Parámetros |
|---|---|---:|---:|
| Entrada | — | 1×1024×1024 | 0 |
| Encoder 1 | Conv 1→8, k3, s2, p1 | 8×512×512 | 80 |
| Encoder 2 | Conv 8→16 | 16×256×256 | 1.168 |
| Encoder 3 | Conv 16→32 | 32×128×128 | 4.640 |
| Encoder 4 | Conv 32→64 | 64×64×64 | 18.496 |
| Encoder 5 | Conv 64→128 | 128×32×32 | 73.856 |
| Encoder 6 | Conv 128→128 | 128×16×16 | 147.584 |
| Decoder 1 | ConvT 128→128, k4, s2, p1 | 128×32×32 | 262.272 |
| Decoder 2 | ConvT 128→64 | 64×64×64 | 131.136 |
| Decoder 3 | ConvT 64→32 | 32×128×128 | 32.800 |
| Decoder 4 | ConvT 32→16 | 16×256×256 | 8.208 |
| Decoder 5 | ConvT 16→8 | 8×512×512 | 2.056 |
| Salida | ConvT 8→1 + sigmoid | 1×1024×1024 | 129 |
| **Total** | | | **682.425** |

Parámetros de una convolución:

```text
canales_salida × (canales_entrada × kernel² + 1 sesgo)
```

El cuello de botella tiene `128×16×16 = 32.768` valores, frente a
`1×1024×1024 = 1.048.576`: 32 veces menos elementos. La resolución espacial
se reduce 64 veces por eje, es decir, 4.096 veces en número de posiciones. No
digas que el latente tiene 128 valores: tiene 128 **mapas** de 16×16.

Una celda del latente posee un campo receptivo teórico de 127×127 píxeles. Eso
recoge estructura regional, pero una sola celda no resume toda la radiografía.

### ¿Por qué un autoencoder convolucional?

La imagen tiene estructura espacial. La convolución comparte filtros por toda
la imagen, conserva relaciones locales y necesita muchos menos parámetros que
capas densas sobre más de un millón de píxeles. El cuello de botella obliga a
retener información útil de las normales para reconstruirlas.

La hipótesis es que una imagen fuera de esa distribución se reconstruirá de
forma diferente. No es una ley: un autoencoder también puede reconstruir bien
anomalías. En este dataset el sentido del MAE llegó a ser contraintuitivo.

### ¿Por qué seis bajadas y esos canales?

Es un compromiso de diseño:

- seis factores 2 convierten exactamente 1024 en 16;
- los canales aumentan al bajar resolución para conservar tipos de rasgos;
- limitar el máximo a 128 contiene el coste;
- la simetría facilita recuperar el tamaño original.

No existe una ablación que demuestre que 8–16–32–64–128–128 sea óptimo. Di
«compromiso razonable y reproducible», no «la mejor arquitectura».

### Activaciones y normalización

ReLU aporta no linealidad, es barata y facilita la optimización; no se comparó
con otras activaciones. La sigmoid final restringe la reconstrucción a `[0,1]`,
el mismo intervalo de entrada, aunque puede saturarse cerca de los extremos.

No hay BatchNorm. Con batch 2 sus estadísticas serían muy ruidosas. GroupNorm
o InstanceNorm serían alternativas que habría que probar. ConvTranspose puede
crear patrones de tablero; kernel 4 con stride 2 da solapamiento regular, pero
no elimina totalmente el riesgo.

## 4. Entrenamiento y optimizador

Para una normal `x`, el encoder produce `z=f(x)` y el decoder `x̂=g(z)`. Se
minimiza:

```text
L1(x, x̂) = (1/N) · Σ |xᵢ - x̂ᵢ|
```

Cada lote sigue:

```text
poner gradientes a cero → forward → L1 → backward → Adam.step()
```

PyTorch acumula gradientes por defecto; por eso se limpian. Usar
`set_to_none=True` puede ahorrar memoria y operaciones.

### ¿Qué optimizador se usa y por qué?

Se usa **Adam**, tasa `10⁻³`. El resto son valores por defecto de PyTorch:
β₁=0,9, β₂=0,999, ε=10⁻⁸, sin weight decay ni scheduler.

Adam mantiene una media móvil del gradiente y otra de su cuadrado:

```text
mₜ = β₁mₜ₋₁ + (1-β₁)gₜ
vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²
θₜ = θₜ₋₁ - α · m̂ₜ / (sqrt(v̂ₜ) + ε)
```

Los sombreros son la corrección del sesgo inicial.

Respuesta oral:

> Elegí Adam porque adapta el paso por parámetro y suele converger con poca
> sintonización en redes convolucionales. Era práctico con tres épocas y lotes
> pequeños. `10⁻³` es su tasa inicial estándar. No afirmo que sea óptimo: eso
> exigiría comparar Adam, AdamW o SGD y varias tasas solo con validación.

### ¿Por qué MAE y no MSE?

MAE penaliza cada error linealmente. MSE los eleva al cuadrado y permite que
unos pocos errores grandes dominen. L1 es menos sensible a píxeles atípicos y
coincide con el MAE usado como señal de anomalía. Es razonable, pero el modelo
actual no contiene una comparación L1–L2.

### ¿Por qué tres épocas?

Es un presupuesto computacional reducido. En cada época se mide MAE sobre
normales de validación y se guarda la mejor. Es selección de checkpoint entre
tres candidatos.

No afirmes que garantiza convergencia. Si las curvas siguen bajando en la
tercera época, el presupuesto es insuficiente. La justificación correcta es:
«fue un compromiso de coste y lo valido observando las curvas».

### ¿Por qué batch 2?

Una imagen float32 de 1024×1024 ocupa unos 4 MiB, pero hay que conservar mapas
intermedios y gradientes. Solo la activación 8×512×512 ocupa unos 8 MiB por
imagen. Batch 2 limita el pico de memoria. Su gradiente es más ruidoso y usa
peor la GPU; la razón principal es memoria, no superioridad estadística.

### Semillas y reproducibilidad

Se usan 13, 42 y 73. Controlan inicialización, orden de lotes y subconjuntos de
prueba. cuDNN se configura determinista. Repetir permite reportar media y
dispersión, en vez de escoger una ejecución afortunada. GPU, CUDA o versiones
distintas aún pueden causar pequeñas diferencias.

## 5. Datos y protocolo

| Split | Normales | Anómalas | Uso |
|---|---:|---:|---|
| Train | 8.000 | 0 | Ajuste de los pesos |
| Validación normal | 70 | 0 | Época y umbral no supervisado |
| Validación etiquetada | 70 | 1.420 | Métricas |
| Test | 781 | 16.413 | Evaluación final congelada |

Se entrena solo con normales porque es detección one-class: no requiere conocer
toda posible patología. La desventaja es que una anomalía similar a lo
aprendido puede reconstruirse bien.

Preprocesado real:

- un canal en escala de grises;
- Lanczos a 1024×1024 solo si el tamaño difiere;
- división por 255 para obtener `[0,1]`;
- sin estandarización por media/desviación;
- sin data augmentation.

1024×1024 conserva más detalle explícito que 64×64, pero cuesta mucho más. No
se puede afirmar que mejore la detección hasta completar la comparación.

### ¿Hay fuga de datos?

El pipeline final no modifica nada después de entrar en test:

- las normales de validación sirven para escoger época y fijar el umbral;
- las anomalías de validación solo sirven para informar métricas;
- ninguna etiqueta ajusta los pesos, la puntuación o el umbral;
- según la memoria histórica, durante el desarrollo se consultó test.

Por ello no es validación clínica independiente. La prueba rigurosa sería
congelar todo y evaluar una sola vez en un dataset externo, preferiblemente de
otro centro.

## 6. Puntuación y umbral

### MAE de reconstrucción

```text
r(x) = media(|x̂ - x|)
```

Condensa el mapa a un escalar. Una lesión pequeña puede diluirse entre más de
un millón de píxeles. El mapa visual marca discrepancia, no una lesión.

No se orienta, tipifica ni combina con ninguna señal auxiliar: cuanto mayor es
el MAE, mayor es la puntuación de anomalía. Esta regla queda fijada antes de
consultar las etiquetas.

### Umbral

Se calcula exclusivamente con errores de radiografías normales de validación:

```text
umbral = percentil 95 { r(x) : x es normal de validación }
```

Esto fija aproximadamente un 5 % de falsos positivos sobre esas normales, sin
necesitar ejemplos anómalos. Ese umbral se congela para la validación etiquetada,
el test y la demo.

## 7. Métricas imprescindibles

| Término | Significado |
|---|---|
| TP | anómala clasificada anómala |
| FN | anómala clasificada normal |
| TN | normal clasificada normal |
| FP | normal clasificada anómala |

```text
sensibilidad = TP / (TP + FN)
especificidad = TN / (TN + FP)
balanced accuracy = (sensibilidad + especificidad) / 2
```

La accuracy convencional engañaría: el test tiene aproximadamente 95,5 % de
anómalas. Predecir siempre «anómala» daría cerca de 95,5 % de accuracy, pero
50 % de balanced accuracy.

AUROC es la probabilidad de que una anómala elegida al azar reciba mayor
puntuación que una normal. Evalúa ranking sin umbral: 0,5 es azar, 1 separación
perfecta y menos de 0,5 indica orden principalmente inverso. Balanced accuracy
evalúa una decisión en un umbral. Una buena AUROC no garantiza un buen punto
operativo.

## 8. No confundas las dos versiones

| Aspecto | Legado 64×64 | Actual 1024×1024 |
|---|---:|---:|
| Bajadas/subidas | 3 + 3 | 6 + 6 |
| Latente | 32×8×8 | 128×16×16 |
| Parámetros | 16.281 | 682.425 |
| Batch | 32 | 2 |
| Épocas | 3 | 3 |
| Optimizador | Adam, 10⁻³ | Adam, 10⁻³ |
| Resultados | Sí | Pendientes |
| AUROC medio | 0,7608 | Pendiente |
| Balanced accuracy media | 0,6790 | Pendiente |

Frase segura:

> El experimento histórico a 64×64 obtuvo 0,7608. La versión 1024×1024 está
> implementada para estudiar si conservar detalle mejora el resultado, pero su
> campaña completa todavía debe ejecutarse.

## 9. Preguntas probables

### «¿Por qué no un clasificador supervisado?»

La pregunta es detectar desviaciones aprendiendo solo normalidad, sin depender
de ejemplos de todas las anomalías. Un clasificador sería una comparación útil,
pero resolvería otro problema.

### «¿Por qué no VAE o GAN?»

Añaden objetivos y dificultad. En el legado, VAE y GANomaly no superaron al AE
simple. Eso apoya la simplicidad allí, pero no demuestra superioridad universal
ni es todavía una comparación a 1024.

### «¿El cuello de botella garantiza aprender normalidad?»

No. Limita capacidad, pero la red puede generalizar y reconstruir anomalías.
Solo las métricas pueden validar la hipótesis.

### «¿Por qué una anómala puede tener menos MAE?»

Puede ser más homogénea, tener menos textura o parecerse a una salida suavizada
del decoder. El error no mide patología directamente.

### «¿Por qué usas anomalías en validación si dices no supervisado?»

No se usan para aprender ni decidir. Solo permiten medir a posteriori AUROC,
sensibilidad y balanced accuracy. La puntuación y el umbral se obtienen
únicamente a partir de normales.

### «¿Qué pasa en otro hospital?»

Puede haber domain shift en contraste, equipo o protocolo. Hay que validar sin
reajuste en datos externos y recalibrar solo si se declara explícitamente.

### «¿Es diagnóstico?»

No. Es detección experimental de desviación en un benchmark. No localiza ni
clasifica patologías y carece de validación clínica.

### «¿Qué mejorarías primero?»

Ejecutar y auditar la campaña 1024. Después, ablaciones de resolución, L1/L2,
percentil del umbral y capacidad. Finalmente, validación externa,
intervalos de confianza y análisis por subgrupos.

## 10. Frases peligrosas

| Evita | Sustituye por |
|---|---|
| «Adam es el mejor» | «Es práctico; no se ha demostrado óptimo aquí» |
| «1024 mejora» | «Conserva detalle; falta medir la mejora» |
| «Las etiquetas optimizan el modelo» | «Solo se usan para calcular métricas» |
| «El mapa localiza la lesión» | «Localiza discrepancia de reconstrucción» |
| «0,7608 es del modelo actual» | «Es del legado 64×64» |
| «Tres épocas bastan» | «Es el presupuesto; hay que mirar las curvas» |
| «Detecta neumonía» | «Detecta desviación en este benchmark» |
| «La arquitectura fue optimizada» | «Es un compromiso de capacidad y coste» |

## 11. Mapa del código y estudio

| Archivo | Pregunta |
|---|---|
| `tfm_ae/data.py` | ¿Cómo entran y se transforman los datos? |
| `tfm_ae/models.py` | ¿Qué capas hay? |
| `tfm_ae/experiment.py` | ¿Cómo se entrena y evalúa? |
| `tfm_ae/scoring.py` | ¿Cómo se obtiene la puntuación? |
| `tfm_ae/metrics.py` | ¿Cómo se calculan las métricas? |
| `tfm_ae/train.py` | ¿Qué configuración usa la campaña? |
| `tfm_ae/demo.py` | ¿Cómo se aplica el checkpoint? |

Orden recomendado:

```text
data → models → train en experiment → scoring → metrics
→ evaluación en experiment → demo
```

Debes poder dibujar sin mirar:

```text
x: 1×1024×1024
↓ encoder, seis reducciones
z: 128×16×16
↓ decoder
x̂: 1×1024×1024

L = media |x - x̂|
s = media |x - x̂|
umbral = percentil 95 del MAE normal de validación
anómala ⇔ s ≥ umbral
```

Plan de seis sesiones:

1. Sigue un tensor y calcula formas y parámetros.
2. Explica forward, loss, backward, Adam, L1 y L2.
3. Dibuja train, validación y test con la función exacta de cada uno.
4. Calcula a mano una matriz de confusión y explica AUROC.
5. Practica primero las limitaciones y los experimentos pendientes.
6. Haz un simulacro respondiendo cada pregunta en 15 segundos y en un minuto.

Entender el proyecto no es justificar cada decisión como perfecta. Es saber qué
se hizo, qué evidencia lo respalda, qué fue un compromiso y qué experimento
resolvería la incertidumbre.
