# Guion de defensa — 10 minutos

## 1. Apertura — 40 segundos

El trabajo estudia detección de anomalías en radiografías cuando solo se dispone
de imágenes normales para entrenar. La motivación es que recopilar y etiquetar
todas las patologías posibles resulta costoso. La salida del sistema es una
puntuación de rareza; no es un diagnóstico clínico.

## 2. Objetivo — 45 segundos

La pregunta es sencilla: bajo exactamente el mismo dataset y protocolo, ¿qué
funciona mejor, un AE, un VAE o una alternativa adversarial GANomaly? La
aportación no es una nueva arquitectura, sino una comparación controlada,
reproducible y acompañada de controles que evitan conclusiones engañosas.

## 3. Dataset — 55 segundos

Chest-RSNA/BMAD contiene 26.684 radiografías. Entrenamiento tiene 8.000 normales;
validación, 70 normales y 1.420 anómalas; test, 781 y 16.413. La etiqueta
anómala significa no normal en el benchmark, no necesariamente neumonía ni
cualquier patología. Se auditó estructura, legibilidad y conteos.

## 4. Modelos — 70 segundos

- AE usa error absoluto de reconstrucción.
- VAE añade una distribución latente regularizada mediante KL.
- GANomaly reconstruye y vuelve a codificar; la distancia entre ambos latentes
  constituye la puntuación, con aprendizaje adversarial.

Los tres usan entrada 64×64, latente 64, Adam, 20 épocas y las mismas tres
semillas.

## 5. Protocolo — 60 segundos

Los pesos ven exclusivamente entrenamiento normal. El checkpoint se elige con
validación normal. Solo después se usan las etiquetas completas de validación
para fijar el umbral. Test se evalúa con todo congelado. La métrica principal es
AUROC; AUPRC se interpreta con cuidado porque test tiene 95,46 % de anomalías.

## 6. Resultados — 90 segundos

AE logra AUROC medio 0,4566; VAE, 0,4506; GANomaly, 0,5636. GANomaly supera a los
otros dos en las tres semillas, pero cuesta aproximadamente 2,7 veces más.
AE/VAE reconstruyen las normales, aunque asignan una mediana de error ligeramente
menor a las anómalas: la hipótesis de que anomalía equivale automáticamente a
peor reconstrucción no se cumple.

## 7. Control simple — 75 segundos

El resultado más importante es el control: una distancia basada en gradiente
medio alcanza AUROC 0,5981, por encima de GANomaly. Esto demuestra que parte de
la señal puede proceder de textura, contraste o proceso de adquisición. Sin el
control se habría presentado GANomaly como un éxito excesivo.

## 8. Errores — 55 segundos

Los falsos positivos extremos incluyen encuadres pediátricos, rotación,
marcadores y proyecciones portátiles. Los falsos negativos suelen mantener una
apariencia de adquisición más estandarizada. Es un análisis visual metodológico,
no una lectura radiológica.

## 9. Cierre — 50 segundos

GANomaly es el mejor modelo neuronal estudiado, pero el rendimiento absoluto es
modesto y no supera un control simple. Por tanto, no existe evidencia para uso
clínico. El resultado es útil porque cuantifica el límite de AE/VAE, mide el
coste adversarial y descubre un atajo. Los siguientes pasos serían mayor
resolución, particiones verificadas por paciente y validación externa.

## Preguntas previsibles

### ¿Por qué 64×64?

Porque permite completar nueve ejecuciones en CPU bajo un presupuesto de TFM.
Se reconoce la pérdida de detalle y 128×128 queda como ablación prioritaria; no
se afirma que 64×64 sea óptimo clínicamente.

### ¿Por qué AUPRC es tan alto si el modelo es malo?

Porque el 95,46 % de test pertenece a la clase positiva. Un ranking casi
aleatorio puede obtener AUPRC cercana a esa prevalencia. Por eso AUROC y
exactitud balanceada sostienen la conclusión.

### ¿Por qué el AE obtiene AUROC menor de 0,5?

Las anomalías tienen, en mediana, un error de reconstrucción ligeramente menor.
El modelo puede reconstruir estructuras suaves o responder a contraste y
adquisición; no existe garantía de que toda anomalía sea más difícil de copiar.

### ¿GANomaly funciona?

Funciona mejor que AE/VAE de forma consistente, pero solo alcanza AUROC medio
0,5636 y pierde frente al control de gradiente. Es una mejora metodológica, no
un detector clínicamente útil.

### ¿Cuál es la principal limitación?

La posible dependencia de atajos de adquisición, seguida de la baja resolución
y la ausencia de validación externa por paciente o centro.
