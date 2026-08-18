"""Estadísticas globales de imagen usadas como señales de detección de anomalías."""

from __future__ import annotations

import numpy as np


# Las señales globales que el modelo híbrido usa como características
GLOBAL_SIGNALS = ("kurt", "cc_largest", "grad_mean", "entropy")


def connected_bright(img: np.ndarray, level: float) -> tuple[float, float]:
    """Devuelve (mayor, segunda mayor) región conectada brillante, en fracción del área total."""
    mask = img > level  # Mapa booleano: 1 = píxel brillante (por encima del umbral)
    h, w = mask.shape
    if not mask.any():  # Ningún píxel brillante: no hay regiones
        return 0.0, 0.0
    visited = np.zeros_like(mask)  # Marca qué píxeles ya se contaron (evita repetir)
    sizes = []  # Tamaño (en píxeles) de cada región conectada que encontremos
    for i in range(h):
        for j in range(w):
            # Si es un píxel brillante aún sin visitar -> nueva región → recorrido BFS/DFS
            if mask[i, j] and not visited[i, j]:
                stack = [(i, j)]  # Pila de píxeles pendientes de explorar
                visited[i, j] = True
                size = 0
                while stack:
                    ci, cj = stack.pop()  # Saca un píxel de la región
                    size += 1
                    # Mira sus 4 vecinos (arriba, abajo, izquierda, derecha)
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ni, nj = ci + di, cj + dj
                        # Si el vecino es brillante, no visitado y está dentro de la imagen...
                        if (
                            0 <= ni < h
                            and 0 <= nj < w
                            and mask[ni, nj]
                            and not visited[ni, nj]
                        ):
                            visited[ni, nj] = True  # ... lo marca y lo encola
                            stack.append((ni, nj))
                sizes.append(size)  # Región completa explorada: guarda su tamaño
    sizes.sort(reverse=True)  # De mayor a menor
    largest = sizes[0]  # Región conectada más grande
    second = sizes[1] if len(sizes) > 1 else 0.0  # Segunda más grande (0 si solo hay una)
    return largest / (h * w), second / (h * w)  # Como fracción del total de píxeles


def image_features(img: np.ndarray) -> dict[str, float]:
    """Estadísticas de una imagen en escala de grises (valores en [0, 1])."""
    # Magnitud del gradiente: mide los "bordes" / cambios bruscos de intensidad por píxel
    gy, gx = np.gradient(img)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    # Histograma de intensidades (32 bines, normalizado a densidad de probabilidad)
    bins = np.histogram(img, bins=32, range=(0, 1), density=True)[0]
    bins = bins[bins > 0]  # Descarta bines con frecuencia 0 (log(0) no existe)
    mean = float(img.mean())  # Intensidad media
    std = float(img.std())  # Desviación típica: cuánto varía la imagen
    # Umbral para "brillo": el percentil 90 (o 0.9 si la imagen es muy oscura/plana)
    level = max(float(np.percentile(img, 90)), 0.9)
    largest, second = connected_bright(img, level)  # Tamaño de las regiones brillantes
    return {
        # Kurtosis: la imagen concentra la energía en pocos valores (picos) o está plana
        "kurt": float(((img - mean) ** 4).mean() / (std + 1e-8) ** 4),
        # Energía media de los bordes (gradiente medio)
        "grad_mean": float(grad.mean()),
        # Entropía (Shannon) del histograma: qué "aleatoria"/rica en información es la imagen
        "entropy": float(-(bins * np.log(bins)).sum()),
        "cc_largest": largest,  # Región brillante conectada más grande
        "cc_second": second,  # Segunda región brillante conectada
    }


def feature_matrix(images: np.ndarray) -> dict[str, np.ndarray]:
    """Calcula el banco de características para un lote de imágenes (N, H, W)."""
    # Toma las claves del dict de la primera imagen como nombre de columnas
    keys = list(image_features(images[0]).keys())
    # Pre-reserva un array por característica, con una entrada por imagen
    out = {k: np.empty(len(images), dtype=float) for k in keys}
    for index, img in enumerate(images):  # Por cada imagen del lote
        for k, v in image_features(img).items():  # Calcula sus características
            out[k][index] = v  # Y las coloca en la fila correspondiente
    return out  # dict: característica -> vector con un valor por imagen