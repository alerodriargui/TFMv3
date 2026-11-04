"""Evaluacion espacial con mascaras de referencia; no modifica los pesos."""
from __future__ import annotations
# Utilidades para argumentos de consola, huellas digitales y archivos de resultados.
import argparse
import hashlib
import csv
import json
from pathlib import Path
# NumPy procesa matrices y PyTorch ejecuta el modelo y las operaciones con tensores.
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
# Carga de imágenes y resolución de la carpeta de datos del proyecto.
from .data import RadiographDataset, resolve_data_root


def model_fingerprint(model) -> str:
    # Identifica el estado del modelo mediante una huella SHA-256.
    digest = hashlib.sha256()
    # Ordena parámetros y buffers por nombre para obtener un resultado reproducible.
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode('utf-8'))
        # Desvincula gradientes, mueve a CPU y convierte los datos contiguos a bytes.
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_mask(image_path: Path, size: int) -> np.ndarray:
    # Exige que la imagen esté dentro de img y busca su máscara en label.
    if image_path.parent.name != 'img':
        raise ValueError(f'Imagen fuera de img/: {image_path}')
    path = image_path.parent.parent / 'label' / image_path.name
    with Image.open(path) as source:
        # Ignorar el canal alfa: es opaco tambien en el fondo.
        # Vecino más próximo redimensiona sin interpolar los valores de la máscara.
        mask = source.convert('RGB').resize((size, size), Image.Resampling.NEAREST)
        # Un píxel es lesión (True) si alguno de sus canales RGB supera cero.
        return np.asarray(mask).max(axis=2) > 0


def anomaly_maps(images: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    # Conserva zonas donde más del 95 % de una ventana 5x5 supera intensidad 0.01.
    foreground = F.avg_pool2d((images > .01).float(), 5, stride=1, padding=2) > .95
    # Diferencia absoluta entre reconstrucción y original; fuera de esas zonas vale cero.
    error = (reconstruction - images).abs() * foreground
    # Refleja los bordes y extrae ventanas 5x5 alrededor de cada píxel.
    patches = F.pad(error, (2, 2, 2, 2), mode='reflect').unfold(2, 5, 1).unfold(3, 5, 1)
    # Mediana de los 25 valores para reducir ruido y máximo entre canales.
    # Devuelve un mapa por imagen, con dimensiones [lote, alto, ancho].
    return patches.contiguous().view(*error.shape, 25).median(-1).values.amax(dim=1)


def overlap(prediction: np.ndarray, reference: np.ndarray) -> dict:
    # Verdaderos positivos: lesión detectada correctamente.
    tp = int((prediction & reference).sum())
    # Falsos positivos: fondo marcado como lesión.
    fp = int((prediction & ~reference).sum())
    # Falsos negativos: lesión que no se detectó.
    fn = int((~prediction & reference).sum())
    # Dice mide coincidencia; IoU, intersección/unión; sensibilidad, lesión detectada.
    # Con ambas máscaras vacías, Dice e IoU son 1; sin lesión, sensibilidad es None.
    return dict(tp=tp, fp=fp, fn=fn,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                sensitivity=tp/(tp+fn) if tp+fn else None)


# Desactiva los gradientes durante la inferencia para ahorrar memoria y cálculo.
@torch.inference_mode()
def iter_maps(model, root, split, size, batch_size, device, limit=None, seed=42):
    # Carga la partición; limit y seed controlan la selección del subconjunto.
    dataset = RadiographDataset.labeled(root / split, size, limit, seed)
    # Agrupa imágenes en lotes sin modificar el orden del dataset.
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for images, labels, paths in loader:
        # Envía el lote al dispositivo del modelo (CPU o GPU).
        images = images.to(device)
        # Reconstruye y calcula mapas de anomalía; después los convierte a NumPy en CPU.
        maps = anomaly_maps(images, model(images)).cpu().numpy()
        for score_map, label, path in zip(maps, labels.tolist(), paths):
            # Entrega mapa, máscara, etiqueta y ruta de una imagen cada vez.
            yield score_map, load_mask(Path(path), size), label, path


def evaluate_localization(model, root: Path, output: Path, size: int, batch_size: int, device, limit=None, seed=42) -> dict:
    # Crea la carpeta de salida y sus directorios padre si no existen.
    output.mkdir(parents=True, exist_ok=True)
    # Candidatos fijos en [0,1]. El histograma
    # permite maximizar Dice micro sin retener todos los mapas en memoria.
    bins = 4096
    # Histogramas separados para puntuaciones de píxeles de lesión y de fondo.
    positive = np.zeros(bins+1, dtype=np.int64)
    negative = positive.copy()
    # Calibra el umbral únicamente con validación, sin utilizar test.
    for score, mask, _, _ in iter_maps(model, root, 'valid', size, batch_size, device, limit=limit, seed=seed):
        # Recorta a [0, 1] y asigna cada puntuación a uno de los 4097 niveles.
        index = np.minimum((score.clip(0, 1)*bins).astype(int), bins)
        # Acumula las frecuencias de puntuación según la máscara de referencia.
        positive += np.bincount(index[mask], minlength=bins+1)
        negative += np.bincount(index[~mask], minlength=bins+1)
    if positive.sum() == 0:
        raise ValueError('Validacion sin pixeles de lesion; no se puede calibrar.')
    # Sumas desde los niveles altos: píxeles detectados para cada umbral candidato.
    tp = positive[::-1].cumsum()[::-1]
    fp = negative[::-1].cumsum()[::-1]
    fn = positive.sum()-tp
    # Dice micro agrega los conteos de todas las imágenes antes de calcular la métrica.
    dice = 2*tp / np.maximum(2*tp+fp+fn, 1)
    # Elige el mayor Dice; si hay empate, argmax toma el umbral más bajo.
    selected = int(np.argmax(dice))
    threshold = selected/bins
    # Guarda configuración, huella del modelo y criterio de selección del umbral.
    report = dict(max_images_per_class=limit, subset_seed=seed, model_sha256=model_fingerprint(model), pixel_threshold=threshold, selection='maximum micro Dice on validation; grid step 1/4096',
                  image_size=size, mask='RGB > 0; alpha ignored; nearest resize; all pixels evaluated', splits={})
    # Evalúa ambas particiones con el mismo umbral seleccionado en validación.
    for split in ['valid', 'test']:
        rows = []
        for score, mask, label, path in iter_maps(model, root, split, size, batch_size, device, limit=limit, seed=seed):
            # Muestra el progreso cada 500 imágenes.
            if len(rows) % 500 == 0:
                print(f'{split}: {len(rows)} cortes evaluados', flush=True)
            # Marca como lesión puntuaciones >= umbral y guarda métricas por imagen.
            rows.append(dict(path=path, label=label, reference_pixels=int(mask.sum()),
                             predicted_pixels=int((score >= threshold).sum()), **overlap(score >= threshold, mask)))
        # Suma los conteos de píxeles de toda la partición para las métricas globales.
        totals = {key: sum(row[key] for row in rows) for key in ['tp','fp','fn']}
        t, f, n = totals.values()
        # Separa por contenido de la máscara: una máscara vacía puede tener etiqueta anómala.
        lesion_rows = [row for row in rows if row['reference_pixels'] > 0]
        normal_rows = [row for row in rows if row['reference_pixels'] == 0]
        # Métricas globales y promedios por imagen solo entre máscaras con lesión.
        # Incluye recuentos de máscaras vacías y falsos positivos en esas imágenes.
        report['splits'][split] = dict(images=len(rows), **totals, micro_dice=2*t/(2*t+f+n),
            micro_iou=t/(t+f+n), sensitivity=t/(t+n),
            mean_dice_lesion=float(np.mean([r['dice'] for r in lesion_rows])),
            mean_iou_lesion=float(np.mean([r['iou'] for r in lesion_rows])),
            lesion_reference_images=len(lesion_rows),
            empty_reference_images=len(normal_rows),
            anomalous_empty_reference_images=sum(r['label'] == 1 for r in normal_rows),
            false_positive_pixels_empty_reference=sum(r['fp'] for r in normal_rows))
        # Exporta un CSV con cabecera y una fila por imagen evaluada.
        with (output / f'{split}_localization.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(split, report['splits'][split], flush=True)
    # Guarda el informe completo en JSON y lo devuelve al código que llamó a la función.
    (output / 'localization.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    # Importa la arquitectura y prepara los argumentos de línea de comandos.
    from .dae import DAE
    parser = argparse.ArgumentParser(description=__doc__)
    # Rutas de datos, pesos entrenados y carpeta opcional de resultados.
    parser.add_argument('--data-root', type=Path, default=resolve_data_root())
    parser.add_argument('--checkpoint', type=Path, default=Path('results/dae_seed42/model.pt'))
    parser.add_argument('--output-dir', type=Path)
    # Número de imágenes procesadas a la vez.
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    # Limita a cuatro los hilos de CPU de PyTorch y elige GPU si hay CUDA disponible.
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Carga en CPU; weights_only restringe los tipos de objetos deserializados.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    # Recupera la configuración de canales base (64 por defecto).
    model = DAE(base_ch=int(checkpoint.get('dae_base_ch', 64))).to(device)
    # Restaura pesos y activa el comportamiento de evaluación de las capas.
    model.load_state_dict(checkpoint['model_state']); model.eval()
    # Usa el tamaño guardado (128 por defecto) y guarda junto al checkpoint si no hay salida.
    evaluate_localization(model, args.data_root, args.output_dir or args.checkpoint.parent, int(checkpoint.get('image_size',128)), args.batch_size, device)

# Ejecuta main solo cuando el módulo es el punto de entrada, no al importarlo.
if __name__ == '__main__':
    main()
