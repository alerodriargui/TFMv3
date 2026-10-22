"""Evaluacion espacial con mascaras de referencia; no modifica los pesos."""
from __future__ import annotations
import argparse
import hashlib
import csv
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .data import RadiographDataset, resolve_data_root


def model_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode('utf-8'))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_mask(image_path: Path, size: int) -> np.ndarray:
    if image_path.parent.name != 'img':
        raise ValueError(f'Imagen fuera de img/: {image_path}')
    path = image_path.parent.parent / 'label' / image_path.name
    with Image.open(path) as source:
        # Ignorar el canal alfa: es opaco tambien en el fondo.
        mask = source.convert('RGB').resize((size, size), Image.Resampling.NEAREST)
        return np.asarray(mask).max(axis=2) > 0


def anomaly_maps(images: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    foreground = F.avg_pool2d((images > .01).float(), 5, stride=1, padding=2) > .95
    error = (reconstruction - images).abs() * foreground
    patches = F.pad(error, (2, 2, 2, 2), mode='reflect').unfold(2, 5, 1).unfold(3, 5, 1)
    return patches.contiguous().view(*error.shape, 25).median(-1).values.amax(dim=1)


def overlap(prediction: np.ndarray, reference: np.ndarray) -> dict:
    tp = int((prediction & reference).sum())
    fp = int((prediction & ~reference).sum())
    fn = int((~prediction & reference).sum())
    return dict(tp=tp, fp=fp, fn=fn,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                sensitivity=tp/(tp+fn) if tp+fn else None)


@torch.inference_mode()
def iter_maps(model, root, split, size, batch_size, device, limit=None, seed=42):
    dataset = RadiographDataset.labeled(root / split, size, limit, seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for images, labels, paths in loader:
        images = images.to(device)
        maps = anomaly_maps(images, model(images)).cpu().numpy()
        for score_map, label, path in zip(maps, labels.tolist(), paths):
            yield score_map, load_mask(Path(path), size), label, path


def evaluate_localization(model, root: Path, output: Path, size: int, batch_size: int, device, limit=None, seed=42) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    # Candidatos fijos en [0,1]. El histograma
    # permite maximizar Dice micro sin retener todos los mapas en memoria.
    bins = 4096
    positive = np.zeros(bins+1, dtype=np.int64)
    negative = positive.copy()
    for score, mask, _, _ in iter_maps(model, root, 'valid', size, batch_size, device, limit=limit, seed=seed):
        index = np.minimum((score.clip(0, 1)*bins).astype(int), bins)
        positive += np.bincount(index[mask], minlength=bins+1)
        negative += np.bincount(index[~mask], minlength=bins+1)
    if positive.sum() == 0:
        raise ValueError('Validacion sin pixeles de lesion; no se puede calibrar.')
    tp = positive[::-1].cumsum()[::-1]
    fp = negative[::-1].cumsum()[::-1]
    fn = positive.sum()-tp
    dice = 2*tp / np.maximum(2*tp+fp+fn, 1)
    selected = int(np.argmax(dice))
    threshold = selected/bins
    report = dict(max_images_per_class=limit, subset_seed=seed, model_sha256=model_fingerprint(model), pixel_threshold=threshold, selection='maximum micro Dice on validation; grid step 1/4096',
                  image_size=size, mask='RGB > 0; alpha ignored; nearest resize; all pixels evaluated', splits={})
    for split in ['valid', 'test']:
        rows = []
        for score, mask, label, path in iter_maps(model, root, split, size, batch_size, device, limit=limit, seed=seed):
            if len(rows) % 500 == 0:
                print(f'{split}: {len(rows)} cortes evaluados', flush=True)
            rows.append(dict(path=path, label=label, reference_pixels=int(mask.sum()),
                             predicted_pixels=int((score >= threshold).sum()), **overlap(score >= threshold, mask)))
        totals = {key: sum(row[key] for row in rows) for key in ['tp','fp','fn']}
        t, f, n = totals.values()
        lesion_rows = [row for row in rows if row['reference_pixels'] > 0]
        normal_rows = [row for row in rows if row['reference_pixels'] == 0]
        report['splits'][split] = dict(images=len(rows), **totals, micro_dice=2*t/(2*t+f+n),
            micro_iou=t/(t+f+n), sensitivity=t/(t+n),
            mean_dice_lesion=float(np.mean([r['dice'] for r in lesion_rows])),
            mean_iou_lesion=float(np.mean([r['iou'] for r in lesion_rows])),
            lesion_reference_images=len(lesion_rows),
            empty_reference_images=len(normal_rows),
            anomalous_empty_reference_images=sum(r['label'] == 1 for r in normal_rows),
            false_positive_pixels_empty_reference=sum(r['fp'] for r in normal_rows))
        with (output / f'{split}_localization.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(split, report['splits'][split], flush=True)
    (output / 'localization.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    from .dae import DAE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=resolve_data_root())
    parser.add_argument('--checkpoint', type=Path, default=Path('results/dae_seed42/model.pt'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = DAE(base_ch=int(checkpoint.get('dae_base_ch', 64))).to(device)
    model.load_state_dict(checkpoint['model_state']); model.eval()
    evaluate_localization(model, args.data_root, args.output_dir or args.checkpoint.parent, int(checkpoint.get('image_size',128)), args.batch_size, device)

if __name__ == '__main__':
    main()
