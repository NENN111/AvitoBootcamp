"""Делим фото HierText по наборам и создаём пары строк 0/180 градусов"""
from dataclasses import replace
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
from urllib.request import urlretrieve

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .config import Config
from .synthetic_data import manifests

DATA_DIR = Path('data/hiertext')
SPLITS = ['train', 'val', 'calibration', 'selection', 'holdout']
HIERTEXT_FILES = {
    'validation.jsonl.gz': (
        'https://raw.githubusercontent.com/google-research-datasets/hiertext/main/gt/validation.jsonl.gz',
        'ce7085d8bf24c2f13df852715c34494e6748cafcabb4bfc7a6e2bb320f58d23a',
    ),
    'validation.tgz': (
        'https://s3.amazonaws.com/open-images-dataset/ocr/validation.tgz',
        '1903446ab63a9be482b7f279cc54198904987efe2b30d3004225d48d1b84c4bf',
    ),
}


def ensure_hiertext_files() -> None:
    #Скачиваем исходные данные один раз, повторный запуск использует локальные файлы
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, (url, expected_sha) in HIERTEXT_FILES.items():
        path = DATA_DIR / name
        if path.exists():
            continue
        temporary = path.with_name(name + '.download')
        print(f'Скачиваю {name}...')
        try:
            urlretrieve(url, temporary)
            digest = hashlib.sha256()
            with temporary.open('rb') as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha:
                raise ValueError(f'Контрольная сумма {name} не совпала')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def eligible_lines(annotation: dict) -> list[dict]:
    lines = []
    for paragraph in annotation['paragraphs']:
        for index, line in enumerate(paragraph['lines']):
            if not line.get('legible') or line.get('vertical') or line.get('handwritten'):
                continue
            if len(line.get('text', '').strip()) < 2:
                continue
            points = np.asarray(line['vertices'], dtype=np.float32)
            if points.shape != (4, 2):
                continue
            width = (np.linalg.norm(points[1]-points[0]) + np.linalg.norm(points[2]-points[3])) / 2
            height = (np.linalg.norm(points[3]-points[0]) + np.linalg.norm(points[2]-points[1])) / 2
            angle = abs(math.degrees(math.atan2(points[1, 1]-points[0, 1],
                                                points[1, 0]-points[0, 0])))
            if width < 20 or height < 8 or not 1.2 <= width/height <= 30 or angle > 15:
                continue
            lines.append(dict(points=points, text=line['text'].strip()))
    return lines


def plan_real(config: Config) -> dict:
    ann = DATA_DIR/'validation.jsonl.gz'
    with gzip.open(ann, 'rt', encoding='utf-8') as handle:
        annotations = json.load(handle)['annotations']
    rng = np.random.default_rng(config.seed)
    rng.shuffle(annotations)
    doc_counts = {'train': 1300, 'val': 160, 'calibration': 88, 'selection': 88}
    doc_counts['holdout'] = len(annotations) - sum(doc_counts.values())
    # Половина изображений реальная, а для каждой строки делаем два поворота.
    wanted = dict(train=config.train_size//4, val=config.validation_size//4,
                  calibration=config.calibration_size//4,
                  selection=config.selection_size//4, holdout=config.holdout_size//4)
    plans, cursor = {}, 0
    for split, count in doc_counts.items():
        docs = annotations[cursor:cursor+count]
        cursor += count
        choices = []
        for annotation in docs:
            lines = eligible_lines(annotation)
            if len(lines) > 100:
                chosen = rng.choice(len(lines), size=100, replace=False)
                lines = [lines[i] for i in sorted(chosen)]
            for index, line in enumerate(lines):
                choices.append(dict(doc=annotation['image_id'], index=index, **line))
        rng.shuffle(choices)
        if len(choices) < wanted[split]:
            raise ValueError(f'{split}: {len(choices)} eligible real lines; need {wanted[split]}')
        plans[split] = choices[:wanted[split]]
    return plans


def extract_real(plans: dict[str, list[dict]], seed: int) -> dict[str, pd.DataFrame]:
    archive = DATA_DIR/'validation.tgz'
    plan_digest = hashlib.sha256()
    for name, entries in plans.items():
        for entry in entries:
            plan_digest.update(json.dumps([name, entry['doc'], entry['index'], entry['text'],
                                           entry['points'].tolist()], ensure_ascii=False).encode('utf-8'))
    plan_sha = plan_digest.hexdigest()
    image_lookup = {}
    for name, entries in plans.items():
        for index, line in enumerate(entries):
            image_lookup.setdefault(line['doc'], []).append((name, index, line))
    lock = DATA_DIR/'source.lock.json'
    expected_paths = [DATA_DIR/'crops'/name/f'{index:06d}.png'
                      for name, entries in plans.items() for index in range(len(entries))]
    if lock.exists():
        info = json.loads(lock.read_text(encoding='utf-8'))
        if (info.get('plan_sha256') == plan_sha and
                all(path.is_file() for path in expected_paths)):
            return _real_frames(plans, seed)
    with tarfile.open(archive, 'r:gz') as package:
        processed = set()
        for member in package:
            if not member.isfile():
                continue
            doc = Path(member.name).stem
            if doc not in image_lookup:
                continue
            source = package.extractfile(member)
            if source is None:
                continue
            content = source.read()
            with Image.open(io.BytesIO(content)) as im:
                pixels = np.asarray(im.convert('RGB'), dtype=np.uint8)
            for split, index, line in image_lookup[doc]:
                pts = np.asarray(line['points'], dtype=np.float32)
                width = max(1, round((np.linalg.norm(pts[1]-pts[0]) + np.linalg.norm(pts[2]-pts[3]))/2))
                height = max(1, round((np.linalg.norm(pts[3]-pts[0]) + np.linalg.norm(pts[2]-pts[1]))/2))
                dest = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
                crop = cv2.warpPerspective(pixels, cv2.getPerspectiveTransform(pts, dest),
                                           (width, height), borderMode=cv2.BORDER_REPLICATE)
                path = DATA_DIR/'crops'/split/f'{index:06d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(crop).save(path)
            processed.add(doc)
            if len(processed) % 200 == 0:
                print(f'Real source images processed: {len(processed)}/{len(image_lookup)}', flush=True)
    assert processed == set(image_lookup), f'Missing {len(set(image_lookup)-processed)} source images'
    source_info = {'plan_sha256': plan_sha}
    lock.write_text(json.dumps(source_info, indent=2), encoding='utf-8')
    return _real_frames(plans, seed)


def _real_frames(plans: dict[str, list[dict]], seed: int) -> dict[str, pd.DataFrame]:
    frames = {}
    for split, entries in plans.items():
        rows = []
        for index, line in enumerate(entries):
            base = f'real_{split}_{index:06d}'
            path = (DATA_DIR/'crops'/split/f'{index:06d}.png').as_posix()
            for label in (0, 1):
                rows.append(dict(sample_id=f'{base}_{label}', base_id=base,
                                 text=line['text'], font='HierText/real', label=label,
                                 render_seed=int(config_seed(split, index, seed)), split=split,
                                 source='real', path=path, source_doc=line['doc']))
        frames[split] = pd.DataFrame(rows)
    return frames


def config_seed(split: str, index: int, seed: int) -> int:
    return seed + SPLITS.index(split)*1000000 + index


def mixed_manifests(config: Config) -> dict[str, pd.DataFrame]:
    #Собираем train 50/50 и разделяем реальные строки по исходному фото
    ensure_hiertext_files()
    synthetic = manifests(replace(config, train_size=config.train_size//2,
        validation_size=config.validation_size//2,
        calibration_size=config.calibration_size//2,
        selection_size=config.selection_size//2,
        holdout_size=config.holdout_size//2))
    plans = plan_real(config)
    real = extract_real(plans, config.seed)
    folder = Path(config.run_dir)/'manifests'
    combined = {}
    for split, frame in synthetic.items():
        frame = frame.assign(source='synthetic', path='', source_doc='')
        other = real[split]
        assert len(frame) == len(other)
        all_rows = pd.concat([frame, other], ignore_index=True)
        pairs = [group for _, group in all_rows.groupby('base_id', sort=False)]
        rng = np.random.default_rng(config.seed + 600000 + len(split))
        order = rng.permutation(len(pairs))
        merged = pd.concat([pairs[i] for i in order], ignore_index=True)
        merged.to_csv(folder/f'{split}_manifest.csv', index=False, lineterminator='\n')
        combined[split] = merged
    return combined


if __name__ == '__main__':
    from .config import MIXED_CONFIG
    frames = mixed_manifests(MIXED_CONFIG)
    print({name: len(frame) for name, frame in frames.items()})
