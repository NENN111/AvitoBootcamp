from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import string

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from .config import Config

TRAIN_FONTS = ['DejaVuSans.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSerif.ttf', 'DejaVuSerif-Bold.ttf']
VAL_FONTS = ['DejaVuSans-Oblique.ttf', 'DejaVuSans-BoldOblique.ttf', 'DejaVuSansMono.ttf', 'DejaVuSansMono-Bold.ttf']
TEXT_BUCKETS = {'train': range(6), 'val': (6,), 'calibration': (7,),
                'selection': (8,), 'holdout': (9,)}


def make_text(rng: np.random.Generator, words: dict) -> str:
    kind = int(rng.integers(8))
    if kind == 0:
        return str(rng.integers(1, 100000))
    if kind == 1:
        return f'{rng.integers(1, 99999)}{rng.choice([" ₽", " руб.", " $", ".99", " EUR"])}'
    if kind == 2:
        return f'{rng.choice(["Размер", "Size", "Длина", "Ширина"])} {rng.integers(10, 200)}{rng.choice(["", " см", " mm", " XL"])}'
    if kind == 3:
        alphabet = list(string.ascii_uppercase + string.digits)
        return ''.join(rng.choice(alphabet, size=int(rng.integers(3, 15))))
    if kind == 4:
        pool = words['ru']
    elif kind == 5:
        pool = words['en']
    else:
        pool = words['ru'] + words['en']
    tokens = list(rng.choice(pool, size=int(rng.integers(1, 8))))
    if rng.random() < .5:
        tokens.append(str(rng.integers(1, 999)))
    text = ' '.join(tokens)
    return text.upper() if rng.random() < .25 else text


def manifests(config: Config) -> dict[str, pd.DataFrame]:
    """Разделяем тексты по хешу до создания поворотных пар."""
    words = json.loads(Path('assets/dictionaries/words.json').read_text(encoding='utf-8'))
    sizes = dict(train=config.train_size, val=config.validation_size,
                 calibration=config.calibration_size, selection=config.selection_size,
                 holdout=config.holdout_size)
    frames = {}
    folder = Path(config.run_dir) / 'manifests'
    folder.mkdir(parents=True, exist_ok=True)
    for split_index, (split, size) in enumerate(sizes.items()):
        assert size > 0 and size % 2 == 0
        rng = np.random.default_rng(config.seed + 1000 * split_index)
        fonts = TRAIN_FONTS if split == 'train' else VAL_FONTS
        allowed_buckets = TEXT_BUCKETS[split]
        rows, seen = [], set()
        while len(rows) < size:
            text = make_text(rng, words)
            bucket = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) % 10
            if bucket not in allowed_buckets or text in seen:
                continue
            seen.add(text)
            base_id = f'{split}_{len(rows)//2:06d}'
            font = str(rng.choice(fonts))
            render_seed = int(rng.integers(0, 2**31)) + split_index * 2**32
            for label in (0, 1):
                rows.append(dict(sample_id=f'{base_id}_{label}', base_id=base_id, text=text,
                                 font=font, label=label, render_seed=render_seed, split=split))
        frame = pd.DataFrame(rows)
        frame.to_csv(folder / f'{split}_manifest.csv', index=False, lineterminator='\n')
        frames[split] = frame
    return frames


@lru_cache(maxsize=1024)
def font_at(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(Path('assets/fonts') / name), size=size)


def render(row) -> Image.Image:
    """Сначала рисуем и искажаем текст, затем применяем точный поворот по метке."""
    rng = np.random.default_rng(int(row['render_seed']))
    font = font_at(row['font'], int(rng.integers(18, 64)))
    bbox = font.getbbox(row['text'])
    pad_x, pad_y = int(rng.integers(1, 14)), int(rng.integers(1, 8))
    w = int(bbox[2] - bbox[0] + 2 * pad_x)
    h = int(bbox[3] - bbox[1] + 2 * pad_y)
    bg = np.asarray(rng.integers(175, 256, size=3), dtype=np.uint8)
    fg = np.asarray(rng.integers(0, 100, size=3), dtype=np.uint8)
    if rng.random() < .2:
        bg, fg = fg, bg
    pixels = np.broadcast_to(bg, (h, w, 3)).astype(np.float32).copy()
    shade_start = -float(rng.uniform(0, 20))
    shade_end = float(rng.uniform(0, 20))
    pixels += np.linspace(shade_start, shade_end, w)[None, :, None]
    im = Image.fromarray(np.uint8(np.clip(pixels, 0, 255)))
    ImageDraw.Draw(im).text((pad_x-bbox[0], pad_y-bbox[1]), row['text'], font=font, fill=tuple(fg))
    im = im.rotate(float(rng.uniform(-3, 3)), resample=Image.Resampling.BICUBIC,
                   expand=False, fillcolor=tuple(bg))
    arr = np.asarray(im, dtype=np.uint8)
    if rng.random() < .4:
        src = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        dst = src + rng.uniform(-.035*h, .035*h, (4, 2)).astype(np.float32)
        arr = cv2.warpPerspective(arr, cv2.getPerspectiveTransform(src, dst), (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < .2:
        k = int(rng.choice([3, 5]))
        kernel = np.zeros((k, k), np.float32)
        kernel[k//2, :] = 1/k
        arr = cv2.filter2D(arr, -1, kernel)
    arr = np.uint8(np.clip(arr.astype(float) + rng.normal(0, rng.uniform(0, 7), arr.shape), 0, 255))
    im = Image.fromarray(arr)
    if rng.random() < .45:
        im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(.1, .8))))
    im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(.65, 1.3)))
    im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(.8, 1.2)))
    if rng.random() < .55:
        factor = float(rng.uniform(.35, .85))
        im = im.resize((max(2, int(w*factor)), max(8, int(h*factor))), Image.Resampling.BILINEAR)
        im = im.resize((w, h), Image.Resampling.BILINEAR)
    if rng.random() < .4:
        buffer = io.BytesIO()
        im.save(buffer, format='JPEG', quality=int(rng.integers(35, 96)))
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            im = encoded.convert('RGB')
    crop = int(rng.integers(0, max(1, pad_y)))
    im = im.crop((0, crop, im.width, im.height-crop))
    return im.transpose(Image.Transpose.ROTATE_180) if int(row['label']) else im


def render_sample(row) -> Image.Image:
    """Для реальных и синтетических кропов используем одинаковые метки поворота."""
    if row.get('source') == 'real':
        with Image.open(row['path']) as image:
            upright = image.convert('RGB')
        return upright.transpose(Image.Transpose.ROTATE_180) if int(row['label']) else upright
    return render(row)
