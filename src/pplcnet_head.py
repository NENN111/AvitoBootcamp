"""Подготовка признаков и экспорт дообученной модели"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from PIL import Image
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from .calibration import calibrate, fit_temperature, safe_logit
from .synthetic_data import render_sample

MODEL_SHA = "94a6a0a0425f2b5f08b5df72086f2d72abe40f1d22f6d12d2cd83674f11f2ff3"
MODEL_PATH = Path("checkpoints/pplcnet_x0_25_official.onnx")
FEATURE_PATH = Path("results/pplcnet_x0_25/features.onnx")
RUN_DIR = Path("results/pplcnet_x0_25")
MANIFEST_DIR = Path("results/mixed/manifests")
MEAN = np.array([.485, .456, .406], np.float32)[:, None, None]
STD = np.array([.229, .224, .225], np.float32)[:, None, None]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_model() -> Path:
    if sha256(MODEL_PATH) != MODEL_SHA:
        raise ValueError("Контрольная сумма исходных весов не совпала")
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not FEATURE_PATH.exists():
        model = onnx.load(str(MODEL_PATH))
        model.graph.output.clear()
        model.graph.output.append(onnx.helper.make_tensor_value_info(
            "p2o.pd_op.flatten.0.0", onnx.TensorProto.FLOAT, [None, 1280]))
        onnx.checker.check_model(model)
        onnx.save(model, str(FEATURE_PATH))
    return FEATURE_PATH


def session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def preprocess(image: Image.Image) -> np.ndarray:
    # Используем размер и нормализацию из официального inference.yml.
    rgb = np.asarray(image.convert("RGB"))
    rgb = cv2.resize(rgb, (160, 80), interpolation=cv2.INTER_LINEAR)
    pixels = np.divide(rgb.transpose(2, 0, 1).astype(np.float32), 255.0)
    return (pixels - MEAN) / STD


def extract_split(split: str, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    frame = pd.read_csv(MANIFEST_DIR / f"{split}_manifest.csv")
    assert len(frame) % 2 == 0
    assert frame.label.value_counts().to_dict() == {0: len(frame)//2, 1: len(frame)//2}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    cache = RUN_DIR / f"{split}_features.npy"
    meta = RUN_DIR / f"{split}_features.json"
    manifest_sha = sha256(MANIFEST_DIR / f"{split}_manifest.csv")
    expected_meta = {"model_sha256": MODEL_SHA, "manifest_sha256": manifest_sha,
                     "rows": len(frame), "shape": [len(frame), 1280],
                     "provider": "CPUExecutionProvider"}
    if cache.exists() and meta.exists():
        saved = json.loads(meta.read_text(encoding="utf-8"))
        if saved == expected_meta:
            return np.load(cache, mmap_mode="r"), frame.label.to_numpy(np.float32), frame
    inference = session(feature_model())
    matrix = np.lib.format.open_memmap(cache.with_suffix(".tmp.npy"), mode="w+",
                                      dtype=np.float32, shape=(len(frame), 1280))
    start = time.perf_counter()
    for offset in range(0, len(frame), batch_size):
        rows = frame.iloc[offset:offset + batch_size].to_dict("records")
        images = []
        for row in rows:
            images.append(preprocess(render_sample(row)))
        features = np.asarray(inference.run(None, {"x": np.stack(images)})[0])
        matrix[offset:offset + len(rows)] = features
        if offset % 2048 == 0:
            print(f"{split}: {offset}/{len(frame)}, {time.perf_counter()-start:.1f} с", flush=True)
    matrix.flush()
    del matrix
    cache.with_suffix(".tmp.npy").replace(cache)
    meta.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return np.load(cache, mmap_mode="r"), frame.label.to_numpy(np.float32), frame


def official_head() -> tuple[np.ndarray, float]:
    model = onnx.load(str(MODEL_PATH))
    constants = {node.output[0]: onnx.numpy_helper.to_array(node.attribute[0].t)
                 for node in model.graph.node if node.op_type == "Constant" and node.attribute}
    weights = constants["linear_0.w_0_deepcopy_144"]
    bias = constants["linear_0.b_0_deepcopy_145"]
    return (weights[:, 1] - weights[:, 0]).astype(np.float32), float(bias[1] - bias[0])


def score(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probability = expit(logits)
    return {"brier": float(np.mean((probability - y)**2)),
            "accuracy": float(np.mean((probability >= .5) == y)),
            "auc": float(roc_auc_score(y, probability))}


def head_logits(features: np.ndarray) -> np.ndarray:
    head = np.load(RUN_DIR / "head.npz")
    return np.asarray(features) @ head["weight"] + float(head["bias"].item())


def export_head() -> Path:
    #Заменяем только последний полносвязный слой
    model = onnx.load(str(MODEL_PATH))
    nodes = {node.output[0]: node for node in model.graph.node}
    head = np.load(RUN_DIR / "head.npz")
    weights = np.zeros((1280, 2), np.float32)
    weights[:, 1] = head["weight"]
    bias = np.array([0, head["bias"].item()], np.float32)
    nodes["linear_0.w_0_deepcopy_144"].attribute[0].t.CopyFrom(onnx.numpy_helper.from_array(weights))
    nodes["linear_0.b_0_deepcopy_145"].attribute[0].t.CopyFrom(onnx.numpy_helper.from_array(bias))
    onnx.checker.check_model(model)
    path = RUN_DIR / "fine_tuned.onnx"
    onnx.save(model, str(path))
    return path


def paired_logits(logits: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    opposite = np.empty(len(frame), dtype=int)
    for _, pair in frame.groupby("base_id"):
        first, second = pair.index
        opposite[first] = second
        opposite[second] = first
    return logits[opposite]


def choose_calibration() -> None:
    x_cal, y_cal, _ = extract_split("calibration")
    x_sel, y_sel, sel_frame = extract_split("selection")
    cal_logits = head_logits(x_cal).astype(np.float64)
    sel_logits = head_logits(x_sel).astype(np.float64)
    calibration = fit_temperature(cal_logits, y_cal)
    normal = calibrate(sel_logits, calibration)
    rotated = calibrate(paired_logits(sel_logits, sel_frame), calibration)
    results = []
    for tta, predictions in [(False, normal), (True, .5 * (normal + 1 - rotated))]:
        results.append({"tta": tta, **score(y_sel, safe_logit(predictions))})
    table = pd.DataFrame(results).sort_values("brier")
    table.to_csv(RUN_DIR / "calibration_selection.csv", index=False)
    best = table.iloc[0]
    selection = {"model": "PP-LCNet_x0_25_textline_ori", "method": "temperature",
                 "calibration": calibration, "tta": bool(best.tta),
                 "selection": {"brier": float(best.brier), "accuracy": float(best.accuracy),
                               "auc": float(best.auc)}}
    (RUN_DIR / "selected.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print("Выбрано:", selection, flush=True)


def evaluate_exported_holdout() -> None:
    #Проверяем экспортированную ONNX-модель тем же путём, что и тестовые прогнозы
    frame = pd.read_csv(MANIFEST_DIR / "holdout_manifest.csv")
    selected = json.loads((RUN_DIR / "selected.json").read_text(encoding="utf-8"))
    inference = session(RUN_DIR / "fine_tuned.onnx")
    probabilities = []
    for offset in range(0, len(frame), 32):
        rows = frame.iloc[offset:offset + 32].to_dict("records")
        prepared = np.stack([preprocess(render_sample(row)) for row in rows])
        raw = np.asarray(inference.run(None, {"x": prepared})[0])[:, 1]
        p = calibrate(safe_logit(raw), selected["calibration"])
        if selected["tta"]:
            rotated = np.asarray(inference.run(None, {"x": prepared[:, :, ::-1, ::-1].copy()})[0])[:, 1]
            p = .5 * (p + 1 - calibrate(safe_logit(rotated), selected["calibration"]))
        probabilities.extend(p.tolist())
    y = frame.label.to_numpy(np.float32)
    p = np.asarray(probabilities)
    records = []
    for source in ["all", "real", "synthetic"]:
        mask = np.ones(len(y), bool) if source == "all" else frame.source.eq(source).to_numpy()
        records.append({"source": source, "images": int(mask.sum()),
                        **score(y[mask], safe_logit(p[mask]))})
    report = pd.DataFrame(records)
    report.to_csv(RUN_DIR / "holdout_exported_metrics.csv", index=False)
    print(report.to_string(index=False), flush=True)
