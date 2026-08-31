"""Benchmark dynamic SuperTonic MNN at VE total_step=1..8."""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import MNN
import numpy as np

sys.path.insert(0, "/home/jhx/Projects/AIGC/supertonic-neko")
from supertonic.data.tokenizer import ZhTokenizer

SR = 44100
MAX_L = 100
CHUNK = 3072
INDEXER = Path("/home/jhx/Projects/AIGC/neko-speech/ref_codes/models/supertonic-3/onnx/unicode_indexer.json")
STYLE = Path("/home/jhx/Projects/AIGC/supertonic-neko/pretrained/catgirl_style.json")
CASES = [
    ("normal", "今天过得怎么样？"),
    ("catgirl", "主人回来啦，猫娘等你很久了喵！"),
    ("tongue", "扁担长，板凳宽，板凳没有扁担长。"),
]


def load_style():
    obj = json.loads(STYLE.read_text())
    return (np.asarray(obj["style_ttl"]["data"], np.float32)[None],
            np.asarray(obj["style_dp"]["data"], np.float32)[None])


def frontend(text):
    tok = ZhTokenizer(str(INDEXER), frontend="parallel_pinyin", max_len=256)
    tid, mask, pid, tone, pros = tok([text])
    return {
        "text_ids": tid.numpy().astype(np.int32),
        "pinyin_ids": pid.numpy().astype(np.int32),
        "tone_ids": tone.numpy().astype(np.int32),
        "prosody_ids": pros.numpy().astype(np.int32),
        "text_mask": mask.numpy().astype(np.float32),
    }


class MNNModel:
    def __init__(self, root: Path, name: str):
        self.interpreter = MNN.Interpreter(str(root / (name + ".mnn")))
        self.session = self.interpreter.createSession()
        self.shape_key = None

    def run(self, arrays):
        inputs = self.interpreter.getSessionInputAll(self.session)
        key = tuple(sorted((n, tuple(a.shape)) for n, a in arrays.items()))
        if key != self.shape_key:
            for name, array in arrays.items():
                self.interpreter.resizeTensor(inputs[name], tuple(array.shape))
            self.interpreter.resizeSession(self.session)
            inputs = self.interpreter.getSessionInputAll(self.session)
            self.shape_key = key
        for name, array in arrays.items():
            tensor = inputs[name]
            host = MNN.Tensor(
                tensor.getShape(), tensor.getDataType(),
                np.ascontiguousarray(array), MNN.Tensor_DimensionType_Caffe,
            )
            tensor.copyFromHostTensor(host)
        self.interpreter.runSession(self.session)
        outputs = {}
        for name, tensor in self.interpreter.getSessionOutputAll(self.session).items():
            host = MNN.Tensor(
                tensor.getShape(), tensor.getDataType(),
                np.zeros(tensor.getShape(), np.float32), MNN.Tensor_DimensionType_Caffe,
            )
            tensor.copyToHostTensor(host)
            outputs[name] = host.getNumpyData().copy()
        return outputs


def write_wav(path: Path, samples: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(pcm.tobytes())


def synth(models, text, steps, style_ttl, style_dp, seed):
    raw = frontend(text)
    t0 = time.perf_counter()
    dur = float(models["dp"].run({
        "text_ids": raw["text_ids"], "style_dp": style_dp,
        "text_mask": raw["text_mask"],
    })["duration"].reshape(-1)[0])
    valid_l = max(1, min(MAX_L, int(max(2.5, min(dur, MAX_L * CHUNK / SR)) * SR / CHUNK)))
    text_emb = models["te"].run({**raw, "style_ttl": style_ttl})["text_emb"]
    rng = np.random.RandomState(seed)
    x = rng.randn(1, 144, valid_l).astype(np.float32)
    mask = np.ones((1, 1, valid_l), np.float32)
    ve_ms = 0.0
    for step in range(steps):
        tick = time.perf_counter()
        x = models["ve"].run({
            "noisy_latent": x, "text_emb": text_emb, "style_ttl": style_ttl,
            "latent_mask": mask, "text_mask": raw["text_mask"],
            "current_step": np.array([step], np.float32),
            "total_step": np.array([steps], np.float32),
        })["denoised"]
        ve_ms += (time.perf_counter() - tick) * 1000
    wav = models["vocoder"].run({"latent": x})["wav"].reshape(-1).astype(np.float32)
    total_ms = (time.perf_counter() - t0) * 1000
    audio_sec = wav.size / SR
    return wav, {
        "text": text, "tokens": int(raw["text_ids"].shape[1]), "valid_l": valid_l,
        "duration_pred_sec": dur, "audio_sec": audio_sec,
        "dp_te_vocoder_total_ms": total_ms - ve_ms, "ve_ms": ve_ms,
        "total_ms": total_ms, "rtf": total_ms / 1000 / audio_sec,
        "peak": float(np.max(np.abs(wav))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    style_ttl, style_dp = load_style()
    models = {name: MNNModel(args.model_dir, name) for name in ("dp", "te", "ve", "vocoder")}
    report = {"config": {"sample_rate": SR, "steps": list(range(1, 9))}, "cases": []}
    for index, (category, text) in enumerate(CASES):
        row = {"category": category, "text": text, "steps": {}}
        for steps in range(1, 9):
            print(f"{category} step={steps}: {text}", flush=True)
            wav, metrics = synth(models, text, steps, style_ttl, style_dp, args.seed + index)
            write_wav(args.out_dir / f"step{steps:02d}" / f"{index:02d}_{category}.wav", wav)
            row["steps"][str(steps)] = metrics
        report["cases"].append(row)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[done] {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
