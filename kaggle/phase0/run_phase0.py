"""LACE Phase 0 Kaggle runner.

This script is intentionally self-contained because Kaggle uploads the kernel
code file and metadata, not the whole local repository. It performs a small
latent-cache sanity check for the LACE research program:

    text -> frozen T5 encoder -> h0 -> pooled z1/z2/z3 -> cache + metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


FALLBACK_TEXTS = [
    "Language generation is not only a sequence of tokens, but a layered representation of intent, meaning, and surface form.",
    "A compression path should preserve semantic structure while discarding recoverable surface detail.",
    "Diffusion models for text often corrupt symbols, but LACE studies information rate in latent space.",
    "The first experiment checks whether frozen encoder states can be cached and compressed consistently.",
    "A useful small-scale result should expose shape errors, cache failures, and metric instability early.",
    "Latent adaptive compression and expansion separates the research question from large model training.",
    "The model should be evaluated first as a representation trajectory before it becomes a text generator.",
    "Gaussian noise, random masking, and learned compression define different forward processes.",
    "The Phase 0 run is deliberately small enough to execute on Kaggle or Colab resources.",
    "If the latent cache is stable, later experiments can compare pooling, Gaussian noise, and LACE-small.",
    "Information-rate schedules can be approximated with token budgets before precise mutual information estimation.",
    "The immediate goal is not SOTA performance, but a clean experimental loop that can scale later.",
]


@dataclass(frozen=True)
class Phase0Config:
    model_name: str = "t5-small"
    max_samples: int = 128
    max_length: int = 128
    batch_size: int = 16
    output_dir: str = "/kaggle/working/lace_phase0"
    input_text_file: str | None = None
    seed: int = 42
    stage_tokens: tuple[int, ...] = (64, 32, 16)


def parse_args() -> Phase0Config:
    parser = argparse.ArgumentParser(description="Run LACE Phase 0 latent-cache sanity check.")
    parser.add_argument("--model-name", default=Phase0Config.model_name)
    parser.add_argument("--max-samples", type=int, default=Phase0Config.max_samples)
    parser.add_argument("--max-length", type=int, default=Phase0Config.max_length)
    parser.add_argument("--batch-size", type=int, default=Phase0Config.batch_size)
    parser.add_argument("--output-dir", default=os.environ.get("LACE_OUTPUT_DIR", Phase0Config.output_dir))
    parser.add_argument("--input-text-file", default=os.environ.get("LACE_INPUT_TEXT_FILE"))
    parser.add_argument("--seed", type=int, default=Phase0Config.seed)
    parser.add_argument(
        "--stage-tokens",
        default="64,32,16",
        help="Comma-separated latent token counts for z stages.",
    )
    args = parser.parse_args()
    stage_tokens = tuple(int(item.strip()) for item in args.stage_tokens.split(",") if item.strip())
    return Phase0Config(
        model_name=args.model_name,
        max_samples=args.max_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        input_text_file=args.input_text_file,
        seed=args.seed,
        stage_tokens=stage_tokens,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def read_text_file(path: Path, max_samples: int) -> list[str]:
    if path.suffix.lower() == ".csv":
        return read_csv_texts(path, max_samples)
    if path.suffix.lower() in {".jsonl", ".json"}:
        return read_json_texts(path, max_samples)
    texts: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            text = line.strip()
            if len(text) >= 20:
                texts.append(text)
            if len(texts) >= max_samples:
                break
    return texts


def read_csv_texts(path: Path, max_samples: int) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values = [value.strip() for value in row.values() if isinstance(value, str)]
            values.sort(key=len, reverse=True)
            if values and len(values[0]) >= 20:
                texts.append(values[0])
            if len(texts) >= max_samples:
                break
    return texts


def read_json_texts(path: Path, max_samples: int) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            values = [value.strip() for value in row.values() if isinstance(value, str)]
            values.sort(key=len, reverse=True)
            if values and len(values[0]) >= 20:
                texts.append(values[0])
            if len(texts) >= max_samples:
                break
    return texts


def discover_kaggle_texts(max_samples: int) -> list[str]:
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return []
    candidate_suffixes = {".txt", ".csv", ".jsonl", ".json"}
    for path in sorted(input_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in candidate_suffixes:
            texts = read_text_file(path, max_samples)
            if texts:
                print(f"Loaded {len(texts)} texts from {path}")
                return texts
    return []


def load_texts(config: Phase0Config) -> list[str]:
    if config.input_text_file:
        texts = read_text_file(Path(config.input_text_file), config.max_samples)
    else:
        texts = discover_kaggle_texts(config.max_samples)

    if not texts:
        repeats = (config.max_samples // len(FALLBACK_TEXTS)) + 1
        texts = (FALLBACK_TEXTS * repeats)[: config.max_samples]
        print(f"Using fallback text corpus with {len(texts)} samples.")

    return texts[: config.max_samples]


def require_runtime_dependencies() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Run this kernel with a Kaggle GPU image.") from exc

    try:
        import transformers  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "sentencepiece"])


def adaptive_pool_tokens(hidden_states, target_tokens: int):
    import torch.nn.functional as functional

    channels_first = hidden_states.float().transpose(1, 2)
    pooled = functional.adaptive_avg_pool1d(channels_first, target_tokens)
    return pooled.transpose(1, 2).contiguous()


def expand_tokens(latents, target_tokens: int):
    import torch.nn.functional as functional

    channels_first = latents.float().transpose(1, 2)
    expanded = functional.interpolate(channels_first, size=target_tokens, mode="linear", align_corners=False)
    return expanded.transpose(1, 2).contiguous()


def reconstruction_metrics(reference, reconstructed, attention_mask):
    import torch
    import torch.nn.functional as functional

    active_tokens = attention_mask.bool()
    reference_active = reference.float()[active_tokens]
    reconstructed_active = reconstructed.float()[active_tokens]
    mse = functional.mse_loss(reconstructed_active, reference_active).item()
    cosine = functional.cosine_similarity(reconstructed_active, reference_active, dim=-1).mean().item()
    finite = bool(torch.isfinite(reconstructed_active).all().item())
    return {"mse": mse, "cosine": cosine, "finite": finite}


def encode_texts(config: Phase0Config, texts: list[str]):
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    encoder = T5EncoderModel.from_pretrained(config.model_name)
    encoder.to(device)
    encoder.eval()

    hidden_batches = []
    mask_batches = []
    for start in range(0, len(texts), config.batch_size):
        batch_texts = texts[start : start + config.batch_size]
        batch = tokenizer(
            batch_texts,
            max_length=config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            hidden = encoder(**batch).last_hidden_state
        hidden_batches.append(hidden.cpu().to(torch.float16))
        mask_batches.append(batch["attention_mask"].cpu().to(torch.int16))

    hidden_states = torch.cat(hidden_batches, dim=0)
    attention_mask = torch.cat(mask_batches, dim=0)
    return hidden_states, attention_mask, str(device)


def save_outputs(config: Phase0Config, texts: list[str], hidden_states, attention_mask, device: str) -> dict:
    import torch

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_metrics = {}
    for index, target_tokens in enumerate(config.stage_tokens, start=1):
        latents = adaptive_pool_tokens(hidden_states, target_tokens)
        reconstructed = expand_tokens(latents, config.max_length)
        stage_name = f"z{index}"
        stage_metrics[stage_name] = {
            "target_tokens": target_tokens,
            "shape": list(latents.shape),
            **reconstruction_metrics(hidden_states, reconstructed, attention_mask),
        }

    cache_path = output_dir / "latent_cache.pt"
    cache_payload = {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "texts": texts,
        "config": asdict(config),
    }
    torch.save(cache_payload, cache_path)
    reloaded = torch.load(cache_path, map_location="cpu")
    cache_allclose = bool(torch.equal(reloaded["hidden_states"], hidden_states))

    metrics = {
        "phase": "phase0",
        "model_name": config.model_name,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "sample_count": len(texts),
        "max_length": config.max_length,
        "hidden_shape": list(hidden_states.shape),
        "hidden_dtype": str(hidden_states.dtype),
        "active_tokens": int(attention_mask.sum().item()),
        "cache_path": str(cache_path),
        "cache_allclose": cache_allclose,
        "stages": stage_metrics,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary_lines = [
        "# LACE Phase 0 Summary",
        "",
        f"- Model: `{config.model_name}`",
        f"- Device: `{device}`",
        f"- Samples: `{len(texts)}`",
        f"- Hidden shape: `{list(hidden_states.shape)}`",
        f"- Cache reload exact match: `{cache_allclose}`",
        "",
        "## Stage Metrics",
        "",
        "| Stage | Tokens | MSE | Cosine | Finite |",
        "|---|---:|---:|---:|---|",
    ]
    for stage_name, values in stage_metrics.items():
        summary_lines.append(
            f"| {stage_name} | {values['target_tokens']} | {values['mse']:.6f} | "
            f"{values['cosine']:.6f} | {values['finite']} |"
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    require_runtime_dependencies()
    texts = load_texts(config)
    hidden_states, attention_mask, device = encode_texts(config, texts)
    metrics = save_outputs(config, texts, hidden_states, attention_mask, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

