"""LACE Phase 2B Kaggle runner.

Phase 2B validates the Phase 2 forward-process signal under stricter controls:

    Does the average-pooling compression signal survive matched Gaussian
    calibration, a larger training budget, stricter gates, and decoder controls?

The script keeps Phase 2's fixed compression/corruption comparison but adds a
lightweight token-head bridge and an h0 decoder control so generation failures
can be separated from hidden-state reconstruction failures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FALLBACK_TEXTS = [
    "Language generation is a layered process of intent, meaning, discourse structure, and surface realization.",
    "A useful compression path should preserve semantic structure while discarding recoverable surface detail.",
    "Diffusion models for text often corrupt symbols, while LACE studies information rate in latent space.",
    "The forward process can be interpreted as a schedule over information rate instead of random destruction.",
    "Average pooling is not novel by itself, but it can test whether compression is a useful diffusion path.",
    "A fair experiment must compare compression, token dropping, and Gaussian noise with the same reverse model.",
    "Recovered latent states should eventually support token prediction and language generation.",
    "The research question is not whether autoencoders reconstruct, but whether compression improves diffusion learning.",
    "A small Kaggle experiment should expose shape errors, weak latent use, and unreliable generation bridges early.",
    "If compression produces a better reverse trajectory, adaptive compression can be studied later.",
    "The first generation bridge can be teacher-forced token likelihood rather than full open-ended sampling.",
    "A language diffusion model should recover more detailed representations step by step during reverse expansion.",
    "Random selection preserves a token budget but removes information without a structured compression rule.",
    "Gaussian noise preserves length but corrupts the representation in a way that may be harder to reverse.",
    "Phase 2 should produce evidence about forward process design, not a claim of final model quality.",
    "Compression, not corruption, is the hypothesis that must be tested under controlled conditions.",
]


@dataclass(frozen=True)
class Phase2BConfig:
    model_name: str = "t5-small"
    max_samples: int = 2048
    max_length: int = 128
    encode_batch_size: int = 16
    train_batch_size: int = 16
    epochs: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    output_dir: str = "/kaggle/working/lace_phase2b"
    input_text_file: str | None = None
    use_hf_dataset: bool = True
    hf_dataset_name: str = "wikitext"
    hf_dataset_config: str = "wikitext-2-raw-v1"
    hf_dataset_split: str = "train"
    seed: int = 42
    stage_tokens: tuple[int, ...] = (64, 32, 16)
    forward_conditions: tuple[str, ...] = ("average_pool", "strided_select", "random_select", "gaussian_noise")
    sigma_values: tuple[float, ...] = (0.05, 0.10, 0.15)
    calibration_sigmas: tuple[float, ...] = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20)
    use_matched_gaussian: bool = True
    lambda_cos: float = 0.1
    lambda_var: float = 0.01
    perturbation_std: float = 0.05
    ablation_fraction: float = 0.25
    enable_generation_bridge: bool = True
    generation_eval_batches: int = 1
    generation_sample_count: int = 4
    generation_max_length: int = 64
    enable_token_head_bridge: bool = True
    token_head_epochs: int = 1
    token_head_learning_rate: float = 1e-3
    token_head_train_batches: int = 12
    token_head_eval_batches: int = 2
    strict_gate_min_evidence: int = 2


Phase2Config = Phase2BConfig


def parse_stage_tokens(value: str) -> tuple[int, ...]:
    tokens = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not tokens:
        raise ValueError("At least one stage token count is required.")
    if any(token <= 0 for token in tokens):
        raise ValueError("Stage token counts must be positive.")
    if any(left <= right for left, right in zip(tokens, tokens[1:])):
        raise ValueError("Stage token counts must be strictly decreasing.")
    return tokens


def parse_forward_conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    valid = {"average_pool", "strided_select", "random_select", "gaussian_noise"}
    invalid = [condition for condition in conditions if condition not in valid]
    if invalid:
        raise ValueError(f"Invalid forward conditions: {invalid}")
    return conditions or ("average_pool",)


def parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("At least one float value is required.")
    if any(item <= 0 for item in values):
        raise ValueError("Float values must be positive.")
    return values


def stage_names(stage_tokens: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"z{index}" for index in range(1, len(stage_tokens) + 1))


def compute_strided_indices(input_tokens: int, target_tokens: int) -> list[int]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive.")
    if target_tokens > input_tokens:
        raise ValueError("target_tokens cannot exceed input_tokens.")
    if target_tokens == input_tokens:
        return list(range(input_tokens))
    if input_tokens % target_tokens == 0:
        step = input_tokens // target_tokens
        return list(range(0, input_tokens, step))[:target_tokens]
    if target_tokens == 1:
        return [0]
    return [round(index * (input_tokens - 1) / (target_tokens - 1)) for index in range(target_tokens)]


def choose_matched_sigmas(stage_losses: dict[str, float], sigma_losses: dict[str, float]) -> dict[str, dict[str, float | str]]:
    matched: dict[str, dict[str, float | str]] = {}
    for stage_name, stage_loss in stage_losses.items():
        sigma_name, sigma_loss = min(sigma_losses.items(), key=lambda item: abs(float(item[1]) - float(stage_loss)))
        matched[stage_name] = {
            "sigma_name": sigma_name,
            "stage_initial_loss": float(stage_loss),
            "sigma_initial_loss": float(sigma_loss),
            "absolute_gap": abs(float(sigma_loss) - float(stage_loss)),
        }
    return matched


def sigma_name_to_float(sigma_name: str) -> float:
    if not sigma_name.startswith("sigma_"):
        raise ValueError(f"Invalid sigma name: {sigma_name}")
    return float(sigma_name.removeprefix("sigma_"))


def resolve_stage_sigmas(
    forward_metadata: dict[str, Any],
    stage_names: tuple[str, ...],
    fallback_sigmas: tuple[float, ...],
    use_matched: bool,
) -> dict[str, float]:
    if len(stage_names) != len(fallback_sigmas):
        raise ValueError("stage_names and fallback_sigmas must have the same length.")

    if not use_matched:
        return {stage_name: float(fallback_sigmas[index]) for index, stage_name in enumerate(stage_names)}

    matched_sigmas = forward_metadata.get("calibration", {}).get("matched_sigmas", {})
    resolved: dict[str, float] = {}
    for index, stage_name in enumerate(stage_names):
        matched = matched_sigmas.get(stage_name)
        if isinstance(matched, dict) and isinstance(matched.get("sigma_name"), str):
            resolved[stage_name] = sigma_name_to_float(matched["sigma_name"])
        else:
            resolved[stage_name] = float(fallback_sigmas[index])
    return resolved


def parse_args() -> Phase2BConfig:
    parser = argparse.ArgumentParser(description="Run LACE Phase 2B calibrated validation.")
    parser.add_argument("--model-name", default=Phase2BConfig.model_name)
    parser.add_argument("--max-samples", type=int, default=Phase2BConfig.max_samples)
    parser.add_argument("--max-length", type=int, default=Phase2BConfig.max_length)
    parser.add_argument("--encode-batch-size", type=int, default=Phase2BConfig.encode_batch_size)
    parser.add_argument("--train-batch-size", type=int, default=Phase2BConfig.train_batch_size)
    parser.add_argument("--epochs", type=int, default=Phase2BConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=Phase2BConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Phase2BConfig.weight_decay)
    parser.add_argument("--val-fraction", type=float, default=Phase2BConfig.val_fraction)
    parser.add_argument("--output-dir", default=os.environ.get("LACE_OUTPUT_DIR", Phase2BConfig.output_dir))
    parser.add_argument("--input-text-file", default=os.environ.get("LACE_INPUT_TEXT_FILE"))
    parser.add_argument("--use-hf-dataset", action=argparse.BooleanOptionalAction, default=Phase2BConfig.use_hf_dataset)
    parser.add_argument("--hf-dataset-name", default=Phase2BConfig.hf_dataset_name)
    parser.add_argument("--hf-dataset-config", default=Phase2BConfig.hf_dataset_config)
    parser.add_argument("--hf-dataset-split", default=Phase2BConfig.hf_dataset_split)
    parser.add_argument("--seed", type=int, default=Phase2BConfig.seed)
    parser.add_argument("--stage-tokens", default="64,32,16")
    parser.add_argument("--forward-conditions", default="average_pool,strided_select,random_select,gaussian_noise")
    parser.add_argument("--sigma-values", default="0.05,0.10,0.15")
    parser.add_argument("--calibration-sigmas", default="0.025,0.05,0.075,0.10,0.125,0.15,0.20")
    parser.add_argument("--use-matched-gaussian", action=argparse.BooleanOptionalAction, default=Phase2BConfig.use_matched_gaussian)
    parser.add_argument("--lambda-cos", type=float, default=Phase2BConfig.lambda_cos)
    parser.add_argument("--lambda-var", type=float, default=Phase2BConfig.lambda_var)
    parser.add_argument("--perturbation-std", type=float, default=Phase2BConfig.perturbation_std)
    parser.add_argument("--ablation-fraction", type=float, default=Phase2BConfig.ablation_fraction)
    parser.add_argument("--enable-generation-bridge", action=argparse.BooleanOptionalAction, default=Phase2BConfig.enable_generation_bridge)
    parser.add_argument("--generation-eval-batches", type=int, default=Phase2BConfig.generation_eval_batches)
    parser.add_argument("--generation-sample-count", type=int, default=Phase2BConfig.generation_sample_count)
    parser.add_argument("--generation-max-length", type=int, default=Phase2BConfig.generation_max_length)
    parser.add_argument("--enable-token-head-bridge", action=argparse.BooleanOptionalAction, default=Phase2BConfig.enable_token_head_bridge)
    parser.add_argument("--token-head-epochs", type=int, default=Phase2BConfig.token_head_epochs)
    parser.add_argument("--token-head-learning-rate", type=float, default=Phase2BConfig.token_head_learning_rate)
    parser.add_argument("--token-head-train-batches", type=int, default=Phase2BConfig.token_head_train_batches)
    parser.add_argument("--token-head-eval-batches", type=int, default=Phase2BConfig.token_head_eval_batches)
    parser.add_argument("--strict-gate-min-evidence", type=int, default=Phase2BConfig.strict_gate_min_evidence)
    args = parser.parse_args()
    stage_tokens = parse_stage_tokens(args.stage_tokens)
    sigma_values = parse_float_tuple(args.sigma_values)
    if len(sigma_values) != len(stage_tokens):
        raise ValueError("sigma-values must have the same count as stage-tokens.")
    return Phase2BConfig(
        model_name=args.model_name,
        max_samples=args.max_samples,
        max_length=args.max_length,
        encode_batch_size=args.encode_batch_size,
        train_batch_size=args.train_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        output_dir=args.output_dir,
        input_text_file=args.input_text_file,
        use_hf_dataset=args.use_hf_dataset,
        hf_dataset_name=args.hf_dataset_name,
        hf_dataset_config=args.hf_dataset_config,
        hf_dataset_split=args.hf_dataset_split,
        seed=args.seed,
        stage_tokens=stage_tokens,
        forward_conditions=parse_forward_conditions(args.forward_conditions),
        sigma_values=sigma_values,
        calibration_sigmas=parse_float_tuple(args.calibration_sigmas),
        use_matched_gaussian=args.use_matched_gaussian,
        lambda_cos=args.lambda_cos,
        lambda_var=args.lambda_var,
        perturbation_std=args.perturbation_std,
        ablation_fraction=args.ablation_fraction,
        enable_generation_bridge=args.enable_generation_bridge,
        generation_eval_batches=args.generation_eval_batches,
        generation_sample_count=args.generation_sample_count,
        generation_max_length=args.generation_max_length,
        enable_token_head_bridge=args.enable_token_head_bridge,
        token_head_epochs=args.token_head_epochs,
        token_head_learning_rate=args.token_head_learning_rate,
        token_head_train_batches=args.token_head_train_batches,
        token_head_eval_batches=args.token_head_eval_batches,
        strict_gate_min_evidence=args.strict_gate_min_evidence,
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
            if isinstance(row, dict):
                values = [value.strip() for value in row.values() if isinstance(value, str)]
            else:
                values = [str(row).strip()]
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


def extract_texts_from_rows(rows, max_samples: int) -> list[str]:
    texts: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            raw_text = row.get("text")
            if raw_text is None:
                values = [value for value in row.values() if isinstance(value, str)]
                raw_text = max(values, key=len) if values else ""
        else:
            raw_text = str(row)
        text = str(raw_text).strip()
        if len(text) < 20:
            continue
        if text.startswith("=") and text.endswith("="):
            continue
        texts.append(text)
        if len(texts) >= max_samples:
            break
    return texts


def load_hf_dataset_texts(config: Phase2Config) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    dataset = load_dataset(config.hf_dataset_name, config.hf_dataset_config, split=config.hf_dataset_split)
    texts = extract_texts_from_rows(dataset, config.max_samples)
    if texts:
        print(
            f"Loaded {len(texts)} texts from Hugging Face dataset "
            f"{config.hf_dataset_name}/{config.hf_dataset_config}:{config.hf_dataset_split}."
        )
    return texts


def load_texts(config: Phase2Config) -> tuple[list[str], str]:
    source = "kaggle_input"
    if config.input_text_file:
        texts = read_text_file(Path(config.input_text_file), config.max_samples)
        source = config.input_text_file
    else:
        texts = discover_kaggle_texts(config.max_samples)

    if not texts and config.use_hf_dataset:
        try:
            texts = load_hf_dataset_texts(config)
            source = f"hf:{config.hf_dataset_name}/{config.hf_dataset_config}:{config.hf_dataset_split}"
        except Exception as exc:
            print(f"Could not load Hugging Face dataset, falling back to local corpus: {exc}")
            texts = []

    if not texts:
        repeats = (config.max_samples // len(FALLBACK_TEXTS)) + 1
        texts = (FALLBACK_TEXTS * repeats)[: config.max_samples]
        source = "fallback"
        print(f"Using fallback text corpus with {len(texts)} samples.")

    return texts[: config.max_samples], source


def require_runtime_dependencies() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Run this kernel with a Kaggle GPU image or the project .venv.") from exc

    missing = []
    for package_name in ("transformers", "datasets", "sentencepiece", "tqdm"):
        try:
            __import__(package_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def adaptive_pool_tokens(hidden_states, target_tokens: int):
    import torch.nn.functional as functional

    channels_first = hidden_states.float().transpose(1, 2)
    pooled = functional.adaptive_avg_pool1d(channels_first, target_tokens)
    return pooled.transpose(1, 2).contiguous()


def interpolate_tokens(latents, target_tokens: int):
    import torch.nn.functional as functional

    channels_first = latents.float().transpose(1, 2)
    expanded = functional.interpolate(channels_first, size=target_tokens, mode="linear", align_corners=False)
    return expanded.transpose(1, 2).contiguous()


def select_tokens(hidden_states, indices: list[int]):
    import torch

    index_tensor = torch.tensor(indices, dtype=torch.long, device=hidden_states.device)
    return hidden_states.index_select(dim=1, index=index_tensor).float().contiguous()


def random_select_tokens(hidden_states, target_tokens: int, seed: int):
    import torch

    generator = torch.Generator(device=hidden_states.device).manual_seed(seed)
    selected_batches = []
    for batch_index in range(hidden_states.shape[0]):
        indices = torch.randperm(hidden_states.shape[1], generator=generator, device=hidden_states.device)[:target_tokens]
        indices, _ = torch.sort(indices)
        selected_batches.append(hidden_states[batch_index].index_select(0, indices))
    return torch.stack(selected_batches, dim=0).float().contiguous()


def gaussian_noise_tokens(hidden_states, sigma: float, seed: int):
    import torch

    generator = torch.Generator(device=hidden_states.device).manual_seed(seed)
    noise = torch.randn(hidden_states.shape, generator=generator, device=hidden_states.device, dtype=hidden_states.float().dtype)
    return hidden_states.float() + (noise * sigma)


def encode_texts(config: Phase2Config, texts: list[str]):
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    encoder = T5EncoderModel.from_pretrained(config.model_name)
    encoder.to(device)
    encoder.eval()

    hidden_batches = []
    mask_batches = []
    input_id_batches = []
    for start in range(0, len(texts), config.encode_batch_size):
        batch_texts = texts[start : start + config.encode_batch_size]
        batch = tokenizer(
            batch_texts,
            max_length=config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_id_batches.append(batch["input_ids"].cpu().to(torch.long))
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.no_grad():
            hidden = encoder(**batch).last_hidden_state
        hidden_batches.append(hidden.cpu().to(torch.float16))
        mask_batches.append(batch["attention_mask"].cpu().to(torch.int16))

    hidden_states = torch.cat(hidden_batches, dim=0)
    attention_mask = torch.cat(mask_batches, dim=0)
    input_ids = torch.cat(input_id_batches, dim=0)
    return hidden_states, attention_mask, input_ids, tokenizer, str(device)


def build_forward_cache(hidden_states, attention_mask, config: Phase2Config) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    del attention_mask
    import torch
    import torch.nn.functional as functional

    cache: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {"conditions": {}, "calibration": {}, "effective_stage_sigmas": {}}
    max_length = int(hidden_states.shape[1])
    hidden = hidden_states.float()

    average_initial_losses: dict[str, float] = {}
    sigma_initial_losses: dict[str, float] = {}
    names = stage_names(config.stage_tokens)

    for stage_name, target_tokens in zip(names, config.stage_tokens):
        pooled = adaptive_pool_tokens(hidden, target_tokens)
        upsampled = interpolate_tokens(pooled, max_length)
        average_initial_losses[stage_name] = float(functional.mse_loss(upsampled.float(), hidden.float()).item())

    for sigma in config.calibration_sigmas:
        sigma_name = f"sigma_{sigma:g}"
        noisy = gaussian_noise_tokens(hidden, sigma, config.seed + int(sigma * 1000))
        sigma_initial_losses[sigma_name] = float(functional.mse_loss(noisy.float(), hidden.float()).item())

    metadata["calibration"] = {
        "average_pool_initial_losses": average_initial_losses,
        "sigma_initial_losses": sigma_initial_losses,
        "matched_sigmas": choose_matched_sigmas(average_initial_losses, sigma_initial_losses),
    }
    effective_stage_sigmas = resolve_stage_sigmas(metadata, names, config.sigma_values, config.use_matched_gaussian)
    metadata["effective_stage_sigmas"] = effective_stage_sigmas

    for condition in config.forward_conditions:
        cache[condition] = {}
        metadata["conditions"][condition] = {}
        for stage_index, target_tokens in enumerate(config.stage_tokens, 1):
            name = f"z{stage_index}"
            seed = config.seed + (stage_index * 101)
            if condition == "average_pool":
                latent = adaptive_pool_tokens(hidden, target_tokens)
            elif condition == "strided_select":
                latent = select_tokens(hidden, compute_strided_indices(max_length, target_tokens))
            elif condition == "random_select":
                latent = random_select_tokens(hidden, target_tokens, seed)
            elif condition == "gaussian_noise":
                sigma = effective_stage_sigmas[name]
                latent = gaussian_noise_tokens(hidden, sigma, seed)
            else:
                raise ValueError(f"Unknown condition: {condition}")

            cache[condition][name] = latent.cpu().to(torch.float16)
            metadata["conditions"][condition][name] = {
                "stage": name,
                "input_tokens": max_length,
                "output_tokens": int(latent.shape[1]),
                "compression_ratio": float(latent.shape[1] / max_length),
                "seed": seed,
            }
            if condition == "gaussian_noise":
                metadata["conditions"][condition][name]["noise_sigma"] = effective_stage_sigmas[name]
                metadata["conditions"][condition][name]["requested_noise_sigma"] = config.sigma_values[stage_index - 1]
                metadata["conditions"][condition][name]["use_matched_gaussian"] = config.use_matched_gaussian
            else:
                metadata["conditions"][condition][name]["noise_sigma"] = None
    return cache, metadata


def split_indices(sample_count: int, val_fraction: float, seed: int):
    import torch

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(sample_count, generator=generator)
    if sample_count <= 2:
        return indices, indices
    val_count = max(1, int(sample_count * val_fraction))
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    if len(train_indices) == 0:
        train_indices = val_indices
    return train_indices, val_indices


def make_loader(latents, hidden_states, attention_mask, input_ids, indices, batch_size: int, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    sample_indices = indices.to(torch.long)
    dataset = TensorDataset(
        latents[sample_indices].float(),
        hidden_states[sample_indices].float(),
        attention_mask[sample_indices].long(),
        input_ids[sample_indices].long(),
        sample_indices,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def cosine_distance_loss(predicted, target, mask=None):
    import torch.nn.functional as functional

    if mask is not None:
        active = mask.bool()
        predicted = predicted[active]
        target = target[active]
    else:
        predicted = predicted.reshape(-1, predicted.shape[-1])
        target = target.reshape(-1, target.shape[-1])
    if predicted.numel() == 0:
        return predicted.new_tensor(0.0)
    return 1.0 - functional.cosine_similarity(predicted.float(), target.float(), dim=-1).mean()


def mse_loss(predicted, target, mask=None):
    import torch.nn.functional as functional

    if mask is not None:
        active = mask.bool()
        predicted = predicted[active]
        target = target[active]
    return functional.mse_loss(predicted.float(), target.float())


def variance_penalty(predicted, target):
    import torch.nn.functional as functional

    predicted_var = predicted.float().var(dim=(0, 1), unbiased=False).mean()
    target_var = target.float().var(dim=(0, 1), unbiased=False).mean().detach().clamp_min(1e-6)
    low_penalty = functional.relu((target_var * 0.05) - predicted_var) / target_var
    high_penalty = functional.relu(predicted_var - (target_var * 5.0)) / target_var
    return low_penalty + high_penalty


def reconstruction_metrics(reference, reconstructed, attention_mask):
    import torch
    import torch.nn.functional as functional

    active_tokens = attention_mask.bool()
    reference_active = reference.float()[active_tokens]
    reconstructed_active = reconstructed.float()[active_tokens]
    mse = functional.mse_loss(reconstructed_active, reference_active).item()
    cosine = functional.cosine_similarity(reconstructed_active, reference_active, dim=-1).mean().item()
    finite = bool(torch.isfinite(reconstructed_active).all().item())
    ref_var = reference_active.var(dim=0, unbiased=False).mean().item()
    rec_var = reconstructed_active.var(dim=0, unbiased=False).mean().item()
    variance_ratio = rec_var / max(ref_var, 1e-8)
    return {"mse": mse, "cosine": cosine, "finite": finite, "variance_ratio": variance_ratio}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def parameter_count(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def create_expander(hidden_size: int, max_length: int, device: str):
    import torch
    import torch.nn as nn

    class DirectExpander(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.position = nn.Parameter(torch.zeros(1, max_length, hidden_size))
            self.refine = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
            )

        def forward(self, latents):
            expanded = interpolate_tokens(latents, max_length)
            expanded = expanded + self.position
            return expanded + self.refine(expanded)

    return DirectExpander().to(device)


def expander_loss(expander, latents, hidden_states, attention_mask, config: Phase2Config):
    reconstructed = expander(latents)
    mse = mse_loss(reconstructed, hidden_states, attention_mask)
    cos = cosine_distance_loss(reconstructed, hidden_states, attention_mask)
    var = variance_penalty(reconstructed, hidden_states)
    loss = mse + (config.lambda_cos * cos) + (config.lambda_var * var)
    return loss, {"mse": float(mse.detach().cpu()), "cosine_distance": float(cos.detach().cpu()), "variance_penalty": float(var.detach().cpu())}


def decoder_nll(decoder_model, encoder_hidden, attention_mask, input_ids):
    from transformers.modeling_outputs import BaseModelOutput

    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)
    outputs = decoder_model(
        encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden),
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )
    return outputs.loss


def create_token_head(hidden_size: int, vocab_size: int, device: str):
    import torch.nn as nn

    return nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, vocab_size)).to(device)


def token_head_nll(token_head, encoder_hidden, attention_mask, input_ids):
    import torch.nn.functional as functional

    logits = token_head(encoder_hidden.float())
    labels = input_ids.clone().masked_fill(attention_mask == 0, -100)
    return functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def train_token_head_bridge(config: Phase2Config, hidden_states, attention_mask, input_ids, tokenizer, device: str):
    if not config.enable_token_head_bridge:
        return None, {"enabled": False}

    import torch

    hidden_size = int(hidden_states.shape[-1])
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    token_head = create_token_head(hidden_size, vocab_size, device)
    train_indices, val_indices = split_indices(hidden_states.shape[0], config.val_fraction, config.seed + 17)
    train_loader = make_loader(hidden_states, hidden_states, attention_mask, input_ids, train_indices, config.train_batch_size, True, config.seed + 17)
    val_loader = make_loader(hidden_states, hidden_states, attention_mask, input_ids, val_indices, config.train_batch_size, False, config.seed + 17)
    optimizer = torch.optim.AdamW(token_head.parameters(), lr=config.token_head_learning_rate, weight_decay=config.weight_decay)

    initial_nll = evaluate_token_head_oracle(token_head, val_loader, device, config.token_head_eval_batches)
    train_losses = []
    for _epoch in range(config.token_head_epochs):
        token_head.train()
        for batch_index, _latents, batch_hidden, batch_mask, batch_input_ids, _sample_indices in iter_token_head_batches(train_loader):
            if batch_index >= config.token_head_train_batches:
                break
            batch_hidden = batch_hidden.to(device)
            batch_mask = batch_mask.to(device)
            batch_input_ids = batch_input_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = token_head_nll(token_head, batch_hidden, batch_mask, batch_input_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(token_head.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

    final_nll = evaluate_token_head_oracle(token_head, val_loader, device, config.token_head_eval_batches)
    token_head.eval()
    return token_head, {
        "enabled": True,
        "parameter_count": parameter_count(token_head),
        "train_batches": min(config.token_head_train_batches, len(train_loader)),
        "train_loss": mean(train_losses),
        "initial_oracle_token_head_nll": initial_nll,
        "final_oracle_token_head_nll": final_nll,
    }


def iter_token_head_batches(loader):
    for batch_index, batch in enumerate(loader):
        yield (batch_index, *batch)


def evaluate_token_head_oracle(token_head, loader, device: str, max_batches: int) -> float:
    values = []
    token_head.eval()
    with __import__("torch").no_grad():
        for batch_index, (_latents, hidden_states, attention_mask, input_ids, _sample_indices) in enumerate(loader):
            if batch_index >= max_batches:
                break
            values.append(
                float(
                    token_head_nll(
                        token_head,
                        hidden_states.to(device),
                        attention_mask.to(device),
                        input_ids.to(device),
                    )
                    .detach()
                    .cpu()
                )
            )
    return mean(values)


def is_meaningful_generation(text: str) -> bool:
    import re

    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    alpha_chars = sum(len(token) for token in tokens)
    if alpha_chars < 12 or len(tokens) < 3:
        return False
    unique_ratio = len({token.lower() for token in tokens}) / len(tokens)
    return unique_ratio >= 0.5


def summarize_generation_samples(samples: list[dict[str, Any]]) -> dict[str, float | int]:
    generated = [str(sample.get("generated", "")).strip() for sample in samples if "generated" in sample]
    if not generated:
        return {"sample_count": 0, "nonempty_sample_rate": 0.0, "meaningful_sample_rate": 0.0}
    nonempty = [text for text in generated if text]
    meaningful = [text for text in generated if is_meaningful_generation(text)]
    return {
        "sample_count": len(generated),
        "nonempty_sample_rate": len(nonempty) / len(generated),
        "meaningful_sample_rate": len(meaningful) / len(generated),
    }


def maybe_decode_samples(decoder_model, tokenizer, encoder_hidden, attention_mask, sample_indices, texts: list[str], config: Phase2Config):
    from transformers.modeling_outputs import BaseModelOutput

    if config.generation_sample_count <= 0:
        return []
    count = min(config.generation_sample_count, encoder_hidden.shape[0])
    try:
        generated_ids = decoder_model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden[:count]),
            attention_mask=attention_mask[:count],
            max_length=config.generation_max_length,
            num_beams=1,
            do_sample=False,
        )
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    except Exception as exc:
        return [{"error": f"generation_failed: {exc}"}]

    samples = []
    for local_index, generated_text in enumerate(generated_texts):
        original_index = int(sample_indices[local_index].detach().cpu().item())
        samples.append({"sample_index": original_index, "original": texts[original_index], "generated": generated_text})
    return samples


def evaluate_stage(
    expander,
    loader,
    config: Phase2Config,
    device: str,
    decoder_model=None,
    token_head_model=None,
    tokenizer=None,
    texts: list[str] | None = None,
    collect_samples: bool = False,
):
    import torch

    expander.eval()
    loss_values = []
    mse_values = []
    cosine_values = []
    variance_values = []
    finite_values = []
    relative_sensitivity_values = []
    ablation_delta_values = []
    swap_delta_values = []
    token_nll_values = []
    oracle_nll_values = []
    token_head_nll_values = []
    oracle_token_head_nll_values = []
    generation_errors = []
    samples = []

    generation_batches_seen = 0
    token_head_batches_seen = 0
    with torch.no_grad():
        for latents, hidden_states, attention_mask, input_ids, sample_indices in loader:
            latents = latents.to(device)
            hidden_states = hidden_states.to(device)
            attention_mask = attention_mask.to(device)
            input_ids = input_ids.to(device)
            reconstructed = expander(latents)
            loss, _details = expander_loss(expander, latents, hidden_states, attention_mask, config)
            metrics = reconstruction_metrics(hidden_states, reconstructed, attention_mask)
            base_mse = metrics["mse"]
            loss_values.append(float(loss.detach().cpu()))
            mse_values.append(metrics["mse"])
            cosine_values.append(metrics["cosine"])
            variance_values.append(metrics["variance_ratio"])
            finite_values.append(metrics["finite"])

            noisy_latents = latents + (torch.randn_like(latents) * config.perturbation_std)
            noisy_reconstruction = expander(noisy_latents)
            perturbation_mse = mse_loss(noisy_reconstruction, reconstructed, attention_mask)
            relative_sensitivity_values.append(float(perturbation_mse.detach().cpu()) / max(base_mse, 1e-8))

            ablated = latents.clone()
            token_count = max(1, int(ablated.shape[1] * config.ablation_fraction))
            ablated[:, :token_count, :] = 0.0
            ablated_reconstruction = expander(ablated)
            ablated_mse = mse_loss(ablated_reconstruction, hidden_states, attention_mask)
            ablation_delta_values.append(float(ablated_mse.detach().cpu()) - base_mse)

            swapped = torch.flip(latents, dims=(0,))
            swapped_reconstruction = expander(swapped)
            swapped_mse = mse_loss(swapped_reconstruction, hidden_states, attention_mask)
            swap_delta_values.append(float(swapped_mse.detach().cpu()) - base_mse)

            if decoder_model is not None and generation_batches_seen < config.generation_eval_batches:
                try:
                    token_nll = decoder_nll(decoder_model, reconstructed, attention_mask, input_ids)
                    oracle_nll = decoder_nll(decoder_model, hidden_states, attention_mask, input_ids)
                    token_nll_values.append(float(token_nll.detach().cpu()))
                    oracle_nll_values.append(float(oracle_nll.detach().cpu()))
                    if collect_samples and tokenizer is not None and texts is not None and not samples:
                        samples = maybe_decode_samples(decoder_model, tokenizer, reconstructed, attention_mask, sample_indices, texts, config)
                except Exception as exc:
                    generation_errors.append(str(exc))
                generation_batches_seen += 1

            if token_head_model is not None and token_head_batches_seen < config.token_head_eval_batches:
                try:
                    head_nll = token_head_nll(token_head_model, reconstructed, attention_mask, input_ids)
                    oracle_head_nll = token_head_nll(token_head_model, hidden_states, attention_mask, input_ids)
                    token_head_nll_values.append(float(head_nll.detach().cpu()))
                    oracle_token_head_nll_values.append(float(oracle_head_nll.detach().cpu()))
                except Exception as exc:
                    generation_errors.append(f"token_head: {exc}")
                token_head_batches_seen += 1

    token_nll = mean(token_nll_values)
    oracle_nll = mean(oracle_nll_values)
    head_nll = mean(token_head_nll_values)
    oracle_head_nll = mean(oracle_token_head_nll_values)
    return {
        "loss": mean(loss_values),
        "mse": mean(mse_values),
        "cosine": mean(cosine_values),
        "variance_ratio": mean(variance_values),
        "finite": all(finite_values),
        "latent_use": {
            "relative_perturbation_sensitivity": mean(relative_sensitivity_values),
            "ablation_delta_mse": mean(ablation_delta_values),
            "swap_delta_mse": mean(swap_delta_values),
        },
        "generation_bridge": {
            "token_nll": token_nll,
            "oracle_token_nll": oracle_nll,
            "delta_token_nll_vs_h0": token_nll - oracle_nll if token_nll_values and oracle_nll_values else None,
            "token_head_nll": head_nll,
            "oracle_token_head_nll": oracle_head_nll,
            "delta_token_head_nll_vs_h0": head_nll - oracle_head_nll if token_head_nll_values and oracle_token_head_nll_values else None,
            "evaluated_batches": generation_batches_seen,
            "token_head_evaluated_batches": token_head_batches_seen,
            "errors": generation_errors[:3],
        },
        "generation_quality": summarize_generation_samples(samples),
        "samples": samples,
    }


def train_condition_stage(
    condition: str,
    stage_name: str,
    latents,
    hidden_states,
    attention_mask,
    input_ids,
    config: Phase2Config,
    output_dir: Path,
    decoder_model,
    token_head_model,
    tokenizer,
    texts: list[str],
):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = int(hidden_states.shape[-1])
    train_indices, val_indices = split_indices(hidden_states.shape[0], config.val_fraction, config.seed)
    train_loader = make_loader(latents, hidden_states, attention_mask, input_ids, train_indices, config.train_batch_size, True, config.seed)
    val_loader = make_loader(latents, hidden_states, attention_mask, input_ids, val_indices, config.train_batch_size, False, config.seed)
    expander = create_expander(hidden_size, config.max_length, device)
    optimizer = torch.optim.AdamW(expander.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_losses = []
    validation_losses = []

    initial_validation = evaluate_stage(expander, val_loader, config, device)
    validation_losses.append(initial_validation["loss"])
    log_path = output_dir / "train_log.jsonl"

    for epoch in range(1, config.epochs + 1):
        expander.train()
        epoch_losses = []
        for batch_latents, batch_hidden, batch_mask, _batch_input_ids, _sample_indices in train_loader:
            batch_latents = batch_latents.to(device)
            batch_hidden = batch_hidden.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _details = expander_loss(expander, batch_latents, batch_hidden, batch_mask, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expander.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        train_loss = mean(epoch_losses)
        validation = evaluate_stage(expander, val_loader, config, device)
        train_losses.append(train_loss)
        validation_losses.append(validation["loss"])
        log_record = {
            "condition": condition,
            "stage": stage_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "validation_mse": validation["mse"],
            "validation_cosine": validation["cosine"],
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_record) + "\n")
        print(json.dumps(log_record, indent=2))

    final_validation = evaluate_stage(
        expander,
        val_loader,
        config,
        device,
        decoder_model=decoder_model,
        token_head_model=token_head_model,
        tokenizer=tokenizer,
        texts=texts,
        collect_samples=True,
    )
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "expander": expander.state_dict(),
            "config": asdict(config),
            "condition": condition,
            "stage": stage_name,
        },
        checkpoints_dir / f"{condition}_{stage_name}.pt",
    )
    return {
        "condition": condition,
        "stage": stage_name,
        "device": device,
        "parameter_count": parameter_count(expander),
        "input_shape": list(latents.shape),
        "train": {"initial_loss": train_losses[0] if train_losses else 0.0, "final_loss": train_losses[-1] if train_losses else 0.0},
        "validation": {"initial_loss": validation_losses[0], "final_loss": validation_losses[-1]},
        "final": final_validation,
    }


def prepare_generation_bridge(config: Phase2Config, device: str):
    if not config.enable_generation_bridge:
        return None
    try:
        from transformers import T5ForConditionalGeneration

        model = T5ForConditionalGeneration.from_pretrained(config.model_name)
        model.to(device)
        model.eval()
        return model
    except Exception as exc:
        print(f"Generation bridge disabled after decoder load failure: {exc}")
        return None


def evaluate_h0_controls(
    config: Phase2Config,
    hidden_states,
    attention_mask,
    input_ids,
    tokenizer,
    texts: list[str],
    decoder_model,
    token_head_model,
    device: str,
) -> dict[str, Any]:
    import torch

    _train_indices, val_indices = split_indices(hidden_states.shape[0], config.val_fraction, config.seed)
    val_loader = make_loader(hidden_states, hidden_states, attention_mask, input_ids, val_indices, config.train_batch_size, False, config.seed)
    decoder_nll_values = []
    token_head_values = []
    samples = []
    errors = []
    decoder_batches = 0
    token_head_batches = 0
    with torch.no_grad():
        for _latents, batch_hidden, batch_mask, batch_input_ids, sample_indices in val_loader:
            batch_hidden = batch_hidden.to(device)
            batch_mask = batch_mask.to(device)
            batch_input_ids = batch_input_ids.to(device)
            if decoder_model is not None and decoder_batches < config.generation_eval_batches:
                try:
                    decoder_nll_values.append(float(decoder_nll(decoder_model, batch_hidden, batch_mask, batch_input_ids).detach().cpu()))
                    if not samples:
                        samples = maybe_decode_samples(decoder_model, tokenizer, batch_hidden, batch_mask, sample_indices, texts, config)
                except Exception as exc:
                    errors.append(str(exc))
                decoder_batches += 1
            if token_head_model is not None and token_head_batches < config.token_head_eval_batches:
                try:
                    token_head_values.append(float(token_head_nll(token_head_model, batch_hidden, batch_mask, batch_input_ids).detach().cpu()))
                except Exception as exc:
                    errors.append(f"token_head: {exc}")
                token_head_batches += 1
            if decoder_batches >= config.generation_eval_batches and token_head_batches >= config.token_head_eval_batches:
                break

    return {
        "oracle_token_nll": mean(decoder_nll_values),
        "oracle_token_head_nll": mean(token_head_values),
        "decoder_evaluated_batches": decoder_batches,
        "token_head_evaluated_batches": token_head_batches,
        "generation_quality": summarize_generation_samples(samples),
        "samples": samples,
        "errors": errors[:3],
    }


def save_cache(output_dir: Path, texts: list[str], text_source: str, hidden_states, attention_mask, input_ids, config: Phase2Config) -> tuple[Path, bool]:
    import torch

    cache_path = output_dir / "latent_cache.pt"
    payload = {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "input_ids": input_ids,
        "texts": texts,
        "text_source": text_source,
        "config": asdict(config),
    }
    torch.save(payload, cache_path)
    reloaded = torch.load(cache_path, map_location="cpu")
    cache_allclose = bool(torch.equal(reloaded["hidden_states"], hidden_states))
    return cache_path, cache_allclose


def average_stage_metric(condition_results: dict[str, Any], metric_path: tuple[str, ...]) -> float:
    values = []
    for stage_result in condition_results.values():
        current: Any = stage_result
        for key in metric_path:
            if current is None:
                break
            current = current.get(key) if isinstance(current, dict) else None
        if isinstance(current, (int, float)):
            values.append(float(current))
    return mean(values)


def monotonic_increasing(values: list[float]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def monotonic_decreasing(values: list[float]) -> bool:
    return all(left > right for left, right in zip(values, values[1:]))


def evaluate_phase2b_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    results = metrics.get("results", {})
    required = {"average_pool", "random_select", "gaussian_noise"}
    p2b_run = required.issubset(set(results))

    average_pool = results.get("average_pool", {})
    random_select = results.get("random_select", {})
    gaussian_noise = results.get("gaussian_noise", {})

    avg_pool_mse = average_stage_metric(average_pool, ("final", "mse"))
    random_mse = average_stage_metric(random_select, ("final", "mse"))
    gaussian_mse = average_stage_metric(gaussian_noise, ("final", "mse"))
    p2b_mse = bool(average_pool and random_select and gaussian_noise and avg_pool_mse < random_mse and avg_pool_mse < gaussian_mse)

    avg_pool_cos = average_stage_metric(average_pool, ("final", "cosine"))
    random_cos = average_stage_metric(random_select, ("final", "cosine"))
    gaussian_cos = average_stage_metric(gaussian_noise, ("final", "cosine"))
    p2b_cos = bool(average_pool and random_select and gaussian_noise and avg_pool_cos > random_cos and avg_pool_cos > gaussian_cos)

    avg_delta_nll = average_stage_metric(average_pool, ("final", "generation_bridge", "delta_token_nll_vs_h0"))
    random_delta_nll = average_stage_metric(random_select, ("final", "generation_bridge", "delta_token_nll_vs_h0"))
    gaussian_delta_nll = average_stage_metric(gaussian_noise, ("final", "generation_bridge", "delta_token_nll_vs_h0"))
    p2b_decoder_nll = bool(average_pool and random_select and gaussian_noise and avg_delta_nll < random_delta_nll and avg_delta_nll < gaussian_delta_nll)

    avg_head_delta = average_stage_metric(average_pool, ("final", "generation_bridge", "delta_token_head_nll_vs_h0"))
    random_head_delta = average_stage_metric(random_select, ("final", "generation_bridge", "delta_token_head_nll_vs_h0"))
    gaussian_head_delta = average_stage_metric(gaussian_noise, ("final", "generation_bridge", "delta_token_head_nll_vs_h0"))
    p2b_token_head = bool(average_pool and random_select and gaussian_noise and avg_head_delta < random_head_delta and avg_head_delta < gaussian_head_delta)

    avg_stage_names = sorted(average_pool)
    avg_mses = [float(average_pool[stage]["final"]["mse"]) for stage in avg_stage_names if stage in average_pool]
    avg_cosines = [float(average_pool[stage]["final"]["cosine"]) for stage in avg_stage_names if stage in average_pool]
    p2b_schedule = len(avg_mses) >= 2 and monotonic_increasing(avg_mses) and monotonic_decreasing(avg_cosines)

    avg_relative = average_stage_metric(average_pool, ("final", "latent_use", "relative_perturbation_sensitivity"))
    avg_ablation = average_stage_metric(average_pool, ("final", "latent_use", "ablation_delta_mse"))
    avg_swap = average_stage_metric(average_pool, ("final", "latent_use", "swap_delta_mse"))
    latent_use_signals = [avg_relative > 0.005, avg_ablation > 0.0005, avg_swap > 0.0005]
    p2b_latent_use = sum(1 for item in latent_use_signals if item) >= 2

    avg_quality = average_stage_metric(average_pool, ("final", "generation_quality", "meaningful_sample_rate"))
    h0_quality = float(metrics.get("h0_decoder_control", {}).get("generation_quality", {}).get("meaningful_sample_rate", 0.0))
    p2b_generation = bool(avg_quality > 0.0 and (h0_quality == 0.0 or avg_quality >= h0_quality * 0.5))

    evidence = [p2b_cos, p2b_decoder_nll, p2b_token_head, p2b_latent_use, p2b_generation]
    min_evidence = int(metrics.get("strict_gate_min_evidence", 2))
    phase3_candidate = p2b_run and sum(1 for item in evidence if item) >= min_evidence

    gates = {
        "P2B-G-RUN": {"pass": p2b_run, "reason": "required compression and corruption conditions ran on the same split"},
        "P2B-G-MSE": {"pass": p2b_mse, "reason": "average_pool beats random_select and matched gaussian_noise on hidden-state MSE"},
        "P2B-G-COS": {"pass": p2b_cos, "reason": "average_pool beats random_select and matched gaussian_noise on cosine direction"},
        "P2B-G-DECODER-NLL": {"pass": p2b_decoder_nll, "reason": "average_pool has lower frozen-decoder delta token NLL"},
        "P2B-G-TOKEN-HEAD-NLL": {"pass": p2b_token_head, "reason": "average_pool has lower lightweight token-head delta NLL"},
        "P2B-G-SCHEDULE": {"pass": p2b_schedule, "reason": "average_pool stages become harder monotonically"},
        "P2B-G-USE": {"pass": p2b_latent_use, "reason": "relative perturbation, ablation, or swap tests show latent use"},
        "P2B-G-GEN": {"pass": p2b_generation, "reason": "open-ended samples contain nontrivial generated text relative to h0 control"},
    }
    gates["overall_pass"] = all(value["pass"] for key, value in gates.items() if key.startswith("P2B-G-"))
    gates["phase3_candidate"] = phase3_candidate
    gates["diagnostics"] = {
        "avg_pool_mse": avg_pool_mse,
        "random_mse": random_mse,
        "gaussian_mse": gaussian_mse,
        "avg_pool_cosine": avg_pool_cos,
        "random_cosine": random_cos,
        "gaussian_cosine": gaussian_cos,
        "avg_delta_token_nll": avg_delta_nll,
        "random_delta_token_nll": random_delta_nll,
        "gaussian_delta_token_nll": gaussian_delta_nll,
        "avg_delta_token_head_nll": avg_head_delta,
        "random_delta_token_head_nll": random_head_delta,
        "gaussian_delta_token_head_nll": gaussian_head_delta,
        "avg_relative_sensitivity": avg_relative,
        "avg_ablation_delta": avg_ablation,
        "avg_swap_delta": avg_swap,
        "avg_meaningful_sample_rate": avg_quality,
        "h0_meaningful_sample_rate": h0_quality,
        "evidence_count": sum(1 for item in evidence if item),
        "min_evidence": min_evidence,
    }
    return gates


evaluate_phase2_gates = evaluate_phase2b_gates


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# LACE Phase 2B Summary",
        "",
        f"- Model: `{metrics['model_name']}`",
        f"- Samples: `{metrics['sample_count']}`",
        f"- Text source: `{metrics['text_source']}`",
        f"- Hidden shape: `{metrics['hidden_shape']}`",
        f"- Cache reload exact match: `{metrics['cache_allclose']}`",
        f"- Generation bridge enabled: `{metrics['generation_bridge_enabled']}`",
        f"- Token-head bridge enabled: `{metrics['token_head_bridge']['enabled']}`",
        f"- Phase 3 candidate: `{metrics['gates']['phase3_candidate']}`",
        "",
        "## Gate Summary",
        "",
        "| Gate | Pass | Reason |",
        "|---|---|---|",
    ]
    for gate_name, gate in metrics["gates"].items():
        if gate_name in {"overall_pass", "phase3_candidate", "diagnostics"}:
            continue
        lines.append(f"| {gate_name} | `{gate['pass']}` | {gate['reason']} |")
    lines.extend(
        [
            "",
            f"- Overall strict pass: `{metrics['gates']['overall_pass']}`",
            f"- Evidence count: `{metrics['gates']['diagnostics']['evidence_count']}` / `{metrics['gates']['diagnostics']['min_evidence']}` required",
            "",
            "## h0 Decoder Control",
            "",
            "```json",
            json.dumps(metrics["h0_decoder_control"], indent=2),
            "```",
            "",
            "## Condition Results",
            "",
        ]
    )
    for condition, condition_results in metrics["results"].items():
        lines.extend(
            [
                f"### {condition}",
                "",
                "| Stage | Val loss | MSE | Cosine | Var ratio | Rel sens | Abl delta | Swap delta | Decoder dNLL | Head dNLL | Meaningful gen |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for stage_name, stage_result in condition_results.items():
            final = stage_result["final"]
            latent_use = final["latent_use"]
            bridge = final["generation_bridge"]
            delta_nll = bridge["delta_token_nll_vs_h0"]
            head_delta_nll = bridge["delta_token_head_nll_vs_h0"]
            delta_text = f"{delta_nll:.6f}" if isinstance(delta_nll, (int, float)) else "n/a"
            head_delta_text = f"{head_delta_nll:.6f}" if isinstance(head_delta_nll, (int, float)) else "n/a"
            meaningful_rate = final["generation_quality"]["meaningful_sample_rate"]
            lines.append(
                f"| {stage_name} | {stage_result['validation']['final_loss']:.6f} | {final['mse']:.6f} | "
                f"{final['cosine']:.6f} | {final['variance_ratio']:.6f} | "
                f"{latent_use['relative_perturbation_sensitivity']:.6f} | "
                f"{latent_use['ablation_delta_mse']:.6f} | {latent_use['swap_delta_mse']:.6f} | "
                f"{delta_text} | {head_delta_text} | {meaningful_rate:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Gaussian Calibration",
            "",
            "```json",
            json.dumps(
                {
                    "calibration": metrics["forward_metadata"]["calibration"],
                    "effective_stage_sigmas": metrics["forward_metadata"]["effective_stage_sigmas"],
                },
                indent=2,
            ),
            "```",
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase2b(config: Phase2Config) -> dict[str, Any]:
    require_runtime_dependencies()
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = output_dir / "train_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()

    texts, text_source = load_texts(config)
    hidden_states, attention_mask, input_ids, tokenizer, encoder_device = encode_texts(config, texts)
    cache_path, cache_allclose = save_cache(output_dir, texts, text_source, hidden_states, attention_mask, input_ids, config)
    forward_cache, forward_metadata = build_forward_cache(hidden_states, attention_mask, config)

    device = "cuda" if encoder_device == "cuda" else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    decoder_model = prepare_generation_bridge(config, device)
    generation_bridge_enabled = decoder_model is not None
    token_head_model, token_head_metrics = train_token_head_bridge(config, hidden_states, attention_mask, input_ids, tokenizer, device)
    h0_decoder_control = evaluate_h0_controls(
        config,
        hidden_states,
        attention_mask,
        input_ids,
        tokenizer,
        texts,
        decoder_model,
        token_head_model,
        device,
    )

    results: dict[str, dict[str, Any]] = {}
    sample_rows: list[dict[str, Any]] = []
    for condition in config.forward_conditions:
        results[condition] = {}
        for stage_name in stage_names(config.stage_tokens):
            print(f"Training condition={condition}, stage={stage_name}")
            stage_result = train_condition_stage(
                condition,
                stage_name,
                forward_cache[condition][stage_name],
                hidden_states,
                attention_mask,
                input_ids,
                config,
                output_dir,
                decoder_model,
                token_head_model,
                tokenizer,
                texts,
            )
            results[condition][stage_name] = stage_result
            for sample in stage_result["final"].get("samples", []):
                sample_rows.append({"condition": condition, "stage": stage_name, **sample})

    metrics = {
        "phase": "phase2b",
        "model_name": config.model_name,
        "encoder_device": encoder_device,
        "cuda_available": encoder_device == "cuda",
        "sample_count": len(texts),
        "text_source": text_source,
        "max_length": config.max_length,
        "stage_tokens": list(config.stage_tokens),
        "forward_conditions": list(config.forward_conditions),
        "sigma_values": list(config.sigma_values),
        "hidden_shape": list(hidden_states.shape),
        "hidden_dtype": str(hidden_states.dtype),
        "active_tokens": int(attention_mask.sum().item()),
        "cache_path": str(cache_path),
        "cache_allclose": cache_allclose,
        "generation_bridge_enabled": generation_bridge_enabled,
        "token_head_bridge": token_head_metrics,
        "h0_decoder_control": h0_decoder_control,
        "strict_gate_min_evidence": config.strict_gate_min_evidence,
        "forward_metadata": forward_metadata,
        "results": results,
    }
    metrics["gates"] = evaluate_phase2b_gates(metrics)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_summary(output_dir, metrics)
    write_jsonl(output_dir / "generation_samples.jsonl", sample_rows)
    return metrics


run_phase2 = run_phase2b


def main() -> None:
    config = parse_args()
    metrics = run_phase2b(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
