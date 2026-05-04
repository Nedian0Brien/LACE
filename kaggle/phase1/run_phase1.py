"""LACE Phase 1 Kaggle runner.

This script is self-contained so Kaggle can run it as a script kernel. Phase 1
checks whether a small compression path and reverse expander can learn a stable
latent reconstruction loop:

    text -> frozen T5 encoder -> h0 -> z1/z2/z3 -> reverse expansion -> h0_hat
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
    "The first trainable experiment should show whether compressed encoder states can reconstruct richer states.",
    "A small latent experiment should expose shape errors, cache failures, collapse, and unstable metrics early.",
    "Latent adaptive compression and expansion separates the research question from large model training.",
    "The model should first be evaluated as a representation trajectory before it becomes a text generator.",
    "Gaussian noise, random masking, average pooling, and learned compression define different forward paths.",
    "The Phase 1 run is deliberately small enough to execute on Kaggle or Colab resources.",
    "If reverse expansion is stable, later experiments can compare Gaussian noise, masking, and LACE-small.",
    "Information-rate schedules can be approximated with latent token budgets before precise mutual information estimation.",
    "The immediate goal is not SOTA performance, but a clean experimental loop that can scale later.",
    "A compressed latent should not be a shortcut that perfectly reconstructs every surface token.",
    "A compressed latent should also not collapse into a constant vector that removes semantic structure.",
    "Stage-wise reverse expansion creates a measurable path from compact meaning to detailed token-level latents.",
    "Perturbation checks help confirm that the expander uses latent information rather than only learned priors.",
]


@dataclass(frozen=True)
class Phase1Config:
    model_name: str = "t5-small"
    max_samples: int = 512
    max_length: int = 128
    encode_batch_size: int = 16
    train_batch_size: int = 16
    epochs: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    output_dir: str = "/kaggle/working/lace_phase1"
    input_text_file: str | None = None
    use_hf_dataset: bool = True
    hf_dataset_name: str = "wikitext"
    hf_dataset_config: str = "wikitext-2-raw-v1"
    hf_dataset_split: str = "train"
    seed: int = 42
    stage_tokens: tuple[int, ...] = (64, 32, 16)
    compression_modes: tuple[str, ...] = ("pooling", "attention")
    lambda_cos: float = 0.1
    lambda_var: float = 0.01
    perturbation_std: float = 0.05


def parse_stage_tokens(value: str) -> tuple[int, ...]:
    tokens = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not tokens:
        raise ValueError("At least one stage token count is required.")
    if any(token <= 0 for token in tokens):
        raise ValueError("Stage token counts must be positive.")
    if any(left <= right for left, right in zip(tokens, tokens[1:])):
        raise ValueError("Stage token counts must be strictly decreasing.")
    return tokens


def parse_compression_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    valid = {"pooling", "attention"}
    invalid = [mode for mode in modes if mode not in valid]
    if invalid:
        raise ValueError(f"Invalid compression modes: {invalid}")
    return modes or ("pooling",)


def build_stage_pairs(stage_tokens: tuple[int, ...], max_length: int) -> list[tuple[str, int, int]]:
    lengths = (max_length, *stage_tokens)
    pairs: list[tuple[str, int, int]] = []
    for stage_index in range(len(stage_tokens), 0, -1):
        source_name = f"z{stage_index}"
        target_name = "h0" if stage_index == 1 else f"z{stage_index - 1}"
        pairs.append((f"{source_name}_to_{target_name}", lengths[stage_index], lengths[stage_index - 1]))
    return pairs


def parse_args() -> Phase1Config:
    parser = argparse.ArgumentParser(description="Run LACE Phase 1 latent compression proof.")
    parser.add_argument("--model-name", default=Phase1Config.model_name)
    parser.add_argument("--max-samples", type=int, default=Phase1Config.max_samples)
    parser.add_argument("--max-length", type=int, default=Phase1Config.max_length)
    parser.add_argument("--encode-batch-size", type=int, default=Phase1Config.encode_batch_size)
    parser.add_argument("--train-batch-size", type=int, default=Phase1Config.train_batch_size)
    parser.add_argument("--epochs", type=int, default=Phase1Config.epochs)
    parser.add_argument("--learning-rate", type=float, default=Phase1Config.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=Phase1Config.weight_decay)
    parser.add_argument("--val-fraction", type=float, default=Phase1Config.val_fraction)
    parser.add_argument("--output-dir", default=os.environ.get("LACE_OUTPUT_DIR", Phase1Config.output_dir))
    parser.add_argument("--input-text-file", default=os.environ.get("LACE_INPUT_TEXT_FILE"))
    parser.add_argument("--use-hf-dataset", action=argparse.BooleanOptionalAction, default=Phase1Config.use_hf_dataset)
    parser.add_argument("--hf-dataset-name", default=Phase1Config.hf_dataset_name)
    parser.add_argument("--hf-dataset-config", default=Phase1Config.hf_dataset_config)
    parser.add_argument("--hf-dataset-split", default=Phase1Config.hf_dataset_split)
    parser.add_argument("--seed", type=int, default=Phase1Config.seed)
    parser.add_argument("--stage-tokens", default="64,32,16")
    parser.add_argument("--compression-modes", default="pooling,attention")
    parser.add_argument("--lambda-cos", type=float, default=Phase1Config.lambda_cos)
    parser.add_argument("--lambda-var", type=float, default=Phase1Config.lambda_var)
    parser.add_argument("--perturbation-std", type=float, default=Phase1Config.perturbation_std)
    args = parser.parse_args()
    return Phase1Config(
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
        stage_tokens=parse_stage_tokens(args.stage_tokens),
        compression_modes=parse_compression_modes(args.compression_modes),
        lambda_cos=args.lambda_cos,
        lambda_var=args.lambda_var,
        perturbation_std=args.perturbation_std,
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


def load_hf_dataset_texts(config: Phase1Config) -> list[str]:
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


def load_texts(config: Phase1Config) -> tuple[list[str], str]:
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
        raise RuntimeError("PyTorch is required. Run this kernel with a Kaggle GPU image.") from exc

    try:
        import transformers  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "sentencepiece"])

    try:
        import datasets  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])


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


def encode_texts(config: Phase1Config, texts: list[str]):
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    encoder = T5EncoderModel.from_pretrained(config.model_name)
    encoder.to(device)
    encoder.eval()

    hidden_batches = []
    mask_batches = []
    for start in range(0, len(texts), config.encode_batch_size):
        batch_texts = texts[start : start + config.encode_batch_size]
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


def create_modules(mode: str, hidden_size: int, stage_tokens: tuple[int, ...], max_length: int, device: str):
    import torch
    import torch.nn as nn

    class PoolingCompressor(nn.Module):
        def forward(self, hidden_states, attention_mask=None):
            return {f"z{index}": adaptive_pool_tokens(hidden_states, tokens) for index, tokens in enumerate(stage_tokens, 1)}

    class AttentionCompressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.queries = nn.ParameterDict(
                {f"z{index}": nn.Parameter(torch.randn(tokens, hidden_size) * 0.02) for index, tokens in enumerate(stage_tokens, 1)}
            )
            self.key_norm = nn.LayerNorm(hidden_size)
            self.out_norm = nn.LayerNorm(hidden_size)

        def forward(self, hidden_states, attention_mask=None):
            keys = self.key_norm(hidden_states.float())
            outputs = {}
            for stage_name, query in self.queries.items():
                scores = torch.einsum("td,bnd->btn", query, keys) / math.sqrt(hidden_size)
                if attention_mask is not None:
                    mask = attention_mask.bool().unsqueeze(1)
                    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
                weights = torch.softmax(scores, dim=-1)
                compressed = torch.einsum("btn,bnd->btd", weights, hidden_states.float())
                outputs[stage_name] = self.out_norm(compressed)
            return outputs

    class TokenExpander(nn.Module):
        def __init__(self, target_tokens: int) -> None:
            super().__init__()
            self.target_tokens = target_tokens
            self.position = nn.Parameter(torch.zeros(1, target_tokens, hidden_size))
            self.refine = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
            )

        def forward(self, latents):
            expanded = interpolate_tokens(latents, self.target_tokens)
            expanded = expanded + self.position
            return expanded + self.refine(expanded)

    class ReverseExpander(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expanders = nn.ModuleDict(
                {name: TokenExpander(target_tokens) for name, _source_tokens, target_tokens in build_stage_pairs(stage_tokens, max_length)}
            )

        def forward_pair(self, pair_name: str, latents):
            return self.expanders[pair_name](latents)

        def reconstruct_h0(self, stage_name: str, latents):
            current = latents
            stage_index = int(stage_name[1:])
            for index in range(stage_index, 0, -1):
                target_name = "h0" if index == 1 else f"z{index - 1}"
                current = self.forward_pair(f"z{index}_to_{target_name}", current)
            return current

    compressor = PoolingCompressor() if mode == "pooling" else AttentionCompressor()
    expander = ReverseExpander()
    return compressor.to(device), expander.to(device)


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


def make_loader(hidden_states, attention_mask, indices, batch_size: int, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(hidden_states[indices].float(), attention_mask[indices].long())
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
    import torch
    import torch.nn.functional as functional

    predicted_var = predicted.float().var(dim=(0, 1), unbiased=False).mean()
    target_var = target.float().var(dim=(0, 1), unbiased=False).mean().detach().clamp_min(1e-6)
    return functional.relu((target_var * 0.05) - predicted_var) / target_var


def pair_loss(expander, latents: dict[str, Any], hidden_states, attention_mask, config: Phase1Config):
    total = hidden_states.new_tensor(0.0)
    details = {}
    for pair_name, _source_tokens, _target_tokens in build_stage_pairs(config.stage_tokens, config.max_length):
        source_name, target_name = pair_name.split("_to_")
        source = latents[source_name]
        target = hidden_states if target_name == "h0" else latents[target_name].detach()
        predicted = expander.forward_pair(pair_name, source)
        mask = attention_mask if target_name == "h0" else None
        mse = mse_loss(predicted, target, mask)
        cos = cosine_distance_loss(predicted, target, mask)
        var = variance_penalty(predicted, target)
        loss = mse + (config.lambda_cos * cos) + (config.lambda_var * var)
        total = total + loss
        details[pair_name] = {"mse": float(mse.detach().cpu()), "cosine_distance": float(cos.detach().cpu()), "variance_penalty": float(var.detach().cpu())}
    return total, details


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


def evaluate_model(mode: str, compressor, expander, loader, config: Phase1Config, device: str):
    import torch

    compressor.eval()
    expander.eval()
    totals: dict[str, list[float]] = {f"z{index}": [] for index in range(1, len(config.stage_tokens) + 1)}
    cosines: dict[str, list[float]] = {f"z{index}": [] for index in range(1, len(config.stage_tokens) + 1)}
    variances: dict[str, list[float]] = {f"z{index}": [] for index in range(1, len(config.stage_tokens) + 1)}
    finite_values: dict[str, list[bool]] = {f"z{index}": [] for index in range(1, len(config.stage_tokens) + 1)}
    sensitivity_values = []
    loss_values = []
    with torch.no_grad():
        for hidden_states, attention_mask in loader:
            hidden_states = hidden_states.to(device)
            attention_mask = attention_mask.to(device)
            latents = compressor(hidden_states, attention_mask)
            loss, _details = pair_loss(expander, latents, hidden_states, attention_mask, config)
            loss_values.append(float(loss.detach().cpu()))
            for stage_name, latents_for_stage in latents.items():
                reconstructed = expander.reconstruct_h0(stage_name, latents_for_stage)
                metrics = reconstruction_metrics(hidden_states, reconstructed, attention_mask)
                totals[stage_name].append(metrics["mse"])
                cosines[stage_name].append(metrics["cosine"])
                variances[stage_name].append(metrics["variance_ratio"])
                finite_values[stage_name].append(metrics["finite"])
                if stage_name == f"z{len(config.stage_tokens)}":
                    noisy = latents_for_stage + (torch.randn_like(latents_for_stage) * config.perturbation_std)
                    noisy_reconstruction = expander.reconstruct_h0(stage_name, noisy)
                    sensitivity = mse_loss(noisy_reconstruction, reconstructed, attention_mask)
                    sensitivity_values.append(float(sensitivity.detach().cpu()))

    stage_metrics = {}
    for stage_name in totals:
        stage_metrics[stage_name] = {
            "mse": mean(totals[stage_name]),
            "cosine": mean(cosines[stage_name]),
            "variance_ratio": mean(variances[stage_name]),
            "finite": all(finite_values[stage_name]),
        }
    return {
        "mode": mode,
        "loss": mean(loss_values),
        "stages": stage_metrics,
        "perturbation": {"sensitivity": mean(sensitivity_values)},
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def train_one_mode(mode: str, hidden_states, attention_mask, config: Phase1Config, output_dir: Path):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = int(hidden_states.shape[-1])
    train_indices, val_indices = split_indices(hidden_states.shape[0], config.val_fraction, config.seed)
    train_loader = make_loader(hidden_states, attention_mask, train_indices, config.train_batch_size, True, config.seed)
    val_loader = make_loader(hidden_states, attention_mask, val_indices, config.train_batch_size, False, config.seed)
    compressor, expander = create_modules(mode, hidden_size, config.stage_tokens, config.max_length, device)
    parameters = [parameter for parameter in list(compressor.parameters()) + list(expander.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)

    log_path = output_dir / "train_log.jsonl"
    train_losses = []
    validation_losses = []
    initial_validation = evaluate_model(mode, compressor, expander, val_loader, config, device)
    validation_losses.append(initial_validation["loss"])

    for epoch in range(1, config.epochs + 1):
        compressor.train()
        expander.train()
        epoch_losses = []
        for batch_hidden, batch_mask in train_loader:
            batch_hidden = batch_hidden.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            latents = compressor(batch_hidden, batch_mask)
            loss, _details = pair_loss(expander, latents, batch_hidden, batch_mask, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        train_loss = mean(epoch_losses)
        validation = evaluate_model(mode, compressor, expander, val_loader, config, device)
        train_losses.append(train_loss)
        validation_losses.append(validation["loss"])
        log_record = {"mode": mode, "epoch": epoch, "train_loss": train_loss, "validation_loss": validation["loss"], "stages": validation["stages"]}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_record) + "\n")
        print(json.dumps(log_record, indent=2))

    final_validation = evaluate_model(mode, compressor, expander, val_loader, config, device)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"compressor": compressor.state_dict(), "expander": expander.state_dict(), "config": asdict(config), "mode": mode},
        checkpoints_dir / f"{mode}.pt",
    )
    mode_metrics = {
        "mode": mode,
        "device": device,
        "train": {"initial_loss": train_losses[0] if train_losses else 0.0, "final_loss": train_losses[-1] if train_losses else 0.0},
        "validation": {"initial_loss": validation_losses[0], "final_loss": validation_losses[-1]},
        "stages": final_validation["stages"],
        "perturbation": final_validation["perturbation"],
    }
    mode_metrics["gates"] = evaluate_phase1_gates(mode_metrics)
    return mode_metrics


def evaluate_phase1_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    validation = metrics.get("validation", {})
    stages = metrics.get("stages", {})
    perturbation = metrics.get("perturbation", {})
    initial_loss = float(validation.get("initial_loss", 0.0))
    final_loss = float(validation.get("final_loss", initial_loss))
    z1 = stages.get("z1", {})
    z2 = stages.get("z2", {})
    z3 = stages.get("z3", {})
    z1_mse = float(z1.get("mse", 0.0))
    z2_mse = float(z2.get("mse", 0.0))
    z3_mse = float(z3.get("mse", 0.0))
    z1_cos = float(z1.get("cosine", 0.0))
    z2_cos = float(z2.get("cosine", 0.0))
    z3_cos = float(z3.get("cosine", 0.0))
    z3_var = float(z3.get("variance_ratio", 0.0))
    sensitivity = float(perturbation.get("sensitivity", 0.0))

    p1_g1 = initial_loss > 0 and final_loss <= (initial_loss * 0.8)
    p1_g2 = z1_mse < z2_mse < z3_mse and z1_cos > z2_cos > z3_cos
    p1_g3 = z3_mse > (z1_mse * 1.1) and z3_cos < (z1_cos - 0.05)
    p1_g4 = z3_var > 0.05 and z3_cos > 0.1 and sensitivity > 0.01
    gates = {
        "P1-G1": {"pass": p1_g1, "reason": "validation loss decreased by at least 20%"},
        "P1-G2": {"pass": p1_g2, "reason": "stage MSE increases and cosine decreases from z1 to z3"},
        "P1-G3": {"pass": p1_g3, "reason": "z3 remains a real bottleneck instead of a near-perfect shortcut"},
        "P1-G4": {"pass": p1_g4, "reason": "z3 does not collapse and perturbation changes reconstruction"},
    }
    gates["overall_pass"] = all(value["pass"] for key, value in gates.items() if key.startswith("P1-"))
    return gates


def save_cache(output_dir: Path, texts: list[str], text_source: str, hidden_states, attention_mask, config: Phase1Config) -> tuple[Path, bool]:
    import torch

    cache_path = output_dir / "latent_cache.pt"
    payload = {"hidden_states": hidden_states, "attention_mask": attention_mask, "texts": texts, "text_source": text_source, "config": asdict(config)}
    torch.save(payload, cache_path)
    reloaded = torch.load(cache_path, map_location="cpu")
    cache_allclose = bool(torch.equal(reloaded["hidden_states"], hidden_states))
    return cache_path, cache_allclose


def write_summary(output_dir: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# LACE Phase 1 Summary",
        "",
        f"- Model: `{metrics['model_name']}`",
        f"- Samples: `{metrics['sample_count']}`",
        f"- Text source: `{metrics['text_source']}`",
        f"- Hidden shape: `{metrics['hidden_shape']}`",
        f"- Cache reload exact match: `{metrics['cache_allclose']}`",
        "",
        "## Mode Results",
        "",
    ]
    for mode, mode_metrics in metrics["modes"].items():
        lines.extend(
            [
                f"### {mode}",
                "",
                f"- Train loss: `{mode_metrics['train']['initial_loss']:.6f}` -> `{mode_metrics['train']['final_loss']:.6f}`",
                f"- Validation loss: `{mode_metrics['validation']['initial_loss']:.6f}` -> `{mode_metrics['validation']['final_loss']:.6f}`",
                f"- Overall gate pass: `{mode_metrics['gates']['overall_pass']}`",
                "",
                "| Stage | MSE | Cosine | Variance ratio | Finite |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for stage_name, stage_metrics in mode_metrics["stages"].items():
            lines.append(
                f"| {stage_name} | {stage_metrics['mse']:.6f} | {stage_metrics['cosine']:.6f} | "
                f"{stage_metrics['variance_ratio']:.6f} | {stage_metrics['finite']} |"
            )
        lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_phase1(config: Phase1Config) -> dict[str, Any]:
    require_runtime_dependencies()
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = output_dir / "train_log.jsonl"
    if train_log_path.exists():
        train_log_path.unlink()

    texts, text_source = load_texts(config)
    hidden_states, attention_mask, encoder_device = encode_texts(config, texts)
    cache_path, cache_allclose = save_cache(output_dir, texts, text_source, hidden_states, attention_mask, config)
    mode_results = {}
    for mode in config.compression_modes:
        mode_results[mode] = train_one_mode(mode, hidden_states, attention_mask, config, output_dir)

    metrics = {
        "phase": "phase1",
        "model_name": config.model_name,
        "encoder_device": encoder_device,
        "cuda_available": mode_results[next(iter(mode_results))]["device"] == "cuda" if mode_results else False,
        "sample_count": len(texts),
        "text_source": text_source,
        "max_length": config.max_length,
        "stage_tokens": list(config.stage_tokens),
        "hidden_shape": list(hidden_states.shape),
        "hidden_dtype": str(hidden_states.dtype),
        "active_tokens": int(attention_mask.sum().item()),
        "cache_path": str(cache_path),
        "cache_allclose": cache_allclose,
        "modes": mode_results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_summary(output_dir, metrics)
    return metrics


def main() -> None:
    config = parse_args()
    metrics = run_phase1(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
