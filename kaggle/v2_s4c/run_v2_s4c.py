"""LACE V2 S4c span-infilling reverse decoder.

S4c keeps the S4a schedule comparison, but replaces free delta sequence
generation with marker-position infilling. The model encodes the visible
current state plus marker entries for newly unmasked positions, then predicts
one token id at each marker entry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


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

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "should",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "with",
    "would",
}

WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9'-]*|\d+(?:\.\d+)?")
SCHEDULES = ("importance_schedule", "random_schedule", "position_only_schedule")
DEFAULT_RATIOS = (0.25, 0.50, 0.75, 1.00)
CONTEXT_ROLE = 0
MARKER_ROLE = 1
PAD_ROLE = 2


@dataclass(frozen=True)
class V2S4cConfig:
    model_name: str = "t5-small"
    max_train_samples: int = 768
    max_eval_samples: int = 192
    max_length: int = 128
    max_masked_positions: int = 64
    skeleton_batch_size: int = 16
    train_batch_size: int = 16
    eval_batch_size: int = 16
    reverse_epochs: int = 2
    learning_rate: float = 5e-4
    d_model: int = 512
    num_heads: int = 4
    encoder_layers: int = 2
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    output_dir: str = "/kaggle/working/lace_v2_s4c"
    input_text_file: str | None = None
    use_hf_dataset: bool = True
    hf_dataset_name: str = "wikitext"
    hf_dataset_config: str = "wikitext-2-raw-v1"
    hf_dataset_split: str = "train"
    seed: int = 42
    ratios: tuple[float, ...] = DEFAULT_RATIOS
    gate_tolerance: float = 0.02
    sample_output_count: int = 30
    min_text_words: int = 6
    schedules: tuple[str, ...] = SCHEDULES


def parse_ratios(value: str) -> tuple[float, ...]:
    ratios = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(ratios) < 2:
        raise ValueError("At least two ratios are required.")
    if any(item <= 0 or item > 1 for item in ratios):
        raise ValueError("Ratios must be in the interval (0, 1].")
    if tuple(sorted(ratios)) != ratios:
        raise ValueError("Ratios must be sorted in ascending order.")
    if ratios[-1] != 1.0:
        raise ValueError("The final ratio must be 1.0.")
    return ratios


def parse_schedules(value: str) -> tuple[str, ...]:
    schedules = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(schedules) - set(SCHEDULES))
    if unknown:
        raise ValueError(f"Unknown schedules: {unknown}")
    if not schedules:
        raise ValueError("At least one schedule is required.")
    return schedules


def parse_args() -> V2S4cConfig:
    parser = argparse.ArgumentParser(description="Run LACE V2 S4c span-infilling reverse decoder.")
    parser.add_argument("--model-name", default=V2S4cConfig.model_name)
    parser.add_argument("--max-train-samples", type=int, default=V2S4cConfig.max_train_samples)
    parser.add_argument("--max-eval-samples", type=int, default=V2S4cConfig.max_eval_samples)
    parser.add_argument("--max-length", type=int, default=V2S4cConfig.max_length)
    parser.add_argument("--max-masked-positions", type=int, default=V2S4cConfig.max_masked_positions)
    parser.add_argument("--skeleton-batch-size", type=int, default=V2S4cConfig.skeleton_batch_size)
    parser.add_argument("--train-batch-size", type=int, default=V2S4cConfig.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=V2S4cConfig.eval_batch_size)
    parser.add_argument("--reverse-epochs", type=int, default=V2S4cConfig.reverse_epochs)
    parser.add_argument("--learning-rate", type=float, default=V2S4cConfig.learning_rate)
    parser.add_argument("--d-model", type=int, default=V2S4cConfig.d_model)
    parser.add_argument("--num-heads", type=int, default=V2S4cConfig.num_heads)
    parser.add_argument("--encoder-layers", type=int, default=V2S4cConfig.encoder_layers)
    parser.add_argument("--dropout", type=float, default=V2S4cConfig.dropout)
    parser.add_argument("--max-grad-norm", type=float, default=V2S4cConfig.max_grad_norm)
    parser.add_argument("--output-dir", default=os.environ.get("LACE_OUTPUT_DIR", V2S4cConfig.output_dir))
    parser.add_argument("--input-text-file", default=os.environ.get("LACE_INPUT_TEXT_FILE"))
    parser.add_argument("--use-hf-dataset", action=argparse.BooleanOptionalAction, default=V2S4cConfig.use_hf_dataset)
    parser.add_argument("--hf-dataset-name", default=V2S4cConfig.hf_dataset_name)
    parser.add_argument("--hf-dataset-config", default=V2S4cConfig.hf_dataset_config)
    parser.add_argument("--hf-dataset-split", default=V2S4cConfig.hf_dataset_split)
    parser.add_argument("--seed", type=int, default=V2S4cConfig.seed)
    parser.add_argument("--ratios", default=",".join(str(item) for item in DEFAULT_RATIOS))
    parser.add_argument("--gate-tolerance", type=float, default=V2S4cConfig.gate_tolerance)
    parser.add_argument("--sample-output-count", type=int, default=V2S4cConfig.sample_output_count)
    parser.add_argument("--min-text-words", type=int, default=V2S4cConfig.min_text_words)
    parser.add_argument("--schedules", default=",".join(SCHEDULES))
    args = parser.parse_args()
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads.")
    if args.max_masked_positions <= 0:
        raise ValueError("--max-masked-positions must be positive.")
    return V2S4cConfig(
        model_name=args.model_name,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_length=args.max_length,
        max_masked_positions=args.max_masked_positions,
        skeleton_batch_size=args.skeleton_batch_size,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        reverse_epochs=args.reverse_epochs,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        num_heads=args.num_heads,
        encoder_layers=args.encoder_layers,
        dropout=args.dropout,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir,
        input_text_file=args.input_text_file,
        use_hf_dataset=args.use_hf_dataset,
        hf_dataset_name=args.hf_dataset_name,
        hf_dataset_config=args.hf_dataset_config,
        hf_dataset_split=args.hf_dataset_split,
        seed=args.seed,
        ratios=parse_ratios(args.ratios),
        gate_tolerance=args.gate_tolerance,
        sample_output_count=args.sample_output_count,
        min_text_words=args.min_text_words,
        schedules=parse_schedules(args.schedules),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text_line(text: str) -> str:
    return " ".join(text.strip().split())


def word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def word_tokens(text: str) -> list[str]:
    return [item.lower() for item in WORD_PATTERN.findall(text)]


def content_words(text: str) -> list[str]:
    return [item for item in word_tokens(text) if item not in STOPWORDS and len(item) > 2]


def surface_entities(text: str) -> list[str]:
    candidates = re.findall(r"\b(?:[A-Z][A-Za-z0-9'-]*|\d+(?:\.\d+)?)\b", text)
    return [item.lower() for item in candidates if item.lower() not in STOPWORDS and len(item) > 1]


def load_texts(config: V2S4cConfig) -> tuple[list[str], str]:
    total_needed = config.max_train_samples + config.max_eval_samples
    if config.input_text_file:
        path = Path(config.input_text_file)
        texts = [normalize_text_line(line) for line in path.read_text(encoding="utf-8").splitlines()]
        texts = [text for text in texts if word_count(text) >= config.min_text_words]
        if not texts:
            raise ValueError(f"No usable texts found in {path}.")
        return texts[:total_needed], f"file:{path}"

    if config.use_hf_dataset:
        try:
            from datasets import load_dataset

            dataset = load_dataset(config.hf_dataset_name, config.hf_dataset_config, split=config.hf_dataset_split)
            texts = []
            for row in dataset:
                text = normalize_text_line(str(row.get("text", "")))
                if word_count(text) >= config.min_text_words:
                    texts.append(text)
                if len(texts) >= total_needed:
                    break
            if texts:
                return texts, f"hf:{config.hf_dataset_name}/{config.hf_dataset_config}:{config.hf_dataset_split}"
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Falling back to built-in texts because dataset load failed: {exc}")

    texts = (FALLBACK_TEXTS * math.ceil(total_needed / len(FALLBACK_TEXTS)))[:total_needed]
    return texts, "fallback:built_in"


def special_token_mask(input_ids, tokenizer):
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for special_id in tokenizer.all_special_ids:
        mask |= input_ids == int(special_id)
    return mask


def active_positions(valid_row) -> list[int]:
    return [index for index, is_active in enumerate(valid_row.tolist()) if bool(is_active)]


def keep_count_for_ratio(active_count: int, ratio: float) -> int:
    if active_count <= 0:
        return 0
    return max(1, min(active_count, int(round(active_count * ratio))))


def attention_received_scores(attentions, attention_mask, valid_mask):
    scores = torch.zeros(attention_mask.shape, dtype=torch.float32, device=attention_mask.device)
    query_mask = attention_mask.bool()
    query_denominator = query_mask.sum(dim=1).clamp(min=1).to(torch.float32)
    for layer_attention in attentions:
        masked = layer_attention * query_mask[:, None, :, None].to(layer_attention.dtype)
        received = masked.sum(dim=2).mean(dim=1)
        received = received / query_denominator[:, None]
        scores += received.to(torch.float32)
    scores = scores / max(1, len(attentions))
    return scores.masked_fill(~valid_mask, float("-inf")).cpu()


def decode_ids(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def marker_token_id(tokenizer) -> int:
    token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    if token_id is None or token_id == tokenizer.unk_token_id:
        return int(tokenizer.pad_token_id)
    return int(token_id)


def select_ranked_positions(scores: list[float], positions: list[int]) -> list[int]:
    return sorted(positions, key=lambda index: (-float(scores[index]), index))


def state_from_rank(
    tokenizer,
    input_ids: list[int],
    ranked_positions: list[int],
    active_count: int,
    ratio: float,
) -> dict[str, Any]:
    keep_count = keep_count_for_ratio(active_count, ratio)
    positions = sorted(ranked_positions[:keep_count])
    token_ids = [int(input_ids[position]) for position in positions]
    return {
        "ids": token_ids,
        "positions": positions,
        "text": decode_ids(tokenizer, token_ids),
        "ratio": ratio,
    }


def build_schedule_payloads(config: V2S4cConfig, tokenizer, encoder, device: str, texts: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(texts), config.skeleton_batch_size):
            batch_texts = texts[start : start + config.skeleton_batch_size]
            batch = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_attentions=True,
                return_dict=True,
            )
            valid_mask = batch["attention_mask"].bool() & ~special_token_mask(batch["input_ids"], tokenizer)
            scores = attention_received_scores(outputs.attentions, batch["attention_mask"], valid_mask)
            input_ids = batch["input_ids"].cpu()
            for row in range(input_ids.shape[0]):
                sample_id = start + row
                rows.append(
                    {
                        "sample_id": sample_id,
                        "input_ids": [int(item) for item in input_ids[row].tolist()],
                        "valid_positions": active_positions(valid_mask[row].cpu()),
                        "attention_scores": [float(item) for item in scores[row].tolist()],
                    }
                )

    payloads: list[dict[str, Any]] = []
    for row in rows:
        sample_id = int(row["sample_id"])
        input_ids = list(row["input_ids"])
        valid_positions = list(row["valid_positions"])
        attention_rank = select_ranked_positions(row["attention_scores"], valid_positions)
        random_rank = list(valid_positions)
        rng = random.Random(config.seed + sample_id)
        rng.shuffle(random_rank)
        states: dict[str, dict[float, dict[str, Any]]] = defaultdict(dict)
        for ratio in config.ratios:
            states["importance_schedule"][ratio] = state_from_rank(
                tokenizer,
                input_ids,
                attention_rank,
                len(valid_positions),
                ratio,
            )
            states["random_schedule"][ratio] = state_from_rank(
                tokenizer,
                input_ids,
                random_rank,
                len(valid_positions),
                ratio,
            )
            attention_state = states["importance_schedule"][ratio]
            states["position_only_schedule"][ratio] = {
                "ids": [tokenizer.pad_token_id] * len(attention_state["positions"]),
                "positions": list(attention_state["positions"]),
                "text": "",
                "ratio": ratio,
            }
        payloads.append({"sample_id": sample_id, "states": {key: dict(value) for key, value in states.items()}})
    return payloads


class SequenceDataset:
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def make_loader(examples: list[dict[str, Any]], batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    return DataLoader(SequenceDataset(examples), batch_size=batch_size, shuffle=shuffle, collate_fn=lambda items: items)


def pad_list(values: list[int], length: int, pad_value: int) -> list[int]:
    return values[:length] + [pad_value] * max(0, length - len(values))


def collate_examples(batch: list[dict[str, Any]], pad_token_id: int, max_position: int) -> dict[str, Any]:
    input_length = max(1, max(len(item["input_ids"]) for item in batch))
    input_ids = []
    positions = []
    role_ids = []
    input_mask = []
    marker_mask = []
    marker_targets = []
    timesteps = []
    for item in batch:
        length = len(item["input_ids"])
        input_ids.append(pad_list(item["input_ids"], input_length, pad_token_id))
        positions.append(pad_list([min(max_position - 1, value) for value in item["positions"]], input_length, 0))
        role_ids.append(pad_list(item["role_ids"], input_length, PAD_ROLE))
        input_mask.append([False] * length + [True] * max(0, input_length - length))
        marker_mask.append(pad_list(item["marker_mask"], input_length, False))
        marker_targets.append(pad_list(item["marker_targets"], input_length, pad_token_id))
        timesteps.append(int(item["transition_index"]))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "positions": torch.tensor(positions, dtype=torch.long),
        "role_ids": torch.tensor(role_ids, dtype=torch.long),
        "input_key_padding_mask": torch.tensor(input_mask, dtype=torch.bool),
        "marker_mask": torch.tensor(marker_mask, dtype=torch.bool),
        "marker_targets": torch.tensor(marker_targets, dtype=torch.long),
        "timesteps": torch.tensor(timesteps, dtype=torch.long),
        "items": batch,
    }


def sinusoidal_encoding(positions: torch.Tensor, d_model: int) -> torch.Tensor:
    half_dim = d_model // 2
    frequencies = torch.exp(
        torch.arange(half_dim, device=positions.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half_dim - 1))
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if encoded.shape[-1] < d_model:
        encoded = F.pad(encoded, (0, d_model - encoded.shape[-1]))
    return encoded[:, :, :d_model]


class S4cSpanInfillingModel(nn.Module):
    def __init__(
        self,
        config: V2S4cConfig,
        vocab_size: int,
        pad_token_id: int,
        initial_embedding: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_token_id)
        if initial_embedding is not None and tuple(initial_embedding.shape) == tuple(self.token_embedding.weight.shape):
            self.token_embedding.weight.data.copy_(initial_embedding)
        self.timestep_embedding = nn.Embedding(max(1, len(config.ratios) - 1), config.d_model)
        self.role_embedding = nn.Embedding(3, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size)

    def forward(self, batch):
        hidden = self.token_embedding(batch["input_ids"])
        hidden = hidden + sinusoidal_encoding(batch["positions"], self.config.d_model).to(hidden.dtype)
        hidden = hidden + self.timestep_embedding(batch["timesteps"]).unsqueeze(1)
        hidden = hidden + self.role_embedding(batch["role_ids"])
        encoded = self.encoder(hidden, src_key_padding_mask=batch["input_key_padding_mask"])
        logits = self.output(self.norm(encoded))
        marker_logits = logits[batch["marker_mask"]]
        marker_targets = batch["marker_targets"][batch["marker_mask"]]
        loss = F.cross_entropy(marker_logits, marker_targets, ignore_index=self.pad_token_id)
        return loss, logits

    @torch.no_grad()
    def predict(self, batch):
        _loss, logits = self.forward(batch)
        return logits.argmax(dim=-1)


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for right_index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[right_index - 1] + 1)
            else:
                current.append(max(previous[right_index], current[-1]))
        previous = current
    return previous[-1]


def f1_from_counts(overlap: int, predicted_count: int, target_count: int) -> float:
    if predicted_count == 0 or target_count == 0 or overlap == 0:
        return 0.0
    precision = overlap / predicted_count
    recall = overlap / target_count
    return 2 * precision * recall / max(1e-12, precision + recall)


def dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def repetition_rate(words: list[str]) -> float:
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    return max(0, len(bigrams) - len(set(bigrams))) / len(bigrams)


def duplicate_rate(items: list[int]) -> float:
    filtered = [item for item in items if item >= 0]
    if not filtered:
        return 0.0
    return max(0, len(filtered) - len(set(filtered))) / len(filtered)


def lexical_metrics(prediction: str, target: str, visible_context_text: str, original_text: str) -> dict[str, float]:
    predicted_words = word_tokens(prediction)
    target_words = word_tokens(target)
    target_content = dedupe(content_words(target))[:12]
    context_content = dedupe(content_words(visible_context_text))[:16]
    original_content = dedupe(content_words(original_text))[:16]
    predicted_counter = Counter(predicted_words)
    target_counter = Counter(target_words)
    overlap = sum((predicted_counter & target_counter).values())
    lcs = lcs_length(predicted_words, target_words)
    predicted_set = set(predicted_words)
    target_entities = dedupe(surface_entities(target))[:8]
    return {
        "span_token_f1": f1_from_counts(overlap, len(predicted_words), len(target_words)),
        "span_rouge_l_f1": f1_from_counts(lcs, len(predicted_words), len(target_words)),
        "content_recall": (
            sum(1 for item in target_content if item in predicted_set) / len(target_content)
            if target_content
            else 0.0
        ),
        "context_copy_rate": (
            sum(1 for item in context_content if item in predicted_set) / len(context_content)
            if context_content
            else 0.0
        ),
        "original_content_recall": (
            sum(1 for item in original_content if item in predicted_set) / len(original_content)
            if original_content
            else 0.0
        ),
        "entity_recall": (
            sum(1 for item in target_entities if item in predicted_set) / len(target_entities)
            if target_entities
            else 0.0
        ),
        "repetition_rate": repetition_rate(predicted_words),
        "nonempty": 1.0 if prediction.strip() else 0.0,
        "target_word_count": float(len(target_words)),
    }


def marker_metrics(
    predicted_ids: list[int],
    target_ids: list[int],
    visible_context_ids: list[int],
    special_ids: set[int],
) -> dict[str, float]:
    marker_count = len(target_ids)
    if marker_count <= 0:
        return {
            "masked_token_accuracy": 0.0,
            "span_token_accuracy": 0.0,
            "span_exact_match": 0.0,
            "duplicate_prediction_rate": 0.0,
            "context_copy_leakage": 0.0,
            "correct_marker_count": 0.0,
            "marker_count": 0.0,
        }
    correct = sum(1 for predicted, target in zip(predicted_ids, target_ids) if int(predicted) == int(target))
    predicted_content_ids = [int(item) for item in predicted_ids if int(item) not in special_ids]
    visible_context_set = {int(item) for item in visible_context_ids if int(item) not in special_ids}
    copied = sum(1 for item in predicted_content_ids if item in visible_context_set)
    return {
        "masked_token_accuracy": correct / marker_count,
        "span_token_accuracy": correct / marker_count,
        "span_exact_match": 1.0 if correct == marker_count else 0.0,
        "duplicate_prediction_rate": duplicate_rate(predicted_content_ids),
        "context_copy_leakage": copied / marker_count,
        "correct_marker_count": float(correct),
        "marker_count": float(marker_count),
    }


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def make_base_examples(texts: list[str], payloads: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    examples = []
    for index in indices:
        examples.append(
            {
                "sample_id": index,
                "original_text": texts[index],
                "states": payloads[index]["states"],
            }
        )
    return examples


def make_transition_examples(
    config: V2S4cConfig,
    tokenizer,
    base_examples: list[dict[str, Any]],
    schedule_name: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    ratios = list(config.ratios)
    marker_id = marker_token_id(tokenizer)
    for base in base_examples:
        for transition_index, (from_ratio, to_ratio) in enumerate(zip(ratios[:-1], ratios[1:])):
            input_state = base["states"][schedule_name][from_ratio]
            target_schedule = "importance_schedule" if schedule_name == "position_only_schedule" else schedule_name
            context_state = base["states"][target_schedule][from_ratio]
            target_state = base["states"][target_schedule][to_ratio]
            current_positions = set(input_state["positions"])
            target_position_to_id = {
                int(position): int(token_id) for position, token_id in zip(target_state["positions"], target_state["ids"])
            }
            delta_positions = [int(position) for position in target_state["positions"] if int(position) not in current_positions]
            delta_pairs = [(position, target_position_to_id[position]) for position in delta_positions]
            if not delta_pairs:
                continue
            delta_pairs = sorted(delta_pairs, key=lambda item: item[0])[: config.max_masked_positions]
            delta_position_set = {position for position, _token_id in delta_pairs}
            context_pairs = [
                (int(position), int(token_id))
                for position, token_id in zip(input_state["positions"], input_state["ids"])
                if int(position) not in delta_position_set
            ]
            marker_pairs = [(position, marker_id, token_id) for position, token_id in delta_pairs]
            encoder_items = [(position, token_id, CONTEXT_ROLE, False, tokenizer.pad_token_id) for position, token_id in context_pairs]
            encoder_items.extend(
                (position, marker_id, MARKER_ROLE, True, token_id) for position, _marker_id, token_id in marker_pairs
            )
            encoder_items = sorted(encoder_items, key=lambda item: (item[0], item[2]))
            if not encoder_items:
                encoder_items = [(0, tokenizer.pad_token_id, PAD_ROLE, False, tokenizer.pad_token_id)]
            target_ids = [token_id for _position, token_id in delta_pairs]
            visible_context_ids = [int(token_id) for _position, token_id in context_pairs]
            examples.append(
                {
                    "sample_id": base["sample_id"],
                    "schedule": schedule_name,
                    "transition_index": transition_index,
                    "transition": f"{from_ratio:.2f}->{to_ratio:.2f}",
                    "from_ratio": from_ratio,
                    "to_ratio": to_ratio,
                    "input_ids": [token_id for _position, token_id, _role, _is_marker, _target in encoder_items],
                    "positions": [position for position, _token_id, _role, _is_marker, _target in encoder_items],
                    "role_ids": [role for _position, _token_id, role, _is_marker, _target in encoder_items],
                    "marker_mask": [is_marker for _position, _token_id, _role, is_marker, _target in encoder_items],
                    "marker_targets": [target for _position, _token_id, _role, _is_marker, target in encoder_items],
                    "input_text": str(input_state["text"]),
                    "visible_context_text": str(input_state["text"]),
                    "reference_context_text": str(context_state["text"]),
                    "visible_context_ids": visible_context_ids,
                    "delta_positions": [position for position, _token_id in delta_pairs],
                    "target_ids": target_ids,
                    "target_text": decode_ids(tokenizer, target_ids),
                    "target_state_text": str(target_state["text"]),
                    "original_text": base["original_text"],
                }
            )
    return examples


def train_model(config, tokenizer, device, train_examples, initial_embedding):
    model = S4cSpanInfillingModel(
        config,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        initial_embedding=initial_embedding,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    train_loader = make_loader(train_examples, config.train_batch_size, shuffle=True)
    losses: list[float] = []
    model.train()
    for _epoch in range(config.reverse_epochs):
        for raw_batch in train_loader:
            batch = move_batch(collate_examples(raw_batch, tokenizer.pad_token_id, config.max_length), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss, _logits = model(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {float(loss.detach().cpu())}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
    return model, losses


def aggregate_rows(rows: list[dict[str, Any]], losses: list[float]) -> dict[str, float]:
    metric_names = [
        "span_token_accuracy",
        "span_exact_match",
        "span_token_f1",
        "span_rouge_l_f1",
        "content_recall",
        "context_copy_rate",
        "context_copy_leakage",
        "original_content_recall",
        "entity_recall",
        "repetition_rate",
        "duplicate_prediction_rate",
        "nonempty",
        "target_word_count",
        "marker_count",
    ]
    aggregate = {name: mean([row["metrics"][name] for row in rows]) for name in metric_names}
    correct_total = sum(row["metrics"]["correct_marker_count"] for row in rows)
    marker_total = sum(row["metrics"]["marker_count"] for row in rows)
    aggregate["masked_token_accuracy"] = correct_total / marker_total if marker_total else 0.0
    aggregate["eval_loss"] = mean(losses)
    aggregate["eval_ppl"] = math.exp(min(20.0, aggregate["eval_loss"])) if math.isfinite(aggregate["eval_loss"]) else float("inf")
    return aggregate


def evaluate_model(config, tokenizer, model, device, eval_examples):
    model.eval()
    eval_loader = make_loader(eval_examples, config.eval_batch_size, shuffle=False)
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    transition_losses: dict[str, list[float]] = defaultdict(list)
    special_ids = {int(item) for item in tokenizer.all_special_ids}
    with torch.no_grad():
        for raw_batch in eval_loader:
            batch = move_batch(collate_examples(raw_batch, tokenizer.pad_token_id, config.max_length), device)
            loss, logits = model(batch)
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            for item in raw_batch:
                transition_losses[item["transition"]].append(loss_value)
            predicted_matrix = logits.argmax(dim=-1).detach().cpu()
            marker_mask = batch["marker_mask"].detach().cpu()
            marker_targets = batch["marker_targets"].detach().cpu()
            for row_index, item in enumerate(raw_batch):
                row_mask = marker_mask[row_index]
                predicted_ids = [int(value) for value in predicted_matrix[row_index][row_mask].tolist()]
                target_ids = [int(value) for value in marker_targets[row_index][row_mask].tolist()]
                prediction = decode_ids(tokenizer, predicted_ids)
                metrics = lexical_metrics(
                    prediction,
                    item["target_text"],
                    item["visible_context_text"],
                    item["original_text"],
                )
                metrics.update(marker_metrics(predicted_ids, target_ids, item["visible_context_ids"], special_ids))
                rows.append(
                    {
                        "sample_id": item["sample_id"],
                        "schedule": item["schedule"],
                        "transition": item["transition"],
                        "input": item["input_text"],
                        "visible_context": item["visible_context_text"],
                        "reference_context": item["reference_context_text"],
                        "delta_positions": item["delta_positions"],
                        "target": item["target_text"],
                        "target_ids": target_ids,
                        "target_state": item["target_state_text"],
                        "original": item["original_text"],
                        "prediction": prediction,
                        "predicted_token_ids": predicted_ids,
                        "metrics": metrics,
                    }
                )
    aggregate = aggregate_rows(rows, losses)
    by_transition = {}
    for transition in sorted(set(row["transition"] for row in rows)):
        transition_rows = [row for row in rows if row["transition"] == transition]
        by_transition[transition] = aggregate_rows(transition_rows, transition_losses[transition])
    return {"metrics": aggregate, "by_transition": by_transition, "samples": rows}


def score_for_gate(metrics: dict[str, float]) -> float:
    overlap = (
        1.80 * metrics.get("masked_token_accuracy", 0.0)
        + 0.50 * metrics.get("span_token_f1", 0.0)
        + 0.35 * metrics.get("span_rouge_l_f1", 0.0)
        + 0.20 * metrics.get("span_exact_match", 0.0)
    )
    semantic = (
        0.45 * metrics.get("content_recall", 0.0)
        + 0.25 * metrics.get("original_content_recall", 0.0)
        + 0.25 * metrics.get("entity_recall", 0.0)
        + 0.05 * metrics.get("nonempty", 0.0)
    )
    loss_bonus = 1.0 / (1.0 + max(0.0, metrics.get("eval_loss", 1e9)))
    copy_penalty = 0.08 * metrics.get("context_copy_leakage", 0.0) + 0.05 * metrics.get("context_copy_rate", 0.0)
    repetition_penalty = 0.12 * metrics.get("duplicate_prediction_rate", 0.0) + 0.08 * metrics.get("repetition_rate", 0.0)
    return overlap + semantic + loss_bonus - repetition_penalty - copy_penalty


def evaluate_gates(metrics: dict[str, dict[str, float]], tolerance: float, required_schedules: tuple[str, ...]) -> dict[str, Any]:
    required = set(required_schedules)
    available = set(metrics)
    losses_finite = all(math.isfinite(item.get("eval_loss", float("inf"))) for item in metrics.values())
    scores = {name: score_for_gate(item) for name, item in metrics.items()}
    best_name = max(scores, key=scores.get) if scores else ""
    importance = metrics.get("importance_schedule", {})
    random_item = metrics.get("random_schedule", {})
    position_only = metrics.get("position_only_schedule", {})
    importance_score = scores.get("importance_schedule")
    random_score = scores.get("random_schedule")
    position_score = scores.get("position_only_schedule")

    def delta(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return left - right

    def beats(left: float | None, right: float | None, margin: float = tolerance) -> bool:
        return left is not None and right is not None and left > right + margin

    importance_semantic = importance.get("content_recall", 0.0) + importance.get("entity_recall", 0.0)
    random_semantic = random_item.get("content_recall", 0.0) + random_item.get("entity_recall", 0.0)
    repetition_non_worse = (
        importance.get("duplicate_prediction_rate", 1.0) <= random_item.get("duplicate_prediction_rate", 0.0) + tolerance
        and importance.get("repetition_rate", 1.0) <= random_item.get("repetition_rate", 0.0) + tolerance
    )

    gates = {
        "S4C-G-RUN": {"pass": required.issubset(available), "detail": sorted(available)},
        "S4C-G-LOSS-FINITE": {"pass": losses_finite, "detail": {name: item["eval_loss"] for name, item in metrics.items()}},
        "S4C-G-IMPORTANCE-BEATS-RANDOM": {
            "pass": beats(importance_score, random_score),
            "detail": {
                "importance_score": importance_score,
                "random_score": random_score,
                "delta": delta(importance_score, random_score),
                "tolerance": tolerance,
            },
        },
        "S4C-G-IMPORTANCE-BEATS-POSITION-ONLY": {
            "pass": beats(importance_score, position_score),
            "detail": {
                "importance_score": importance_score,
                "position_only_score": position_score,
                "delta": delta(importance_score, position_score),
                "tolerance": tolerance,
            },
        },
        "S4C-G-MASKED-TOKEN-ACCURACY": {
            "pass": importance.get("masked_token_accuracy", 0.0) > random_item.get("masked_token_accuracy", 0.0),
            "detail": {
                "importance_masked_token_accuracy": importance.get("masked_token_accuracy"),
                "random_masked_token_accuracy": random_item.get("masked_token_accuracy"),
                "delta": delta(importance.get("masked_token_accuracy"), random_item.get("masked_token_accuracy")),
            },
        },
        "S4C-G-ENTITY-CONTENT-IMPROVEMENT": {
            "pass": importance_semantic > random_semantic,
            "detail": {
                "importance_content_recall": importance.get("content_recall"),
                "random_content_recall": random_item.get("content_recall"),
                "importance_entity_recall": importance.get("entity_recall"),
                "random_entity_recall": random_item.get("entity_recall"),
                "importance_semantic_sum": importance_semantic,
                "random_semantic_sum": random_semantic,
            },
        },
        "S4C-G-COPY-LEAKAGE": {
            "pass": importance.get("context_copy_leakage", 1.0) <= random_item.get("context_copy_leakage", 0.0) + tolerance,
            "detail": {
                "importance_context_copy_leakage": importance.get("context_copy_leakage"),
                "random_context_copy_leakage": random_item.get("context_copy_leakage"),
                "delta": delta(importance.get("context_copy_leakage"), random_item.get("context_copy_leakage")),
                "tolerance": tolerance,
            },
        },
        "S4C-G-REPETITION-NONWORSE": {
            "pass": repetition_non_worse,
            "detail": {
                "importance_duplicate_prediction_rate": importance.get("duplicate_prediction_rate"),
                "random_duplicate_prediction_rate": random_item.get("duplicate_prediction_rate"),
                "importance_repetition_rate": importance.get("repetition_rate"),
                "random_repetition_rate": random_item.get("repetition_rate"),
                "tolerance": tolerance,
            },
        },
        "S4C-G-BEST-IDENTIFIED": {
            "pass": bool(best_name),
            "detail": {"best_schedule": best_name, "best_score": scores.get(best_name, 0.0), "scores": scores},
        },
    }
    gates["overall_pass"] = bool(
        gates["S4C-G-RUN"]["pass"]
        and gates["S4C-G-LOSS-FINITE"]["pass"]
        and gates["S4C-G-IMPORTANCE-BEATS-RANDOM"]["pass"]
        and gates["S4C-G-IMPORTANCE-BEATS-POSITION-ONLY"]["pass"]
        and gates["S4C-G-MASKED-TOKEN-ACCURACY"]["pass"]
        and gates["S4C-G-ENTITY-CONTENT-IMPROVEMENT"]["pass"]
    )
    gates["process_ready"] = bool(gates["S4C-G-RUN"]["pass"] and gates["S4C-G-LOSS-FINITE"]["pass"])
    gates["structure_review_needed"] = not gates["overall_pass"] or not gates["S4C-G-REPETITION-NONWORSE"]["pass"]
    gates["s5_ready"] = False
    return gates


def render_summary(result: dict[str, Any]) -> str:
    run_info = result["run_info"]
    gates = result["gates"]
    metrics = result["schedule_metrics"]
    by_transition = result["transition_metrics"]
    lines = [
        "# LACE V2 S4c Summary",
        "",
        "## Run Info",
        "",
        f"- phase: `{run_info['phase']}`",
        f"- experiment: `{run_info['experiment_name']}`",
        f"- model: `{run_info['model_name']}`",
        f"- data: `{run_info['text_source']}`",
        f"- train samples: `{run_info['train_samples']}`",
        f"- eval samples: `{run_info['eval_samples']}`",
        f"- device: `{run_info['device']}`",
        f"- ratios: `{run_info['ratios']}`",
        f"- schedules: `{run_info['schedules']}`",
        "",
        "## Gates",
        "",
        "| Gate | Pass | Detail |",
        "|---|---:|---|",
    ]
    for gate_name, gate_value in gates.items():
        if isinstance(gate_value, dict):
            detail = json.dumps(gate_value.get("detail"), sort_keys=True)
            lines.append(f"| `{gate_name}` | `{str(gate_value.get('pass')).lower()}` | `{detail}` |")
    lines.extend(
        [
            f"| `overall_pass` | `{str(gates.get('overall_pass')).lower()}` | Marker-position infilling signal. |",
            f"| `process_ready` | `{str(gates.get('process_ready')).lower()}` | Whether the schedule comparison ran cleanly. |",
            f"| `structure_review_needed` | `{str(gates.get('structure_review_needed')).lower()}` | Whether decoder/control changes remain necessary before S5. |",
            f"| `s5_ready` | `{str(gates.get('s5_ready')).lower()}` | S4c is still constrained position-level infilling, so this remains false. |",
            "",
            "## Schedule Metrics",
            "",
            "| Schedule | Loss | PPL | Mask Acc | Span Acc | Span EM | Span F1 | Content | Context Leak | Entity | Duplicate | Repetition | Score |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for schedule_name in run_info["schedules"]:
        item = metrics[schedule_name]
        lines.append(
            "| `{}` | {:.4f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                schedule_name,
                item["eval_loss"],
                item["eval_ppl"],
                item["masked_token_accuracy"],
                item["span_token_accuracy"],
                item["span_exact_match"],
                item["span_token_f1"],
                item["content_recall"],
                item["context_copy_leakage"],
                item["entity_recall"],
                item["duplicate_prediction_rate"],
                item["repetition_rate"],
                score_for_gate(item),
            )
        )
    lines.extend(["", "## Transition Metrics", ""])
    for schedule_name in run_info["schedules"]:
        lines.extend(
            [
                f"### `{schedule_name}`",
                "",
                "| Transition | Loss | Mask Acc | Span Acc | Span EM | Span F1 | Content | Context Leak | Entity | Duplicate | Repetition | Score |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for transition, item in by_transition[schedule_name].items():
            lines.append(
                "| `{}` | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    transition,
                    item["eval_loss"],
                    item["masked_token_accuracy"],
                    item["span_token_accuracy"],
                    item["span_exact_match"],
                    item["span_token_f1"],
                    item["content_recall"],
                    item["context_copy_leakage"],
                    item["entity_recall"],
                    item["duplicate_prediction_rate"],
                    item["repetition_rate"],
                    score_for_gate(item),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Guardrail",
            "",
            "S4c is a constrained position-level infilling experiment, not open-ended generation.",
            "Contiguous spans are represented as multiple marker positions in this first version; the head predicts each marker independently.",
            "The key comparison is whether semantic skeleton content improves masked-position token prediction beyond random and position-only controls.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config: V2S4cConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer, T5EncoderModel

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts, text_source = load_texts(config)
    rng = random.Random(config.seed)
    rng.shuffle(texts)
    train_count = min(config.max_train_samples, max(1, len(texts) - config.max_eval_samples))
    eval_count = min(config.max_eval_samples, len(texts) - train_count)
    if eval_count <= 0:
        raise ValueError("At least one eval sample is required.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    skeleton_encoder = T5EncoderModel.from_pretrained(config.model_name).to(device)
    skeleton_encoder.eval()
    initial_embedding = skeleton_encoder.get_input_embeddings().weight.detach().cpu()
    payloads = build_schedule_payloads(config, tokenizer, skeleton_encoder, device, texts)
    del skeleton_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_indices = list(range(train_count))
    eval_indices = list(range(train_count, train_count + eval_count))
    train_bases = make_base_examples(texts, payloads, train_indices)
    eval_bases = make_base_examples(texts, payloads, eval_indices)

    schedule_metrics: dict[str, dict[str, float]] = {}
    transition_metrics: dict[str, dict[str, dict[str, float]]] = {}
    training_losses: dict[str, list[float]] = {}
    sample_rows: list[dict[str, Any]] = []
    for schedule_name in config.schedules:
        train_examples = make_transition_examples(config, tokenizer, train_bases, schedule_name)
        eval_examples = make_transition_examples(config, tokenizer, eval_bases, schedule_name)
        model, losses = train_model(config, tokenizer, device, train_examples, initial_embedding)
        evaluation = evaluate_model(config, tokenizer, model, device, eval_examples)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        training_losses[schedule_name] = losses
        schedule_metrics[schedule_name] = evaluation["metrics"]
        transition_metrics[schedule_name] = evaluation["by_transition"]
        for row in evaluation["samples"][: config.sample_output_count]:
            sample_rows.append(row)

    result = {
        "run_info": {
            "phase": "v2_s4c",
            "experiment_name": "S4c span-infilling reverse decoder",
            "config": asdict(config),
            "text_source": text_source,
            "train_samples": train_count,
            "eval_samples": eval_count,
            "model_name": config.model_name,
            "device": device,
            "ratios": config.ratios,
            "schedules": config.schedules,
        },
        "schedule_metrics": schedule_metrics,
        "transition_metrics": transition_metrics,
        "training_losses": training_losses,
    }
    result["gates"] = evaluate_gates(schedule_metrics, config.gate_tolerance, config.schedules)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_summary(result), encoding="utf-8")
    with (output_dir / "span_infilling_samples.jsonl").open("w", encoding="utf-8") as file:
        for row in sample_rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def main() -> None:
    config = parse_args()
    result = run(config)
    print(render_summary(result))


if __name__ == "__main__":
    main()
