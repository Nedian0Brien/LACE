"""LACE V2 S2a positional encoding comparison.

S2a keeps the attention-selected semantic skeleton fixed and compares how
different positional encodings for skeleton tokens affect short
skeleton-to-text reconstruction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
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
CONDITIONS = (
    "no_position",
    "coarse_bins",
    "learned_absolute",
    "sinusoidal_absolute",
    "relative_position_bias",
    "rotary_position",
)
STANDARD_CONDITIONS = (
    "learned_absolute",
    "sinusoidal_absolute",
    "relative_position_bias",
    "rotary_position",
)


@dataclass(frozen=True)
class V2S2aConfig:
    model_name: str = "t5-small"
    max_train_samples: int = 768
    max_eval_samples: int = 192
    max_length: int = 128
    target_max_length: int = 96
    skeleton_batch_size: int = 16
    train_batch_size: int = 16
    eval_batch_size: int = 16
    epochs: int = 1
    learning_rate: float = 5e-4
    d_model: int = 512
    num_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    output_dir: str = "/kaggle/working/lace_v2_s2a"
    input_text_file: str | None = None
    use_hf_dataset: bool = True
    hf_dataset_name: str = "wikitext"
    hf_dataset_config: str = "wikitext-2-raw-v1"
    hf_dataset_split: str = "train"
    seed: int = 42
    keep_ratio: float = 0.25
    sample_output_count: int = 24
    min_text_words: int = 6
    conditions: tuple[str, ...] = CONDITIONS


def parse_conditions(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(items) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    if not items:
        raise ValueError("At least one condition is required.")
    return items


def parse_args() -> V2S2aConfig:
    parser = argparse.ArgumentParser(description="Run LACE V2 S2a positional encoding comparison.")
    parser.add_argument("--model-name", default=V2S2aConfig.model_name)
    parser.add_argument("--max-train-samples", type=int, default=V2S2aConfig.max_train_samples)
    parser.add_argument("--max-eval-samples", type=int, default=V2S2aConfig.max_eval_samples)
    parser.add_argument("--max-length", type=int, default=V2S2aConfig.max_length)
    parser.add_argument("--target-max-length", type=int, default=V2S2aConfig.target_max_length)
    parser.add_argument("--skeleton-batch-size", type=int, default=V2S2aConfig.skeleton_batch_size)
    parser.add_argument("--train-batch-size", type=int, default=V2S2aConfig.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=V2S2aConfig.eval_batch_size)
    parser.add_argument("--epochs", type=int, default=V2S2aConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=V2S2aConfig.learning_rate)
    parser.add_argument("--d-model", type=int, default=V2S2aConfig.d_model)
    parser.add_argument("--num-heads", type=int, default=V2S2aConfig.num_heads)
    parser.add_argument("--encoder-layers", type=int, default=V2S2aConfig.encoder_layers)
    parser.add_argument("--decoder-layers", type=int, default=V2S2aConfig.decoder_layers)
    parser.add_argument("--dropout", type=float, default=V2S2aConfig.dropout)
    parser.add_argument("--max-grad-norm", type=float, default=V2S2aConfig.max_grad_norm)
    parser.add_argument("--output-dir", default=os.environ.get("LACE_OUTPUT_DIR", V2S2aConfig.output_dir))
    parser.add_argument("--input-text-file", default=os.environ.get("LACE_INPUT_TEXT_FILE"))
    parser.add_argument("--use-hf-dataset", action=argparse.BooleanOptionalAction, default=V2S2aConfig.use_hf_dataset)
    parser.add_argument("--hf-dataset-name", default=V2S2aConfig.hf_dataset_name)
    parser.add_argument("--hf-dataset-config", default=V2S2aConfig.hf_dataset_config)
    parser.add_argument("--hf-dataset-split", default=V2S2aConfig.hf_dataset_split)
    parser.add_argument("--seed", type=int, default=V2S2aConfig.seed)
    parser.add_argument("--keep-ratio", type=float, default=V2S2aConfig.keep_ratio)
    parser.add_argument("--sample-output-count", type=int, default=V2S2aConfig.sample_output_count)
    parser.add_argument("--min-text-words", type=int, default=V2S2aConfig.min_text_words)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    args = parser.parse_args()
    if args.keep_ratio <= 0 or args.keep_ratio > 1:
        raise ValueError("--keep-ratio must be in the interval (0, 1].")
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads.")
    if (args.d_model // args.num_heads) % 2 != 0:
        raise ValueError("Head dimension must be even for rotary position.")
    return V2S2aConfig(
        model_name=args.model_name,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_length=args.max_length,
        target_max_length=args.target_max_length,
        skeleton_batch_size=args.skeleton_batch_size,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        num_heads=args.num_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
        max_grad_norm=args.max_grad_norm,
        output_dir=args.output_dir,
        input_text_file=args.input_text_file,
        use_hf_dataset=args.use_hf_dataset,
        hf_dataset_name=args.hf_dataset_name,
        hf_dataset_config=args.hf_dataset_config,
        hf_dataset_split=args.hf_dataset_split,
        seed=args.seed,
        keep_ratio=args.keep_ratio,
        sample_output_count=args.sample_output_count,
        min_text_words=args.min_text_words,
        conditions=parse_conditions(args.conditions),
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


def load_texts(config: V2S2aConfig) -> tuple[list[str], str]:
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
            texts: list[str] = []
            for row in dataset:
                text = normalize_text_line(str(row.get("text", "")))
                if word_count(text) >= config.min_text_words:
                    texts.append(text)
                if len(texts) >= total_needed:
                    break
            if texts:
                return texts, f"hf:{config.hf_dataset_name}/{config.hf_dataset_config}:{config.hf_dataset_split}"
        except Exception as exc:  # pragma: no cover - Kaggle dependency/network fallback.
            print(f"[warn] Falling back to built-in texts after dataset load failure: {exc}", file=sys.stderr)

    texts = [text for text in FALLBACK_TEXTS if word_count(text) >= config.min_text_words]
    repeated: list[str] = []
    while len(repeated) < total_needed:
        repeated.extend(texts)
    return repeated[:total_needed], "fallback-repeated"


def special_token_mask(input_ids, tokenizer):
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for special_id in tokenizer.all_special_ids:
        mask |= input_ids == int(special_id)
    return mask


def active_positions(valid_row) -> list[int]:
    return [index for index, is_active in enumerate(valid_row.tolist()) if bool(is_active)]


def target_keep_count(active_count: int, keep_ratio: float) -> int:
    if active_count <= 0:
        return 0
    return max(1, min(active_count, int(round(active_count * keep_ratio))))


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


def build_skeletons(config: V2S2aConfig, tokenizer, encoder, device: str, texts: list[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
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
                positions = active_positions(valid_mask[row].cpu())
                keep_count = target_keep_count(len(positions), config.keep_ratio)
                ranked = sorted(positions, key=lambda index: (-float(scores[row, index].item()), index))
                kept_positions = sorted(ranked[:keep_count])
                token_ids = [int(input_ids[row, position].item()) for position in kept_positions]
                payloads.append(
                    {
                        "sample_id": start + row,
                        "skeleton_token_ids": token_ids,
                        "positions": kept_positions,
                        "skeleton_text": tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True),
                    }
                )
    return payloads


def coarse_bin_ids(positions: list[int], max_position: int) -> list[int]:
    bins = []
    denominator = max(1, max_position - 1)
    for position in positions:
        normalized = position / denominator
        if normalized < 0.33:
            bins.append(0)
        elif normalized < 0.66:
            bins.append(1)
        else:
            bins.append(2)
    return bins


class SkeletonDataset:
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def make_examples(tokenizer, config: V2S2aConfig, texts: list[str], payloads: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    examples = []
    for index in indices:
        target = tokenizer(
            texts[index],
            truncation=True,
            max_length=config.target_max_length,
            return_tensors=None,
        )["input_ids"]
        payload = payloads[index]
        examples.append(
            {
                "sample_id": index,
                "skeleton_token_ids": payload["skeleton_token_ids"],
                "positions": payload["positions"],
                "coarse_bins": coarse_bin_ids(payload["positions"], config.max_length),
                "target_ids": target,
                "target_text": texts[index],
                "skeleton_text": payload["skeleton_text"],
            }
        )
    return examples


def pad_list(values: list[int], length: int, pad_value: int) -> list[int]:
    return values[:length] + [pad_value] * max(0, length - len(values))


def collate_examples(batch: list[dict[str, Any]], pad_token_id: int, max_position: int) -> dict[str, Any]:
    skeleton_length = max(1, max(len(item["skeleton_token_ids"]) for item in batch))
    target_length = max(2, max(len(item["target_ids"]) for item in batch))
    skeleton_ids = []
    positions = []
    coarse_bins = []
    skeleton_mask = []
    target_ids = []
    for item in batch:
        length = len(item["skeleton_token_ids"])
        skeleton_ids.append(pad_list(item["skeleton_token_ids"], skeleton_length, pad_token_id))
        positions.append(pad_list([min(max_position - 1, value) for value in item["positions"]], skeleton_length, 0))
        coarse_bins.append(pad_list(item["coarse_bins"], skeleton_length, 0))
        skeleton_mask.append([False] * length + [True] * max(0, skeleton_length - length))
        target_ids.append(pad_list(item["target_ids"], target_length, pad_token_id))
    return {
        "skeleton_ids": torch.tensor(skeleton_ids, dtype=torch.long),
        "positions": torch.tensor(positions, dtype=torch.long),
        "coarse_bins": torch.tensor(coarse_bins, dtype=torch.long),
        "skeleton_key_padding_mask": torch.tensor(skeleton_mask, dtype=torch.bool),
        "target_ids": torch.tensor(target_ids, dtype=torch.long),
        "items": batch,
    }


def make_loader(examples: list[dict[str, Any]], batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    return DataLoader(SkeletonDataset(examples), batch_size=batch_size, shuffle=shuffle, collate_fn=lambda items: items)


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


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value[..., ::2], value[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(-2)


def apply_rotary(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    head_dim = value.shape[-1]
    half_dim = head_dim // 2
    frequencies = torch.exp(
        torch.arange(half_dim, device=value.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half_dim - 1))
    )
    angles = positions.float()[:, None, :, None] * frequencies[None, None, None, :]
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)
    return (value * torch.cos(angles)) + (rotate_half(value) * torch.sin(angles))


class PositionalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_length: int, dropout: float, mode: str) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.mode = mode
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relative_bias = nn.Embedding((2 * max_length) - 1, num_heads) if mode == "relative_position_bias" else None
        self.max_length = max_length

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        qkv = self.qkv(hidden).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if self.mode == "rotary_position":
            query = apply_rotary(query, positions)
            key = apply_rotary(key, positions)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.relative_bias is not None:
            relative = positions[:, :, None] - positions[:, None, :]
            relative = relative.clamp(min=-(self.max_length - 1), max=self.max_length - 1) + self.max_length - 1
            bias = self.relative_bias(relative).permute(0, 3, 1, 2)
            scores = scores + bias
        scores = scores.masked_fill(key_padding_mask[:, None, None, :], -1e4)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out(output)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_length: int, dropout: float, mode: str) -> None:
        super().__init__()
        self.attention = PositionalSelfAttention(d_model, num_heads, max_length, dropout, mode)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, positions: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(hidden + self.dropout(self.attention(hidden, positions, key_padding_mask)))
        hidden = self.norm2(hidden + self.dropout(self.feed_forward(hidden)))
        return hidden


class ReconstructionModel(nn.Module):
    def __init__(
        self,
        config: V2S2aConfig,
        vocab_size: int,
        pad_token_id: int,
        eos_token_id: int,
        mode: str,
        initial_embedding: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.token_embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_token_id)
        if initial_embedding is not None and tuple(initial_embedding.shape) == tuple(self.token_embedding.weight.shape):
            self.token_embedding.weight.data.copy_(initial_embedding)
        self.learned_position = nn.Embedding(config.max_length, config.d_model)
        self.coarse_position = nn.Embedding(3, config.d_model)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(config.d_model, config.num_heads, config.max_length, config.dropout, mode)
                for _ in range(config.encoder_layers)
            ]
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.decoder_position = nn.Embedding(config.target_max_length + 2, config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size)

    def encode(self, skeleton_ids, positions, coarse_bins, key_padding_mask):
        hidden = self.token_embedding(skeleton_ids)
        if self.mode == "learned_absolute":
            hidden = hidden + self.learned_position(positions.clamp(max=self.config.max_length - 1))
        elif self.mode == "sinusoidal_absolute":
            hidden = hidden + sinusoidal_encoding(positions, self.config.d_model).to(hidden.dtype)
        elif self.mode == "coarse_bins":
            hidden = hidden + self.coarse_position(coarse_bins.clamp(min=0, max=2))
        for layer in self.encoder_layers:
            hidden = layer(hidden, positions, key_padding_mask)
        return hidden

    def decode(self, decoder_ids, memory, memory_key_padding_mask):
        seq_len = decoder_ids.shape[1]
        decoder_positions = torch.arange(seq_len, device=decoder_ids.device)[None, :].expand(decoder_ids.shape[0], -1)
        hidden = self.token_embedding(decoder_ids) + self.decoder_position(decoder_positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=decoder_ids.device)
        decoded = self.decoder(
            hidden,
            memory,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output(decoded)

    def forward(self, batch):
        target_ids = batch["target_ids"]
        decoder_input = torch.cat(
            [
                torch.full((target_ids.shape[0], 1), self.pad_token_id, dtype=torch.long, device=target_ids.device),
                target_ids[:, :-1],
            ],
            dim=1,
        )
        memory = self.encode(
            batch["skeleton_ids"],
            batch["positions"],
            batch["coarse_bins"],
            batch["skeleton_key_padding_mask"],
        )
        logits = self.decode(decoder_input, memory, batch["skeleton_key_padding_mask"])
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        return loss, logits

    @torch.no_grad()
    def generate(self, batch, max_length: int):
        memory = self.encode(
            batch["skeleton_ids"],
            batch["positions"],
            batch["coarse_bins"],
            batch["skeleton_key_padding_mask"],
        )
        generated = torch.full(
            (batch["skeleton_ids"].shape[0], 1),
            self.pad_token_id,
            dtype=torch.long,
            device=batch["skeleton_ids"].device,
        )
        for _step in range(max_length - 1):
            logits = self.decode(generated, memory, batch["skeleton_key_padding_mask"])
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if bool((next_token == self.eos_token_id).all()):
                break
        return generated[:, 1:]


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


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


def lexical_metrics(prediction: str, target: str, skeleton: str) -> dict[str, float]:
    predicted_words = word_tokens(prediction)
    target_words = word_tokens(target)
    predicted_counter = Counter(predicted_words)
    target_counter = Counter(target_words)
    overlap = sum((predicted_counter & target_counter).values())
    lcs = lcs_length(predicted_words, target_words)
    target_keywords = list(dict.fromkeys(content_words(target)))[:8]
    skeleton_keywords = list(dict.fromkeys(content_words(skeleton)))
    predicted_set = set(predicted_words)
    return {
        "token_f1": f1_from_counts(overlap, len(predicted_words), len(target_words)),
        "rouge_l_f1": f1_from_counts(lcs, len(predicted_words), len(target_words)),
        "keyword_recall": (
            sum(1 for item in target_keywords if item in predicted_set) / len(target_keywords)
            if target_keywords
            else 0.0
        ),
        "skeleton_coverage": (
            sum(1 for item in skeleton_keywords if item in predicted_set) / len(skeleton_keywords)
            if skeleton_keywords
            else 0.0
        ),
        "nonempty": 1.0 if prediction.strip() else 0.0,
    }


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def train_one_condition(config, tokenizer, device, condition_name, train_examples, eval_examples, initial_embedding):
    model = ReconstructionModel(
        config,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        mode=condition_name,
        initial_embedding=initial_embedding,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    losses: list[float] = []
    train_loader = make_loader(train_examples, config.train_batch_size, shuffle=True)
    model.train()
    for _epoch in range(config.epochs):
        for raw_batch in train_loader:
            batch = move_batch(collate_examples(raw_batch, tokenizer.pad_token_id, config.max_length), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss, _logits = model(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {condition_name}: {float(loss.detach().cpu())}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
    evaluation = evaluate_model(config, tokenizer, model, device, eval_examples)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return losses, evaluation


def evaluate_model(config, tokenizer, model, device, eval_examples):
    model.eval()
    eval_loader = make_loader(eval_examples, config.eval_batch_size, shuffle=False)
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_batch in eval_loader:
            batch = move_batch(collate_examples(raw_batch, tokenizer.pad_token_id, config.max_length), device)
            loss, _logits = model(batch)
            losses.append(float(loss.detach().cpu()))
            generated = model.generate(batch, config.target_max_length)
            predictions = tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for item, prediction in zip(raw_batch, predictions):
                metrics = lexical_metrics(prediction, item["target_text"], item["skeleton_text"])
                rows.append(
                    {
                        "sample_id": item["sample_id"],
                        "skeleton": item["skeleton_text"],
                        "target": item["target_text"],
                        "prediction": prediction,
                        "metrics": metrics,
                    }
                )
    metric_names = ["token_f1", "rouge_l_f1", "keyword_recall", "skeleton_coverage", "nonempty"]
    aggregate = {name: mean([row["metrics"][name] for row in rows]) for name in metric_names}
    aggregate["eval_loss"] = mean(losses)
    aggregate["eval_ppl"] = math.exp(min(20.0, aggregate["eval_loss"])) if math.isfinite(aggregate["eval_loss"]) else float("inf")
    return {"metrics": aggregate, "samples": rows}


def score_for_gate(metrics: dict[str, float]) -> float:
    overlap_score = metrics.get("token_f1", 0.0) + metrics.get("rouge_l_f1", 0.0)
    loss_bonus = 1.0 / (1.0 + max(0.0, metrics.get("eval_loss", 1e9)))
    return overlap_score + loss_bonus


def evaluate_gates(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    standard_available = [name for name in STANDARD_CONDITIONS if name in metrics]
    best_name = max(metrics, key=lambda name: score_for_gate(metrics[name])) if metrics else ""
    best_standard = max(standard_available, key=lambda name: score_for_gate(metrics[name])) if standard_available else ""
    best_standard_score = score_for_gate(metrics[best_standard]) if best_standard else 0.0
    no_position_score = score_for_gate(metrics.get("no_position", {}))
    coarse_score = score_for_gate(metrics.get("coarse_bins", {}))
    losses_finite = all(math.isfinite(item.get("eval_loss", float("inf"))) for item in metrics.values())
    gates = {
        "S2A-G-RUN": {"pass": bool(metrics), "detail": f"{len(metrics)} conditions evaluated."},
        "S2A-G-LOSS-FINITE": {"pass": losses_finite, "detail": {name: item["eval_loss"] for name, item in metrics.items()}},
        "S2A-G-STANDARD-BEATS-NONE": {
            "pass": best_standard_score > no_position_score,
            "detail": {"best_standard": best_standard, "best_standard_score": best_standard_score, "no_position_score": no_position_score},
        },
        "S2A-G-STANDARD-BEATS-COARSE": {
            "pass": best_standard_score > coarse_score,
            "detail": {"best_standard": best_standard, "best_standard_score": best_standard_score, "coarse_bins_score": coarse_score},
        },
        "S2A-G-BEST-IDENTIFIED": {
            "pass": bool(best_name),
            "detail": {"best_condition": best_name, "best_score": score_for_gate(metrics[best_name]) if best_name else 0.0},
        },
    }
    gates["overall_pass"] = bool(
        gates["S2A-G-RUN"]["pass"]
        and gates["S2A-G-LOSS-FINITE"]["pass"]
        and gates["S2A-G-STANDARD-BEATS-NONE"]["pass"]
        and gates["S2A-G-STANDARD-BEATS-COARSE"]["pass"]
        and gates["S2A-G-BEST-IDENTIFIED"]["pass"]
    )
    gates["s3_ready"] = bool(gates["S2A-G-BEST-IDENTIFIED"]["pass"])
    return gates


def render_summary(result: dict[str, Any]) -> str:
    run_info = result["run_info"]
    gates = result["gates"]
    metrics = result["condition_metrics"]
    lines = [
        "# LACE V2 S2a Summary",
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
        f"- keep ratio: `{run_info['keep_ratio']}`",
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
            f"| `overall_pass` | `{str(gates.get('overall_pass')).lower()}` | Positional encoding upgrade evidence. |",
            f"| `s3_ready` | `{str(gates.get('s3_ready')).lower()}` | Whether a best positional scaffold can be passed to S3. |",
            "",
            "## Condition Metrics",
            "",
            "| Condition | Loss | PPL | Token F1 | ROUGE-L | Keyword Recall | Skeleton Coverage | Nonempty |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition_name in sorted(metrics):
        item = metrics[condition_name]
        lines.append(
            "| `{}` | {:.4f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                condition_name,
                item["eval_loss"],
                item["eval_ppl"],
                item["token_f1"],
                item["rouge_l_f1"],
                item["keyword_recall"],
                item["skeleton_coverage"],
                item["nonempty"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "S2a compares positional encoding variants with the attention-selected skeleton fixed.",
            "It does not compare skeleton scorers and does not prove open-ended generation quality.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config: V2S2aConfig) -> dict[str, Any]:
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
    payloads = build_skeletons(config, tokenizer, skeleton_encoder, device, texts)
    del skeleton_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_indices = list(range(train_count))
    eval_indices = list(range(train_count, train_count + eval_count))
    train_examples = make_examples(tokenizer, config, texts, payloads, train_indices)
    eval_examples = make_examples(tokenizer, config, texts, payloads, eval_indices)

    condition_metrics: dict[str, dict[str, float]] = {}
    training_losses: dict[str, list[float]] = {}
    sample_rows: list[dict[str, Any]] = []
    for condition_name in config.conditions:
        losses, evaluation = train_one_condition(
            config,
            tokenizer,
            device,
            condition_name,
            train_examples,
            eval_examples,
            initial_embedding,
        )
        training_losses[condition_name] = losses
        condition_metrics[condition_name] = evaluation["metrics"]
        for row in evaluation["samples"][: config.sample_output_count]:
            row["condition"] = condition_name
            sample_rows.append(row)

    result = {
        "run_info": {
            "phase": "v2_s2a",
            "experiment_name": "S2a-positional encoding",
            "config": asdict(config),
            "text_source": text_source,
            "train_samples": train_count,
            "eval_samples": eval_count,
            "model_name": config.model_name,
            "device": device,
            "keep_ratio": config.keep_ratio,
            "conditions": config.conditions,
        },
        "condition_metrics": condition_metrics,
        "training_losses": training_losses,
    }
    result["gates"] = evaluate_gates(condition_metrics)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_summary(result), encoding="utf-8")
    with (output_dir / "reconstruction_samples.jsonl").open("w", encoding="utf-8") as file:
        for row in sample_rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def main() -> None:
    config = parse_args()
    result = run(config)
    print(render_summary(result))


if __name__ == "__main__":
    main()
