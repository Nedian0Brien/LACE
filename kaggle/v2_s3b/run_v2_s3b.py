"""LACE V2 S3b probe calibration.

S3b trains one reverse probe on attention-terminal inputs, then evaluates that
same probe under input ablations. This separates terminal content use from
condition-specific retraining, positional scaffold priors, and anchor artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
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
TRAIN_CONDITION = "attention_terminal"
CONDITIONS = (
    "attention_terminal",
    "attention_no_position",
    "attention_shuffled_position",
    "attention_gold_anchor",
    "random_terminal",
    "same_position_random_terminal",
    "position_only",
    "random_terminal_predicted_anchor",
    "random_terminal_gold_anchor_oracle",
)
TERMINAL_BY_CONDITION = {
    "attention_terminal": "attention",
    "attention_no_position": "attention",
    "attention_shuffled_position": "attention",
    "attention_gold_anchor": "attention",
    "random_terminal": "random",
    "same_position_random_terminal": "same_position_random",
    "position_only": "position_only",
    "random_terminal_predicted_anchor": "random",
    "random_terminal_gold_anchor_oracle": "random",
}
ANCHOR_MODE_BY_CONDITION = {
    "attention_gold_anchor": "gold",
    "random_terminal_predicted_anchor": "predicted",
    "random_terminal_gold_anchor_oracle": "gold",
}
POSITION_MODE_BY_CONDITION = {
    "attention_no_position": "zero",
    "attention_shuffled_position": "shuffle",
}


@dataclass(frozen=True)
class V2S3BConfig:
    model_name: str = "t5-small"
    max_train_samples: int = 768
    max_eval_samples: int = 192
    max_length: int = 128
    target_max_length: int = 96
    anchor_max_length: int = 48
    skeleton_batch_size: int = 16
    train_batch_size: int = 16
    eval_batch_size: int = 16
    anchor_epochs: int = 1
    reverse_epochs: int = 1
    learning_rate: float = 5e-4
    d_model: int = 512
    num_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    output_dir: str = "/kaggle/working/lace_v2_s3b"
    input_text_file: str | None = None
    use_hf_dataset: bool = True
    hf_dataset_name: str = "wikitext"
    hf_dataset_config: str = "wikitext-2-raw-v1"
    hf_dataset_split: str = "train"
    seed: int = 42
    keep_ratio: float = 0.25
    gate_tolerance: float = 0.02
    sample_output_count: int = 24
    min_text_words: int = 6
    train_condition: str = TRAIN_CONDITION
    conditions: tuple[str, ...] = CONDITIONS


def parse_conditions(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(items) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    if not items:
        raise ValueError("At least one condition is required.")
    return items


def parse_args() -> V2S3BConfig:
    parser = argparse.ArgumentParser(description="Run LACE V2 S3b probe calibration.")
    parser.add_argument("--model-name", default=V2S3BConfig.model_name)
    parser.add_argument("--max-train-samples", type=int, default=V2S3BConfig.max_train_samples)
    parser.add_argument("--max-eval-samples", type=int, default=V2S3BConfig.max_eval_samples)
    parser.add_argument("--max-length", type=int, default=V2S3BConfig.max_length)
    parser.add_argument("--target-max-length", type=int, default=V2S3BConfig.target_max_length)
    parser.add_argument("--anchor-max-length", type=int, default=V2S3BConfig.anchor_max_length)
    parser.add_argument("--skeleton-batch-size", type=int, default=V2S3BConfig.skeleton_batch_size)
    parser.add_argument("--train-batch-size", type=int, default=V2S3BConfig.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=V2S3BConfig.eval_batch_size)
    parser.add_argument("--anchor-epochs", type=int, default=V2S3BConfig.anchor_epochs)
    parser.add_argument("--reverse-epochs", type=int, default=V2S3BConfig.reverse_epochs)
    parser.add_argument("--learning-rate", type=float, default=V2S3BConfig.learning_rate)
    parser.add_argument("--d-model", type=int, default=V2S3BConfig.d_model)
    parser.add_argument("--num-heads", type=int, default=V2S3BConfig.num_heads)
    parser.add_argument("--encoder-layers", type=int, default=V2S3BConfig.encoder_layers)
    parser.add_argument("--decoder-layers", type=int, default=V2S3BConfig.decoder_layers)
    parser.add_argument("--dropout", type=float, default=V2S3BConfig.dropout)
    parser.add_argument("--max-grad-norm", type=float, default=V2S3BConfig.max_grad_norm)
    parser.add_argument("--output-dir", default=os.environ.get("LACE_OUTPUT_DIR", V2S3BConfig.output_dir))
    parser.add_argument("--input-text-file", default=os.environ.get("LACE_INPUT_TEXT_FILE"))
    parser.add_argument("--use-hf-dataset", action=argparse.BooleanOptionalAction, default=V2S3BConfig.use_hf_dataset)
    parser.add_argument("--hf-dataset-name", default=V2S3BConfig.hf_dataset_name)
    parser.add_argument("--hf-dataset-config", default=V2S3BConfig.hf_dataset_config)
    parser.add_argument("--hf-dataset-split", default=V2S3BConfig.hf_dataset_split)
    parser.add_argument("--seed", type=int, default=V2S3BConfig.seed)
    parser.add_argument("--keep-ratio", type=float, default=V2S3BConfig.keep_ratio)
    parser.add_argument("--gate-tolerance", type=float, default=V2S3BConfig.gate_tolerance)
    parser.add_argument("--sample-output-count", type=int, default=V2S3BConfig.sample_output_count)
    parser.add_argument("--min-text-words", type=int, default=V2S3BConfig.min_text_words)
    parser.add_argument("--train-condition", default=TRAIN_CONDITION)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    args = parser.parse_args()
    if args.keep_ratio <= 0 or args.keep_ratio > 1:
        raise ValueError("--keep-ratio must be in the interval (0, 1].")
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads.")
    if args.train_condition not in CONDITIONS:
        raise ValueError(f"Unknown train condition: {args.train_condition}")
    return V2S3BConfig(
        model_name=args.model_name,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_length=args.max_length,
        target_max_length=args.target_max_length,
        anchor_max_length=args.anchor_max_length,
        skeleton_batch_size=args.skeleton_batch_size,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        anchor_epochs=args.anchor_epochs,
        reverse_epochs=args.reverse_epochs,
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
        gate_tolerance=args.gate_tolerance,
        sample_output_count=args.sample_output_count,
        min_text_words=args.min_text_words,
        train_condition=args.train_condition,
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


def surface_entities(text: str) -> list[str]:
    candidates = re.findall(r"\b(?:[A-Z][A-Za-z0-9'-]*|\d+(?:\.\d+)?)\b", text)
    return [item.lower() for item in candidates if item.lower() not in STOPWORDS and len(item) > 1]


def load_texts(config: V2S3BConfig) -> tuple[list[str], str]:
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


def decode_ids(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def ids_with_eos(token_ids: list[int], eos_token_id: int, max_length: int) -> list[int]:
    kept = token_ids[: max(1, max_length - 1)]
    return kept + [eos_token_id]


def clean_generated_ids(tokenizer, token_ids: list[int]) -> list[int]:
    special = set(int(item) for item in tokenizer.all_special_ids)
    output = []
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id in special:
            if token_id == tokenizer.eos_token_id:
                break
            continue
        output.append(token_id)
    return output


def select_top_positions(scores: list[float], positions: list[int], keep_count: int) -> list[int]:
    ranked = sorted(positions, key=lambda index: (-float(scores[index]), index))
    return sorted(ranked[:keep_count])


def build_token_idf(encoded_rows: list[dict[str, Any]]) -> dict[int, float]:
    document_frequency: Counter[int] = Counter()
    for row in encoded_rows:
        token_ids = [int(row["input_ids"][position]) for position in row["valid_positions"]]
        document_frequency.update(set(token_ids))
    total_documents = max(1, len(encoded_rows))
    return {
        token_id: math.log((total_documents + 1) / (count + 1)) + 1.0
        for token_id, count in document_frequency.items()
    }


def wrong_index(sample_index: int, sample_count: int, seed: int) -> int:
    if sample_count <= 1:
        return sample_index
    return (sample_index + 1 + (seed % (sample_count - 1))) % sample_count


def same_position_ids_from_other(
    encoded_rows: list[dict[str, Any]],
    sample_index: int,
    positions: list[int],
    seed: int,
) -> list[int]:
    other = encoded_rows[wrong_index(sample_index, len(encoded_rows), seed)]
    other_valid = list(other["valid_positions"])
    other_position_set = set(other_valid)
    token_ids: list[int] = []
    for offset, position in enumerate(positions):
        if position in other_position_set:
            token_ids.append(int(other["input_ids"][position]))
        elif other_valid:
            token_ids.append(int(other["input_ids"][other_valid[offset % len(other_valid)]]))
    return token_ids


def build_terminals(config: V2S3BConfig, tokenizer, encoder, device: str, texts: list[str]) -> list[dict[str, Any]]:
    encoded_rows: list[dict[str, Any]] = []
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
                positions = active_positions(valid_mask[row].cpu())
                encoded_rows.append(
                    {
                        "sample_id": sample_id,
                        "input_ids": [int(item) for item in input_ids[row].tolist()],
                        "valid_positions": positions,
                        "attention_scores": [float(item) for item in scores[row].tolist()],
                    }
                )

    token_idf = build_token_idf(encoded_rows)
    payloads: list[dict[str, Any]] = []
    for row_index, row in enumerate(encoded_rows):
        positions = list(row["valid_positions"])
        input_ids = list(row["input_ids"])
        sample_id = int(row["sample_id"])
        if not positions:
            payloads.append(
                {
                    "sample_id": sample_id,
                    "attention_ids": [],
                    "attention_positions": [],
                    "attention_text": "",
                    "idf_ids": [],
                    "idf_positions": [],
                    "idf_text": "",
                    "random_ids": [],
                    "random_positions": [],
                    "random_text": "",
                    "same_position_random_ids": [],
                    "same_position_random_positions": [],
                    "same_position_random_text": "",
                    "position_only_positions": [],
                    "position_only_text": "",
                }
            )
            continue
        keep_count = target_keep_count(len(positions), config.keep_ratio)
        attention_positions = select_top_positions(row["attention_scores"], positions, keep_count)
        idf_scores = [float(token_idf.get(int(token_id), 0.0)) for token_id in input_ids]
        idf_positions = select_top_positions(idf_scores, positions, keep_count)
        rng = random.Random(config.seed + sample_id)
        random_positions = sorted(rng.sample(positions, k=keep_count)) if keep_count > 0 else []
        attention_ids = [int(input_ids[position]) for position in attention_positions]
        idf_ids = [int(input_ids[position]) for position in idf_positions]
        random_ids = [int(input_ids[position]) for position in random_positions]
        same_position_ids = same_position_ids_from_other(encoded_rows, row_index, attention_positions, config.seed)
        payloads.append(
            {
                "sample_id": sample_id,
                "attention_ids": attention_ids,
                "attention_positions": attention_positions,
                "attention_text": decode_ids(tokenizer, attention_ids),
                "idf_ids": idf_ids,
                "idf_positions": idf_positions,
                "idf_text": decode_ids(tokenizer, idf_ids),
                "random_ids": random_ids,
                "random_positions": random_positions,
                "random_text": decode_ids(tokenizer, random_ids),
                "same_position_random_ids": same_position_ids,
                "same_position_random_positions": attention_positions,
                "same_position_random_text": decode_ids(tokenizer, same_position_ids),
                "position_only_positions": attention_positions,
                "position_only_text": "",
            }
        )
    return payloads


class SequenceDataset:
    def __init__(self, examples: list[dict[str, Any]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def pad_list(values: list[int], length: int, pad_value: int) -> list[int]:
    return values[:length] + [pad_value] * max(0, length - len(values))


def make_loader(examples: list[dict[str, Any]], batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    return DataLoader(SequenceDataset(examples), batch_size=batch_size, shuffle=shuffle, collate_fn=lambda items: items)


def collate_examples(batch: list[dict[str, Any]], pad_token_id: int, max_position: int) -> dict[str, Any]:
    input_length = max(1, max(len(item["input_ids"]) for item in batch))
    target_length = max(2, max(len(item["target_ids"]) for item in batch))
    input_ids = []
    positions = []
    segments = []
    input_mask = []
    target_ids = []
    for item in batch:
        length = len(item["input_ids"])
        input_ids.append(pad_list(item["input_ids"], input_length, pad_token_id))
        positions.append(pad_list([min(max_position - 1, value) for value in item["positions"]], input_length, 0))
        segments.append(pad_list(item["segments"], input_length, 0))
        input_mask.append([False] * length + [True] * max(0, input_length - length))
        target_ids.append(pad_list(item["target_ids"], target_length, pad_token_id))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "positions": torch.tensor(positions, dtype=torch.long),
        "segments": torch.tensor(segments, dtype=torch.long),
        "input_key_padding_mask": torch.tensor(input_mask, dtype=torch.bool),
        "target_ids": torch.tensor(target_ids, dtype=torch.long),
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


class S3BSeq2SeqModel(nn.Module):
    def __init__(
        self,
        config: V2S3BConfig,
        vocab_size: int,
        pad_token_id: int,
        eos_token_id: int,
        initial_embedding: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.token_embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_token_id)
        if initial_embedding is not None and tuple(initial_embedding.shape) == tuple(self.token_embedding.weight.shape):
            self.token_embedding.weight.data.copy_(initial_embedding)
        self.segment_embedding = nn.Embedding(2, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.decoder_position = nn.Embedding(max(config.target_max_length, config.anchor_max_length) + 2, config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size)

    def encode(self, input_ids, positions, segments, key_padding_mask):
        hidden = self.token_embedding(input_ids)
        hidden = hidden + sinusoidal_encoding(positions, self.config.d_model).to(hidden.dtype)
        hidden = hidden + self.segment_embedding(segments.clamp(min=0, max=1))
        return self.encoder(hidden, src_key_padding_mask=key_padding_mask)

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
            batch["input_ids"],
            batch["positions"],
            batch["segments"],
            batch["input_key_padding_mask"],
        )
        logits = self.decode(decoder_input, memory, batch["input_key_padding_mask"])
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        return loss, logits

    @torch.no_grad()
    def generate(self, batch, max_length: int):
        memory = self.encode(
            batch["input_ids"],
            batch["positions"],
            batch["segments"],
            batch["input_key_padding_mask"],
        )
        generated = torch.full(
            (batch["input_ids"].shape[0], 1),
            self.pad_token_id,
            dtype=torch.long,
            device=batch["input_ids"].device,
        )
        for _step in range(max_length - 1):
            logits = self.decode(generated, memory, batch["input_key_padding_mask"])
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
    target_entities = list(dict.fromkeys(surface_entities(target)))[:8]
    predicted_set = set(predicted_words)
    predicted_bigrams = list(zip(predicted_words, predicted_words[1:]))
    repeated_bigram_count = max(0, len(predicted_bigrams) - len(set(predicted_bigrams)))
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
        "entity_recall": (
            sum(1 for item in target_entities if item in predicted_set) / len(target_entities)
            if target_entities
            else 0.0
        ),
        "repetition_rate": repeated_bigram_count / len(predicted_bigrams) if predicted_bigrams else 0.0,
        "nonempty": 1.0 if prediction.strip() else 0.0,
    }


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def terminal_key_for_condition(condition_name: str) -> str:
    return TERMINAL_BY_CONDITION[condition_name]


def anchor_mode_for_condition(condition_name: str) -> str | None:
    return ANCHOR_MODE_BY_CONDITION.get(condition_name)


def position_mode_for_condition(condition_name: str) -> str | None:
    return POSITION_MODE_BY_CONDITION.get(condition_name)


def apply_position_mode(condition_name: str, positions: list[int], sample_id: int, seed: int) -> list[int]:
    mode = position_mode_for_condition(condition_name)
    if mode == "zero":
        return [0] * len(positions)
    if mode == "shuffle":
        shuffled = list(positions)
        rng = random.Random(seed + int(sample_id) + 9973)
        rng.shuffle(shuffled)
        return shuffled
    return list(positions)


def make_base_examples(tokenizer, config: V2S3BConfig, texts: list[str], payloads: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
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
                "target_ids": target,
                "target_text": texts[index],
                "gold_anchor_ids": ids_with_eos(payload["attention_ids"], tokenizer.eos_token_id, config.anchor_max_length),
                "gold_anchor_text": payload["attention_text"],
                "attention_ids": payload["attention_ids"],
                "attention_positions": payload["attention_positions"],
                "attention_text": payload["attention_text"],
                "idf_ids": payload["idf_ids"],
                "idf_positions": payload["idf_positions"],
                "idf_text": payload["idf_text"],
                "random_ids": payload["random_ids"],
                "random_positions": payload["random_positions"],
                "random_text": payload["random_text"],
                "same_position_random_ids": payload["same_position_random_ids"],
                "same_position_random_positions": payload["same_position_random_positions"],
                "same_position_random_text": payload["same_position_random_text"],
                "position_only_ids": [tokenizer.pad_token_id] * len(payload["position_only_positions"]),
                "position_only_positions": payload["position_only_positions"],
                "position_only_text": payload["position_only_text"],
            }
        )
    return examples


def make_terminal_input(base: dict[str, Any], terminal_key: str, target_ids: list[int]) -> dict[str, Any]:
    terminal_ids = list(base[f"{terminal_key}_ids"])
    terminal_positions = list(base[f"{terminal_key}_positions"])
    return {
        "sample_id": base["sample_id"],
        "input_ids": terminal_ids,
        "positions": terminal_positions,
        "segments": [0] * len(terminal_ids),
        "target_ids": target_ids,
        "target_text": base.get("target_text", ""),
        "input_text": base[f"{terminal_key}_text"],
        "terminal_text": base[f"{terminal_key}_text"],
        "gold_anchor_text": base.get("gold_anchor_text", ""),
    }


def make_anchor_examples(base_examples: list[dict[str, Any]], terminal_key: str) -> list[dict[str, Any]]:
    return [make_terminal_input(base, terminal_key, list(base["gold_anchor_ids"])) for base in base_examples]


def make_reverse_examples(
    config: V2S3BConfig,
    tokenizer,
    base_examples: list[dict[str, Any]],
    condition_name: str,
    predicted_anchors: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    terminal_key = terminal_key_for_condition(condition_name)
    anchor_mode = anchor_mode_for_condition(condition_name)
    examples = []
    for base in base_examples:
        terminal_ids = list(base[f"{terminal_key}_ids"])
        terminal_positions = list(base[f"{terminal_key}_positions"])
        input_ids = list(terminal_ids)
        positions = apply_position_mode(condition_name, terminal_positions, base["sample_id"], config.seed)
        segments = [0] * len(input_ids)
        terminal_text = base[f"{terminal_key}_text"]
        anchor_text = ""
        anchor_label = ""
        if anchor_mode == "predicted":
            anchor = predicted_anchors.get(base["sample_id"], {"token_ids": [], "text": ""})
            anchor_ids = clean_generated_ids(tokenizer, list(anchor.get("token_ids", [])))[: max(0, len(base["gold_anchor_ids"]) - 1)]
            input_ids.extend(anchor_ids)
            positions.extend([0] * len(anchor_ids))
            segments.extend([1] * len(anchor_ids))
            anchor_text = str(anchor.get("text", ""))
            anchor_label = "predicted_anchor"
        elif anchor_mode == "gold":
            anchor_ids = list(base["gold_anchor_ids"][:-1])
            input_ids.extend(anchor_ids)
            positions.extend([0] * len(anchor_ids))
            segments.extend([1] * len(anchor_ids))
            anchor_text = str(base["gold_anchor_text"])
            anchor_label = "gold_anchor"
        input_text = terminal_text if not anchor_text else f"{terminal_text} || {anchor_label}: {anchor_text}"
        examples.append(
            {
                "sample_id": base["sample_id"],
                "input_ids": input_ids,
                "positions": positions,
                "segments": segments,
                "target_ids": list(base["target_ids"]),
                "target_text": base["target_text"],
                "input_text": input_text,
                "terminal_text": terminal_text,
                "predicted_anchor_text": anchor_text,
                "gold_anchor_text": base["gold_anchor_text"],
            }
        )
    return examples


def train_model(config, tokenizer, device, train_examples, initial_embedding, epochs: int):
    model = S3BSeq2SeqModel(
        config,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        initial_embedding=initial_embedding,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    train_loader = make_loader(train_examples, config.train_batch_size, shuffle=True)
    losses: list[float] = []
    model.train()
    for _epoch in range(epochs):
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


def evaluate_model(config, tokenizer, model, device, eval_examples, max_length: int):
    model.eval()
    eval_loader = make_loader(eval_examples, config.eval_batch_size, shuffle=False)
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_batch in eval_loader:
            batch = move_batch(collate_examples(raw_batch, tokenizer.pad_token_id, config.max_length), device)
            loss, _logits = model(batch)
            losses.append(float(loss.detach().cpu()))
            generated = model.generate(batch, max_length)
            predictions = tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for item, prediction, generated_ids in zip(raw_batch, predictions, generated.cpu().tolist()):
                metrics = lexical_metrics(prediction, item["target_text"], item["input_text"])
                rows.append(
                    {
                        "sample_id": item["sample_id"],
                        "input": item["input_text"],
                        "terminal": item.get("terminal_text", ""),
                        "predicted_anchor": item.get("predicted_anchor_text", ""),
                        "gold_anchor": item.get("gold_anchor_text", ""),
                        "target": item["target_text"],
                        "prediction": prediction,
                        "generated_token_ids": generated_ids,
                        "metrics": metrics,
                    }
                )
    metric_names = [
        "token_f1",
        "rouge_l_f1",
        "keyword_recall",
        "skeleton_coverage",
        "entity_recall",
        "repetition_rate",
        "nonempty",
    ]
    aggregate = {name: mean([row["metrics"][name] for row in rows]) for name in metric_names}
    aggregate["eval_loss"] = mean(losses)
    aggregate["eval_ppl"] = math.exp(min(20.0, aggregate["eval_loss"])) if math.isfinite(aggregate["eval_loss"]) else float("inf")
    return {"metrics": aggregate, "samples": rows}


def predict_anchors(config, tokenizer, model, device, examples: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[str, float]]:
    evaluation = evaluate_model(config, tokenizer, model, device, examples, config.anchor_max_length)
    predictions: dict[int, dict[str, Any]] = {}
    anchor_rows = []
    for row in evaluation["samples"]:
        token_ids = clean_generated_ids(tokenizer, row["generated_token_ids"])
        text = decode_ids(tokenizer, token_ids)
        predictions[row["sample_id"]] = {"token_ids": token_ids, "text": text}
        metrics = lexical_metrics(text, row["gold_anchor"], row["gold_anchor"])
        anchor_rows.append(metrics)
    metric_names = [
        "token_f1",
        "rouge_l_f1",
        "keyword_recall",
        "skeleton_coverage",
        "entity_recall",
        "repetition_rate",
        "nonempty",
    ]
    aggregate = {name: mean([row[name] for row in anchor_rows]) for name in metric_names}
    aggregate["eval_loss"] = evaluation["metrics"]["eval_loss"]
    aggregate["eval_ppl"] = evaluation["metrics"]["eval_ppl"]
    return predictions, aggregate


def train_anchor_predictor(config, tokenizer, device, terminal_key: str, train_bases, eval_bases, initial_embedding):
    train_examples = make_anchor_examples(train_bases, terminal_key)
    eval_examples = make_anchor_examples(eval_bases, terminal_key)
    model, losses = train_model(config, tokenizer, device, train_examples, initial_embedding, config.anchor_epochs)
    train_predictions, _train_metrics = predict_anchors(config, tokenizer, model, device, train_examples)
    eval_predictions, eval_metrics = predict_anchors(config, tokenizer, model, device, eval_examples)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "train_predictions": train_predictions,
        "eval_predictions": eval_predictions,
        "eval_metrics": eval_metrics,
        "training_losses": losses,
    }


def score_for_gate(metrics: dict[str, float]) -> float:
    overlap_score = metrics.get("token_f1", 0.0) + metrics.get("rouge_l_f1", 0.0)
    semantic_bonus = 0.2 * metrics.get("keyword_recall", 0.0) + 0.2 * metrics.get("entity_recall", 0.0)
    loss_bonus = 1.0 / (1.0 + max(0.0, metrics.get("eval_loss", 1e9)))
    repetition_penalty = 0.1 * metrics.get("repetition_rate", 0.0)
    return overlap_score + semantic_bonus + loss_bonus - repetition_penalty


def evaluate_gates(
    metrics: dict[str, dict[str, float]],
    tolerance: float,
    required_conditions: tuple[str, ...],
) -> dict[str, Any]:
    required = set(required_conditions)
    available = set(metrics)
    scores = {name: score_for_gate(item) for name, item in metrics.items()}
    best_name = max(scores, key=scores.get) if scores else ""
    attention_score = scores.get("attention_terminal")
    no_position_score = scores.get("attention_no_position")
    shuffled_position_score = scores.get("attention_shuffled_position")
    same_position_score = scores.get("same_position_random_terminal")
    position_only_score = scores.get("position_only")
    random_score = scores.get("random_terminal")
    predicted_anchor_score = scores.get("random_terminal_predicted_anchor")
    oracle_anchor_score = scores.get("random_terminal_gold_anchor_oracle")
    attention_gold_anchor_score = scores.get("attention_gold_anchor")
    losses_finite = all(math.isfinite(item.get("eval_loss", float("inf"))) for item in metrics.values())

    def beats(left: float | None, right: float | None, margin: float = tolerance) -> bool:
        return left is not None and right is not None and left > right + margin

    def delta(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return left - right

    gates = {
        "S3B-G-RUN": {"pass": required.issubset(available), "detail": sorted(available)},
        "S3B-G-LOSS-FINITE": {"pass": losses_finite, "detail": {name: item["eval_loss"] for name, item in metrics.items()}},
        "S3B-G-CONTENT-BEATS-SAME-POSITION": {
            "pass": beats(attention_score, same_position_score),
            "detail": {
                "attention_score": attention_score,
                "same_position_random_score": same_position_score,
                "delta": delta(attention_score, same_position_score),
                "tolerance": tolerance,
            },
        },
        "S3B-G-CONTENT-BEATS-POSITION-ONLY": {
            "pass": beats(attention_score, position_only_score),
            "detail": {
                "attention_score": attention_score,
                "position_only_score": position_only_score,
                "delta": delta(attention_score, position_only_score),
                "tolerance": tolerance,
            },
        },
        "S3B-G-CONTENT-BEATS-RANDOM": {
            "pass": beats(attention_score, random_score),
            "detail": {
                "attention_score": attention_score,
                "random_score": random_score,
                "delta": delta(attention_score, random_score),
                "tolerance": tolerance,
            },
        },
        "S3B-G-POSITION-ABLATION-DROP": {
            "pass": beats(attention_score, no_position_score),
            "detail": {
                "attention_score": attention_score,
                "attention_no_position_score": no_position_score,
                "delta": delta(attention_score, no_position_score),
                "tolerance": tolerance,
            },
        },
        "S3B-G-SHUFFLED-POSITION-DROP": {
            "pass": beats(attention_score, shuffled_position_score),
            "detail": {
                "attention_score": attention_score,
                "attention_shuffled_position_score": shuffled_position_score,
                "delta": delta(attention_score, shuffled_position_score),
                "tolerance": tolerance,
            },
        },
        "S3B-G-ANCHOR-SANITY": {
            "pass": oracle_anchor_score is not None
            and predicted_anchor_score is not None
            and oracle_anchor_score + tolerance >= predicted_anchor_score,
            "detail": {
                "random_gold_anchor_score": oracle_anchor_score,
                "random_predicted_anchor_score": predicted_anchor_score,
                "gold_minus_predicted": delta(oracle_anchor_score, predicted_anchor_score),
                "random_gold_lift": delta(oracle_anchor_score, random_score),
                "random_predicted_lift": delta(predicted_anchor_score, random_score),
                "tolerance": tolerance,
            },
        },
        "S3B-G-ATTENTION-GOLD-ANCHOR-LIFT": {
            "pass": attention_gold_anchor_score is not None
            and attention_score is not None
            and attention_gold_anchor_score >= attention_score - tolerance,
            "detail": {
                "attention_gold_anchor_score": attention_gold_anchor_score,
                "attention_score": attention_score,
                "attention_gold_lift": delta(attention_gold_anchor_score, attention_score),
                "tolerance": tolerance,
            },
        },
        "S3B-G-BEST-IDENTIFIED": {
            "pass": bool(best_name),
            "detail": {"best_condition": best_name, "best_score": scores.get(best_name, 0.0), "scores": scores},
        },
    }
    gates["overall_pass"] = bool(
        gates["S3B-G-RUN"]["pass"]
        and gates["S3B-G-LOSS-FINITE"]["pass"]
        and gates["S3B-G-CONTENT-BEATS-SAME-POSITION"]["pass"]
        and gates["S3B-G-CONTENT-BEATS-POSITION-ONLY"]["pass"]
        and gates["S3B-G-CONTENT-BEATS-RANDOM"]["pass"]
        and gates["S3B-G-POSITION-ABLATION-DROP"]["pass"]
        and gates["S3B-G-BEST-IDENTIFIED"]["pass"]
    )
    gates["diagnostic_ready"] = bool(gates["S3B-G-RUN"]["pass"] and gates["S3B-G-LOSS-FINITE"]["pass"])
    gates["s4_ready"] = False
    return gates


def render_summary(result: dict[str, Any]) -> str:
    run_info = result["run_info"]
    gates = result["gates"]
    metrics = result["condition_metrics"]
    anchor_metrics = result["anchor_metrics"]
    lines = [
        "# LACE V2 S3b Summary",
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
        f"- train condition: `{run_info['train_condition']}`",
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
            f"| `overall_pass` | `{str(gates.get('overall_pass')).lower()}` | Core S3b content-control signal. |",
            f"| `diagnostic_ready` | `{str(gates.get('diagnostic_ready')).lower()}` | Whether the diagnostic comparison is interpretable. |",
            f"| `s4_ready` | `{str(gates.get('s4_ready')).lower()}` | S3b is diagnostic, so this remains false by default. |",
            "",
            "## Condition Metrics",
            "",
            "| Condition | Loss | PPL | Token F1 | ROUGE-L | Keyword Recall | Entity Recall | Skeleton Coverage | Repetition | Nonempty | Score |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition_name in run_info["conditions"]:
        if condition_name not in metrics:
            continue
        item = metrics[condition_name]
        lines.append(
            "| `{}` | {:.4f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                condition_name,
                item["eval_loss"],
                item["eval_ppl"],
                item["token_f1"],
                item["rouge_l_f1"],
                item["keyword_recall"],
                item["entity_recall"],
                item["skeleton_coverage"],
                item["repetition_rate"],
                item["nonempty"],
                score_for_gate(item),
            )
        )
    lines.extend(
        [
            "",
            "## Anchor Predictor Metrics",
            "",
            "| Terminal | Loss | PPL | Anchor Token F1 | Anchor ROUGE-L | Nonempty |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for terminal_key in sorted(anchor_metrics):
        item = anchor_metrics[terminal_key]
        lines.append(
            "| `{}` | {:.4f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} |".format(
                terminal_key,
                item["eval_loss"],
                item["eval_ppl"],
                item["token_f1"],
                item["rouge_l_f1"],
                item["nonempty"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "S3b is a same-model input-ablation diagnostic, not an open-ended generation result.",
            "A positional ablation failure means the probe is still relying on scaffold/model priors too strongly.",
            "Gold-anchor oracle is an upper-bound control and should not be treated as a deployable baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config: V2S3BConfig) -> dict[str, Any]:
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
    payloads = build_terminals(config, tokenizer, skeleton_encoder, device, texts)
    del skeleton_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_indices = list(range(train_count))
    eval_indices = list(range(train_count, train_count + eval_count))
    train_bases = make_base_examples(tokenizer, config, texts, payloads, train_indices)
    eval_bases = make_base_examples(tokenizer, config, texts, payloads, eval_indices)

    anchor_runs = {
        terminal_key: train_anchor_predictor(config, tokenizer, device, terminal_key, train_bases, eval_bases, initial_embedding)
        for terminal_key in ("attention", "random")
    }
    anchor_metrics = {terminal_key: run_data["eval_metrics"] for terminal_key, run_data in anchor_runs.items()}

    condition_metrics: dict[str, dict[str, float]] = {}
    sample_rows: list[dict[str, Any]] = []
    train_terminal_key = terminal_key_for_condition(config.train_condition)
    train_anchor_predictions = anchor_runs.get(train_terminal_key, {}).get("train_predictions", {})
    train_examples = make_reverse_examples(config, tokenizer, train_bases, config.train_condition, train_anchor_predictions)
    model, reverse_training_losses = train_model(
        config,
        tokenizer,
        device,
        train_examples,
        initial_embedding,
        config.reverse_epochs,
    )
    for condition_name in config.conditions:
        terminal_key = terminal_key_for_condition(condition_name)
        eval_anchor_predictions = anchor_runs.get(terminal_key, {}).get("eval_predictions", {})
        eval_examples = make_reverse_examples(config, tokenizer, eval_bases, condition_name, eval_anchor_predictions)
        evaluation = evaluate_model(config, tokenizer, model, device, eval_examples, config.target_max_length)
        condition_metrics[condition_name] = evaluation["metrics"]
        for row in evaluation["samples"][: config.sample_output_count]:
            row["condition"] = condition_name
            sample_rows.append(row)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "run_info": {
            "phase": "v2_s3b",
            "experiment_name": "S3b-probe calibration",
            "config": asdict(config),
            "text_source": text_source,
            "train_samples": train_count,
            "eval_samples": eval_count,
            "model_name": config.model_name,
            "device": device,
            "keep_ratio": config.keep_ratio,
            "train_condition": config.train_condition,
            "conditions": config.conditions,
        },
        "condition_metrics": condition_metrics,
        "anchor_metrics": anchor_metrics,
        "training_losses": {config.train_condition: reverse_training_losses},
        "anchor_training_losses": {terminal_key: run_data["training_losses"] for terminal_key, run_data in anchor_runs.items()},
    }
    result["gates"] = evaluate_gates(condition_metrics, config.gate_tolerance, config.conditions)
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
