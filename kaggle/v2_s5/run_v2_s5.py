"""LACE V2 S5 semantic plan bridge.

S5 tests whether an explicit semantic plan between the skeleton state and
surface span realization can recover generated-span content/entity signal. The
first runner keeps the S4g pretrained text-to-text realizer but adds oracle,
heuristic-predicted, random, wrong-document, shuffled, no-plan, and position-only
plan conditions inside one phase.
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
SCHEDULES = (
    "oracle_plan_schedule",
    "predicted_plan_schedule",
    "no_plan_schedule",
    "random_plan_schedule",
    "same_position_random_plan_schedule",
    "wrong_document_plan_schedule",
    "position_only_plan_schedule",
    "shuffled_plan_schedule",
)
CONDITION_IDS = {name: index for index, name in enumerate(SCHEDULES)}
DEFAULT_RATIOS = (0.25, 0.50, 0.75, 1.00)
ORACLE_TARGET_SCHEDULE = "oracle_plan_schedule"

ROLE_CONTEXT = 0
ROLE_SPAN_MARKER = 1
ROLE_LEFT_ANCHOR = 2
ROLE_RIGHT_ANCHOR = 3

S4D_BASELINES = {
    "span_content_recall": 0.01716961498439126,
    "span_entity_recall": 0.004725979882067291,
    "final_content_recall": 0.33396680720899474,
    "final_entity_recall": 0.2988,
    "semantic_drift_proxy": 0.643269570022845,
    "repetition_rate": 0.04692930208528173,
}

S4E_BASELINES = {
    "span_content_recall": 0.0029483177245924383,
    "span_entity_recall": 0.0013007284079084287,
    "artifact_rate": 0.6906000693721818,
    "final_content_recall": 0.3463531332671958,
    "final_entity_recall": 0.29762524801587303,
    "semantic_drift_proxy": 0.6323082927624784,
    "repetition_rate": 0.03753022486204332,
    "rollout_score": 0.7569323546557519,
}

S4G_BASELINES = {
    "span_content_recall": 0.0,
    "span_entity_recall": 0.0,
    "artifact_rate": 0.99609375,
    "final_content_recall": 0.3432,
    "final_entity_recall": 0.2901,
    "semantic_drift_proxy": 0.6311,
    "repetition_rate": 0.0015,
    "rollout_score": 0.7314,
}


@dataclass(frozen=True)
class V2S5Config:
    model_name: str = "t5-small"
    max_train_samples: int = 768
    max_eval_samples: int = 192
    max_length: int = 128
    source_max_length: int = 192
    target_max_length: int = 32
    generation_max_new_tokens: int = 16
    max_span_length: int = 8
    skeleton_batch_size: int = 16
    train_batch_size: int = 8
    eval_batch_size: int = 8
    reverse_epochs: int = 1
    learning_rate: float = 5e-5
    use_amp: bool = False
    max_grad_norm: float = 1.0
    no_repeat_ngram_size: int = 3
    num_beams: int = 1
    max_train_examples_per_schedule: int = 3000
    max_eval_examples_per_schedule: int = 512
    rollout_eval_samples: int = 48
    output_dir: str = "/kaggle/working/lace_v2_s5"
    shared_condition_model: bool = True
    pretrained_decoder: bool = True
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


def parse_args() -> V2S5Config:
    parser = argparse.ArgumentParser(description="Run LACE V2 S5 semantic plan bridge.")
    parser.add_argument("--model-name", default=V2S5Config.model_name)
    parser.add_argument("--max-train-samples", type=int, default=V2S5Config.max_train_samples)
    parser.add_argument("--max-eval-samples", type=int, default=V2S5Config.max_eval_samples)
    parser.add_argument("--max-length", type=int, default=V2S5Config.max_length)
    parser.add_argument("--source-max-length", type=int, default=V2S5Config.source_max_length)
    parser.add_argument("--target-max-length", type=int, default=V2S5Config.target_max_length)
    parser.add_argument("--generation-max-new-tokens", type=int, default=V2S5Config.generation_max_new_tokens)
    parser.add_argument("--max-span-length", type=int, default=V2S5Config.max_span_length)
    parser.add_argument("--skeleton-batch-size", type=int, default=V2S5Config.skeleton_batch_size)
    parser.add_argument("--train-batch-size", type=int, default=V2S5Config.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=V2S5Config.eval_batch_size)
    parser.add_argument("--reverse-epochs", type=int, default=V2S5Config.reverse_epochs)
    parser.add_argument("--learning-rate", type=float, default=V2S5Config.learning_rate)
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=V2S5Config.use_amp)
    parser.add_argument("--max-grad-norm", type=float, default=V2S5Config.max_grad_norm)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=V2S5Config.no_repeat_ngram_size)
    parser.add_argument("--num-beams", type=int, default=V2S5Config.num_beams)
    parser.add_argument(
        "--max-train-examples-per-schedule",
        type=int,
        default=V2S5Config.max_train_examples_per_schedule,
    )
    parser.add_argument(
        "--max-eval-examples-per-schedule",
        type=int,
        default=V2S5Config.max_eval_examples_per_schedule,
    )
    parser.add_argument("--rollout-eval-samples", type=int, default=V2S5Config.rollout_eval_samples)
    parser.add_argument("--output-dir", default=os.environ.get("LACE_OUTPUT_DIR", V2S5Config.output_dir))
    parser.add_argument("--input-text-file", default=os.environ.get("LACE_INPUT_TEXT_FILE"))
    parser.add_argument("--use-hf-dataset", action=argparse.BooleanOptionalAction, default=V2S5Config.use_hf_dataset)
    parser.add_argument("--hf-dataset-name", default=V2S5Config.hf_dataset_name)
    parser.add_argument("--hf-dataset-config", default=V2S5Config.hf_dataset_config)
    parser.add_argument("--hf-dataset-split", default=V2S5Config.hf_dataset_split)
    parser.add_argument("--seed", type=int, default=V2S5Config.seed)
    parser.add_argument("--ratios", default=",".join(str(item) for item in DEFAULT_RATIOS))
    parser.add_argument("--gate-tolerance", type=float, default=V2S5Config.gate_tolerance)
    parser.add_argument("--sample-output-count", type=int, default=V2S5Config.sample_output_count)
    parser.add_argument("--min-text-words", type=int, default=V2S5Config.min_text_words)
    parser.add_argument("--schedules", default=",".join(SCHEDULES))
    args = parser.parse_args()
    if args.max_span_length <= 0:
        raise ValueError("--max-span-length must be positive.")
    if args.target_max_length <= 1:
        raise ValueError("--target-max-length must be larger than 1.")
    if args.max_train_examples_per_schedule <= 0:
        raise ValueError("--max-train-examples-per-schedule must be positive.")
    if args.max_eval_examples_per_schedule <= 0:
        raise ValueError("--max-eval-examples-per-schedule must be positive.")
    if args.rollout_eval_samples <= 0:
        raise ValueError("--rollout-eval-samples must be positive.")
    return V2S5Config(
        model_name=args.model_name,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_length=args.max_length,
        source_max_length=args.source_max_length,
        target_max_length=args.target_max_length,
        generation_max_new_tokens=args.generation_max_new_tokens,
        max_span_length=args.max_span_length,
        skeleton_batch_size=args.skeleton_batch_size,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        reverse_epochs=args.reverse_epochs,
        learning_rate=args.learning_rate,
        use_amp=args.use_amp,
        max_grad_norm=args.max_grad_norm,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        num_beams=args.num_beams,
        max_train_examples_per_schedule=args.max_train_examples_per_schedule,
        max_eval_examples_per_schedule=args.max_eval_examples_per_schedule,
        rollout_eval_samples=args.rollout_eval_samples,
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


def load_texts(config: V2S5Config) -> tuple[list[str], str]:
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


def ids_with_eos(token_ids: list[int], eos_token_id: int, max_length: int) -> list[int]:
    kept = token_ids[: max(1, max_length - 1)]
    return kept + [eos_token_id]


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


def build_schedule_payloads(config: V2S5Config, tokenizer, encoder, device: str, texts: list[str]) -> list[dict[str, Any]]:
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
        valid_token_ids = [int(input_ids[position]) for position in valid_positions]
        if not valid_token_ids:
            valid_token_ids = [int(tokenizer.pad_token_id)]
        states: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for ratio in config.ratios:
            attention_state = state_from_rank(
                tokenizer,
                input_ids,
                attention_rank,
                len(valid_positions),
                ratio,
            )
            random_state = state_from_rank(
                tokenizer,
                input_ids,
                random_rank,
                len(valid_positions),
                ratio,
            )
            for schedule_name in (
                "oracle_plan_schedule",
                "predicted_plan_schedule",
                "no_plan_schedule",
                "random_plan_schedule",
                "shuffled_plan_schedule",
            ):
                states[schedule_name][ratio] = {
                    "ids": list(attention_state["ids"]),
                    "positions": list(attention_state["positions"]),
                    "text": str(attention_state["text"]),
                    "ratio": ratio,
                }
            states["position_only_plan_schedule"][ratio] = {
                "ids": [tokenizer.pad_token_id] * len(attention_state["positions"]),
                "positions": list(attention_state["positions"]),
                "text": "",
                "ratio": ratio,
            }
            random_token_rng = random.Random(config.seed * 100_003 + sample_id * 97 + int(ratio * 10_000))
            same_position_random_ids = [int(random_token_rng.choice(valid_token_ids)) for _ in attention_state["positions"]]
            states["same_position_random_plan_schedule"][ratio] = {
                "ids": same_position_random_ids,
                "positions": list(attention_state["positions"]),
                "text": decode_ids(tokenizer, same_position_random_ids),
                "ratio": ratio,
            }
            # Keep a random terminal state available for random-plan sampling.
            states["_random_reference"][ratio] = random_state
        payloads.append({"sample_id": sample_id, "states": {key: dict(value) for key, value in states.items()}})
    if payloads:
        for index, payload in enumerate(payloads):
            wrong_payload = payloads[(index + 1) % len(payloads)]
            wrong_states: dict[float, dict[str, Any]] = {}
            for ratio in config.ratios:
                current_state = payload["states"]["oracle_plan_schedule"][ratio]
                wrong_state = wrong_payload["states"]["oracle_plan_schedule"][ratio]
                needed = len(current_state["positions"])
                wrong_ids = list(wrong_state["ids"]) or [int(tokenizer.pad_token_id)]
                fitted_ids = [int(wrong_ids[item % len(wrong_ids)]) for item in range(needed)]
                wrong_states[ratio] = {
                    "ids": fitted_ids,
                    "positions": list(current_state["positions"]),
                    "text": decode_ids(tokenizer, fitted_ids),
                    "ratio": ratio,
                }
            payload["states"]["wrong_document_plan_schedule"] = wrong_states
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
    target_length = max(2, max(len(item["target_ids"]) for item in batch))
    input_ids = []
    positions = []
    roles = []
    input_mask = []
    target_ids = []
    timesteps = []
    condition_ids = []
    gap_lengths = []
    left_anchor_distances = []
    right_anchor_distances = []
    for item in batch:
        length = len(item["input_ids"])
        input_ids.append(pad_list(item["input_ids"], input_length, pad_token_id))
        positions.append(pad_list([min(max_position - 1, value) for value in item["positions"]], input_length, 0))
        roles.append(pad_list([int(value) for value in item.get("roles", [ROLE_CONTEXT] * length)], input_length, ROLE_CONTEXT))
        input_mask.append([False] * length + [True] * max(0, input_length - length))
        target_ids.append(pad_list(item["target_ids"], target_length, pad_token_id))
        timesteps.append(int(item["transition_index"]))
        condition_ids.append(int(item.get("condition_id", CONDITION_IDS.get(item.get("schedule", ORACLE_TARGET_SCHEDULE), 0))))
        gap_lengths.append(min(max_position, int(item.get("span_length", len(item.get("delta_positions", []))))))
        left_anchor_distances.append(min(max_position, int(item.get("left_anchor_distance", max_position))))
        right_anchor_distances.append(min(max_position, int(item.get("right_anchor_distance", max_position))))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "positions": torch.tensor(positions, dtype=torch.long),
        "roles": torch.tensor(roles, dtype=torch.long),
        "input_key_padding_mask": torch.tensor(input_mask, dtype=torch.bool),
        "target_ids": torch.tensor(target_ids, dtype=torch.long),
        "timesteps": torch.tensor(timesteps, dtype=torch.long),
        "condition_ids": torch.tensor(condition_ids, dtype=torch.long),
        "gap_lengths": torch.tensor(gap_lengths, dtype=torch.long),
        "left_anchor_distances": torch.tensor(left_anchor_distances, dtype=torch.long),
        "right_anchor_distances": torch.tensor(right_anchor_distances, dtype=torch.long),
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


class S5SharedSpanReverseModel(nn.Module):
    def __init__(
        self,
        config: V2S5Config,
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
        self.timestep_embedding = nn.Embedding(max(1, len(config.ratios) - 1), config.d_model)
        self.role_embedding = nn.Embedding(4, config.d_model)
        self.condition_embedding = nn.Embedding(len(SCHEDULES), config.d_model)
        self.gap_length_embedding = nn.Embedding(config.max_length + 1, config.d_model)
        self.left_distance_embedding = nn.Embedding(config.max_length + 1, config.d_model)
        self.right_distance_embedding = nn.Embedding(config.max_length + 1, config.d_model)
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
        self.decoder_position = nn.Embedding(config.target_max_length + 2, config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size)

    def encode(
        self,
        input_ids,
        positions,
        roles,
        timesteps,
        condition_ids,
        gap_lengths,
        left_anchor_distances,
        right_anchor_distances,
        key_padding_mask,
    ):
        hidden = self.token_embedding(input_ids)
        hidden = hidden + sinusoidal_encoding(positions, self.config.d_model).to(hidden.dtype)
        hidden = hidden + self.role_embedding(roles.clamp(min=0, max=3))
        hidden = hidden + self.timestep_embedding(timesteps).unsqueeze(1)
        hidden = hidden + self.condition_embedding(condition_ids.clamp(min=0, max=len(SCHEDULES) - 1)).unsqueeze(1)
        hidden = hidden + self.gap_length_embedding(gap_lengths.clamp(min=0, max=self.config.max_length)).unsqueeze(1)
        hidden = hidden + self.left_distance_embedding(
            left_anchor_distances.clamp(min=0, max=self.config.max_length)
        ).unsqueeze(1)
        hidden = hidden + self.right_distance_embedding(
            right_anchor_distances.clamp(min=0, max=self.config.max_length)
        ).unsqueeze(1)
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
            batch["roles"],
            batch["timesteps"],
            batch["condition_ids"],
            batch["gap_lengths"],
            batch["left_anchor_distances"],
            batch["right_anchor_distances"],
            batch["input_key_padding_mask"],
        )
        logits = self.decode(decoder_input, memory, batch["input_key_padding_mask"])
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids.reshape(-1), ignore_index=self.pad_token_id)
        return loss, logits

    def _block_repeated_ngrams(self, logits: torch.Tensor, generated: torch.Tensor, ngram_size: int) -> torch.Tensor:
        if ngram_size <= 0 or generated.shape[1] + 1 < ngram_size:
            return logits
        logits = logits.clone()
        for row in range(generated.shape[0]):
            tokens = [int(item) for item in generated[row].tolist()]
            prefix = tuple(tokens[-(ngram_size - 1) :]) if ngram_size > 1 else tuple()
            banned: set[int] = set()
            for index in range(0, len(tokens) - ngram_size + 1):
                ngram = tuple(tokens[index : index + ngram_size])
                if ngram[:-1] == prefix:
                    banned.add(ngram[-1])
            if banned:
                logits[row, list(banned)] = -float("inf")
        return logits

    @torch.no_grad()
    def generate(self, batch, max_length: int):
        memory = self.encode(
            batch["input_ids"],
            batch["positions"],
            batch["roles"],
            batch["timesteps"],
            batch["condition_ids"],
            batch["gap_lengths"],
            batch["left_anchor_distances"],
            batch["right_anchor_distances"],
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
            next_logits = self._block_repeated_ngrams(logits[:, -1], generated, self.config.no_repeat_ngram_size)
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if bool((next_token == self.eos_token_id).all()):
                break
        return generated[:, 1:]


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


def lexical_metrics(prediction: str, target: str, context_text: str, original_text: str) -> dict[str, float]:
    predicted_words = word_tokens(prediction)
    target_words = word_tokens(target)
    predicted_content_words = content_words(prediction)
    target_content = dedupe(content_words(target))[:12]
    context_content = dedupe(content_words(context_text))[:16]
    original_content = dedupe(content_words(original_text))[:16]
    predicted_counter = Counter(predicted_words)
    target_counter = Counter(target_words)
    predicted_content_counter = Counter(predicted_content_words)
    target_content_counter = Counter(target_content)
    content_overlap = sum((predicted_content_counter & target_content_counter).values())
    overlap = sum((predicted_counter & target_counter).values())
    lcs = lcs_length(predicted_words, target_words)
    predicted_set = set(predicted_words)
    predicted_content_set = set(predicted_content_words)
    target_entities = dedupe(surface_entities(target))[:8]
    target_entity_counter = Counter(target_entities)
    predicted_entity_counter = Counter(surface_entities(prediction))
    entity_overlap = sum((predicted_entity_counter & target_entity_counter).values())
    content_word_rate = len(predicted_content_words) / max(1, len(predicted_words))
    function_token_rate = (
        sum(1 for item in predicted_words if item in STOPWORDS) / max(1, len(predicted_words)) if predicted_words else 0.0
    )
    artifact_rate = 1.0 if prediction.strip() and content_word_rate < 0.20 else 0.0
    return {
        "delta_token_f1": f1_from_counts(overlap, len(predicted_words), len(target_words)),
        "delta_rouge_l_f1": f1_from_counts(lcs, len(predicted_words), len(target_words)),
        "delta_content_f1": f1_from_counts(content_overlap, len(predicted_content_words), len(target_content)),
        "delta_content_recall": (
            sum(1 for item in target_content if item in predicted_content_set) / len(target_content)
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
        "entity_f1": f1_from_counts(entity_overlap, len(predicted_entity_counter), len(target_entity_counter)),
        "content_word_rate": content_word_rate,
        "function_token_rate": function_token_rate,
        "artifact_rate": artifact_rate,
        "repetition_rate": repetition_rate(predicted_words),
        "nonempty": 1.0 if prediction.strip() else 0.0,
        "target_word_count": float(len(target_words)),
    }


def teacher_forced_metrics(predicted_ids: torch.Tensor, target_ids: torch.Tensor, pad_token_id: int, eos_token_id: int) -> dict[str, float]:
    content_mask = (target_ids != pad_token_id) & (target_ids != eos_token_id)
    content_total = int(content_mask.sum().item())
    if content_total <= 0:
        return {"tf_delta_accuracy": 0.0}
    correct = int(((predicted_ids == target_ids) & content_mask).sum().item())
    return {"tf_delta_accuracy": correct / content_total}


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


def target_schedule_for(schedule_name: str) -> str:
    return ORACLE_TARGET_SCHEDULE


def pairs_from_state(state: dict[str, Any]) -> list[tuple[int, int]]:
    return [(int(position), int(token_id)) for position, token_id in zip(state["positions"], state["ids"])]


def semantic_plan_terms(text: str, limit: int = 8) -> list[str]:
    terms = dedupe(content_words(text) + surface_entities(text))
    return terms[:limit]


def render_plan(terms: list[str]) -> str:
    cleaned = [item for item in dedupe([term.strip() for term in terms]) if item]
    return " ; ".join(cleaned) if cleaned else "none"


def semantic_plan_from_text(text: str, limit: int = 8) -> str:
    return render_plan(semantic_plan_terms(text, limit=limit))


def predicted_plan_from_context(context_text: str, left_text: str, right_text: str, limit: int = 8) -> str:
    """Non-parametric stage-2 predictor: use content near the gap as the plan.

    This is intentionally simple. S5 first asks whether a semantic-plan bridge is
    useful at all; a later runner can replace this heuristic with a learned plan
    predictor if the oracle condition is promising.
    """
    terms = semantic_plan_terms(" ".join([left_text, right_text, context_text]), limit=limit)
    return render_plan(terms)


def shuffled_plan(oracle_plan: str, seed: int) -> str:
    terms = [item.strip() for item in oracle_plan.split(";") if item.strip() and item.strip() != "none"]
    rng = random.Random(seed)
    rng.shuffle(terms)
    return render_plan(terms)


def plan_overlap_metrics(plan_text: str, oracle_plan_text: str) -> dict[str, float]:
    predicted = [item.strip().lower() for item in plan_text.split(";") if item.strip() and item.strip() != "none"]
    oracle = [item.strip().lower() for item in oracle_plan_text.split(";") if item.strip() and item.strip() != "none"]
    if not oracle:
        empty_match = 1.0 if not predicted else 0.0
        return {
            "plan_precision": empty_match,
            "plan_recall": empty_match,
            "plan_f1": empty_match,
            "plan_nonempty": 1.0 if predicted else 0.0,
            "plan_applicable": 0.0,
        }
    predicted_counter = Counter(predicted)
    oracle_counter = Counter(oracle)
    overlap = sum((predicted_counter & oracle_counter).values())
    return {
        "plan_precision": overlap / len(predicted) if predicted else 0.0,
        "plan_recall": overlap / len(oracle) if oracle else 0.0,
        "plan_f1": f1_from_counts(overlap, len(predicted), len(oracle)),
        "plan_nonempty": 1.0 if predicted else 0.0,
        "plan_applicable": 1.0,
    }


def split_contiguous_spans(delta_pairs: list[tuple[int, int]], max_span_length: int) -> list[list[tuple[int, int]]]:
    spans: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for position, token_id in sorted(delta_pairs, key=lambda item: item[0]):
        if current and (position != current[-1][0] + 1 or len(current) >= max_span_length):
            spans.append(current)
            current = []
        current.append((int(position), int(token_id)))
    if current:
        spans.append(current)
    return spans


def nearest_anchor_pairs(
    context_pairs: list[tuple[int, int]],
    span_start: int,
    span_end: int,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    left_candidates = [pair for pair in context_pairs if pair[0] < span_start]
    right_candidates = [pair for pair in context_pairs if pair[0] > span_end]
    left_anchor = max(left_candidates, key=lambda item: item[0]) if left_candidates else None
    right_anchor = min(right_candidates, key=lambda item: item[0]) if right_candidates else None
    return left_anchor, right_anchor


def span_encoder_fields(
    tokenizer,
    schedule_name: str,
    input_state: dict[str, Any],
    span_pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    context_pairs = pairs_from_state(input_state)
    span_start = int(span_pairs[0][0])
    span_end = int(span_pairs[-1][0])
    left_anchor, right_anchor = nearest_anchor_pairs(context_pairs, span_start, span_end)
    encoder_items: list[tuple[int, int, int]] = [
        (position, token_id, ROLE_CONTEXT) for position, token_id in sorted(context_pairs, key=lambda item: item[0])
    ]
    if left_anchor is not None:
        encoder_items.append((left_anchor[0], left_anchor[1], ROLE_LEFT_ANCHOR))
    if right_anchor is not None:
        encoder_items.append((right_anchor[0], right_anchor[1], ROLE_RIGHT_ANCHOR))
    encoder_items.extend(
        (position, int(tokenizer.pad_token_id), ROLE_SPAN_MARKER) for position, _token_id in span_pairs
    )
    if not encoder_items:
        encoder_items = [(0, int(tokenizer.pad_token_id), ROLE_SPAN_MARKER)]
    return {
        "input_ids": [token_id for _position, token_id, _role in encoder_items],
        "positions": [position for position, _token_id, _role in encoder_items],
        "roles": [role for _position, _token_id, role in encoder_items],
        "context_text": decode_ids(
            tokenizer,
            [token_id for _position, token_id in context_pairs if int(token_id) != int(tokenizer.pad_token_id)],
        ),
        "left_anchor_position": None if left_anchor is None else int(left_anchor[0]),
        "right_anchor_position": None if right_anchor is None else int(right_anchor[0]),
        "span_length": len(span_pairs),
        "left_anchor_distance": 999_999 if left_anchor is None else int(span_start - left_anchor[0]),
        "right_anchor_distance": 999_999 if right_anchor is None else int(right_anchor[0] - span_end),
    }


def span_text_from_state(tokenizer, state: dict[str, Any], span_positions: list[int]) -> str:
    position_to_id = {int(position): int(token_id) for position, token_id in zip(state["positions"], state["ids"])}
    token_ids = [position_to_id[position] for position in span_positions if position in position_to_id]
    if not token_ids:
        token_ids = [int(token_id) for token_id in list(state.get("ids", []))[: len(span_positions)]]
    return decode_ids(tokenizer, token_ids)


def plan_for_condition(
    config: V2S5Config,
    tokenizer,
    base: dict[str, Any],
    schedule_name: str,
    to_ratio: float,
    span_positions: list[int],
    target_text: str,
    context_text: str,
    left_anchor_position: int | None,
    right_anchor_position: int | None,
    transition_index: int,
    span_index: int,
) -> tuple[str, str]:
    oracle_plan = semantic_plan_from_text(target_text)
    if schedule_name == "oracle_plan_schedule":
        return oracle_plan, oracle_plan
    if schedule_name == "predicted_plan_schedule":
        left_text = "" if left_anchor_position is None else context_text
        right_text = "" if right_anchor_position is None else context_text
        return predicted_plan_from_context(context_text, left_text, right_text), oracle_plan
    if schedule_name == "no_plan_schedule":
        return "none", oracle_plan
    if schedule_name == "random_plan_schedule":
        random_text = str(base["states"].get("_random_reference", {}).get(to_ratio, {}).get("text", base["original_text"]))
        return semantic_plan_from_text(random_text), oracle_plan
    if schedule_name == "same_position_random_plan_schedule":
        random_span_text = span_text_from_state(
            tokenizer,
            base["states"]["same_position_random_plan_schedule"][to_ratio],
            span_positions,
        )
        return semantic_plan_from_text(random_span_text), oracle_plan
    if schedule_name == "wrong_document_plan_schedule":
        wrong_span_text = span_text_from_state(
            tokenizer,
            base["states"]["wrong_document_plan_schedule"][to_ratio],
            span_positions,
        )
        return semantic_plan_from_text(wrong_span_text), oracle_plan
    if schedule_name == "position_only_plan_schedule":
        return "none", oracle_plan
    if schedule_name == "shuffled_plan_schedule":
        seed = config.seed * 70_001 + int(base["sample_id"]) * 257 + transition_index * 31 + span_index
        return shuffled_plan(oracle_plan, seed), oracle_plan
    raise ValueError(f"Unsupported S5 plan condition: {schedule_name}")


def make_transition_examples(
    config: V2S5Config,
    tokenizer,
    base_examples: list[dict[str, Any]],
    schedule_name: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    ratios = list(config.ratios)
    for base in base_examples:
        for transition_index, (from_ratio, to_ratio) in enumerate(zip(ratios[:-1], ratios[1:])):
            input_state = base["states"][schedule_name][from_ratio]
            target_schedule = target_schedule_for(schedule_name)
            target_state = base["states"][target_schedule][to_ratio]
            current_positions = set(input_state["positions"])
            target_position_to_id = {
                int(position): int(token_id) for position, token_id in zip(target_state["positions"], target_state["ids"])
            }
            delta_positions = [int(position) for position in target_state["positions"] if int(position) not in current_positions]
            delta_pairs = [(position, target_position_to_id[position]) for position in delta_positions]
            if not delta_pairs:
                continue
            for span_index, span_pairs in enumerate(split_contiguous_spans(delta_pairs, config.max_span_length)):
                encoder_fields = span_encoder_fields(tokenizer, schedule_name, input_state, span_pairs)
                span_positions = [position for position, _token_id in span_pairs]
                span_ids = [token_id for _position, token_id in span_pairs]
                target_ids = ids_with_eos(span_ids, tokenizer.eos_token_id, config.target_max_length)
                target_text = decode_ids(tokenizer, span_ids)
                plan_text, oracle_plan_text = plan_for_condition(
                    config,
                    tokenizer,
                    base,
                    schedule_name,
                    to_ratio,
                    span_positions,
                    target_text,
                    encoder_fields["context_text"],
                    encoder_fields["left_anchor_position"],
                    encoder_fields["right_anchor_position"],
                    transition_index,
                    span_index,
                )
                examples.append(
                    {
                        "sample_id": base["sample_id"],
                        "schedule": schedule_name,
                        "condition_id": CONDITION_IDS[schedule_name],
                        "transition_index": transition_index,
                        "transition": f"{from_ratio:.2f}->{to_ratio:.2f}",
                        "from_ratio": from_ratio,
                        "to_ratio": to_ratio,
                        "span_index": span_index,
                        "span_start": int(span_positions[0]),
                        "span_end": int(span_positions[-1]),
                        "span_length": encoder_fields["span_length"],
                        "left_anchor_position": encoder_fields["left_anchor_position"],
                        "right_anchor_position": encoder_fields["right_anchor_position"],
                        "left_anchor_distance": encoder_fields["left_anchor_distance"],
                        "right_anchor_distance": encoder_fields["right_anchor_distance"],
                        "input_ids": encoder_fields["input_ids"],
                        "positions": encoder_fields["positions"],
                        "roles": encoder_fields["roles"],
                        "input_text": str(input_state["text"]),
                        "context_text": encoder_fields["context_text"],
                        "semantic_plan": plan_text,
                        "oracle_plan": oracle_plan_text,
                        "delta_positions": span_positions,
                        "target_ids": target_ids,
                        "target_text": target_text,
                        "target_state_text": str(target_state["text"]),
                        "original_text": base["original_text"],
                    }
                )
    return examples


def format_position_list(values: list[int], limit: int = 24) -> str:
    kept = [str(int(item)) for item in values[:limit]]
    suffix = "" if len(values) <= limit else " ..."
    return ",".join(kept) + suffix


def source_text_for_example(item: dict[str, Any]) -> str:
    context_text = normalize_text_line(str(item.get("context_text", "")))
    if not context_text:
        context_text = "<empty>"
    left = item.get("left_anchor_position")
    right = item.get("right_anchor_position")
    left_text = "none" if left is None else str(left)
    right_text = "none" if right is None else str(right)
    return " | ".join(
        [
            "lace reverse span expansion",
            f"condition: {item['schedule']}",
            f"transition: {item['transition']}",
            f"span positions: {format_position_list(item['delta_positions'])}",
            f"span length: {item.get('span_length')}",
            f"left anchor position: {left_text}",
            f"right anchor position: {right_text}",
            f"left distance: {item.get('left_anchor_distance')}",
            f"right distance: {item.get('right_anchor_distance')}",
            f"current skeleton: {context_text}",
            f"semantic plan: {item.get('semantic_plan', 'none')}",
            "realize missing span from plan:",
        ]
    )


def sample_examples(examples: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if len(examples) <= limit:
        return list(examples)
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    return [examples[index] for index in sorted(indices[:limit])]


def collate_seq2seq_examples(batch: list[dict[str, Any]], tokenizer, config: V2S5Config) -> dict[str, Any]:
    sources = [source_text_for_example(item) for item in batch]
    targets = [str(item["target_text"]) for item in batch]
    encoded = tokenizer(
        sources,
        padding=True,
        truncation=True,
        max_length=config.source_max_length,
        return_tensors="pt",
    )
    target_encoded = tokenizer(
        text_target=targets,
        padding=True,
        truncation=True,
        max_length=config.target_max_length,
        return_tensors="pt",
    )
    labels = target_encoded["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    encoded["labels"] = labels
    encoded["items"] = batch
    encoded["target_input_ids"] = target_encoded["input_ids"]
    return encoded


def move_seq2seq_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def teacher_forced_seq2seq_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels != -100
    total = int(mask.sum().item())
    if total <= 0:
        return 0.0
    predicted = logits.argmax(dim=-1)
    correct = int(((predicted == labels) & mask).sum().item())
    return correct / total


def generated_text_to_delta_ids(tokenizer, prediction: str, expected_count: int) -> list[int]:
    token_ids = tokenizer.encode(prediction, add_special_tokens=False)
    special_ids = {int(item) for item in tokenizer.all_special_ids}
    cleaned = [int(item) for item in token_ids if int(item) not in special_ids]
    if len(cleaned) < expected_count:
        cleaned.extend([int(tokenizer.pad_token_id)] * (expected_count - len(cleaned)))
    return cleaned[:expected_count]


def train_model(config, tokenizer, device, train_examples):
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name).to(device)
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    amp_enabled = bool(config.use_amp and device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader = make_loader(train_examples, config.train_batch_size, shuffle=True)
    losses: list[float] = []
    model.train()
    for epoch in range(config.reverse_epochs):
        for batch_index, raw_batch in enumerate(train_loader):
            batch = move_seq2seq_batch(collate_seq2seq_examples(raw_batch, tokenizer, config), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss for pretrained decoder at epoch={epoch} batch={batch_index}: "
                    f"{float(loss.detach().cpu())}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
    return model, losses


def aggregate_rows(rows: list[dict[str, Any]], losses: list[float]) -> dict[str, float]:
    metric_names = [
        "tf_delta_accuracy",
        "delta_token_f1",
        "delta_rouge_l_f1",
        "delta_content_f1",
        "delta_content_recall",
        "context_copy_rate",
        "original_content_recall",
        "entity_recall",
        "entity_f1",
        "content_word_rate",
        "function_token_rate",
        "artifact_rate",
        "repetition_rate",
        "nonempty",
        "target_word_count",
        "plan_precision",
        "plan_recall",
        "plan_f1",
        "plan_nonempty",
        "plan_applicable",
    ]
    aggregate = {name: mean([row["metrics"][name] for row in rows]) for name in metric_names}
    aggregate["eval_loss"] = mean(losses)
    aggregate["eval_ppl"] = math.exp(min(20.0, aggregate["eval_loss"])) if math.isfinite(aggregate["eval_loss"]) else float("inf")
    return aggregate


def evaluate_model(config, tokenizer, model, device, eval_examples):
    model.eval()
    eval_loader = make_loader(eval_examples, config.eval_batch_size, shuffle=False)
    losses: list[float] = []
    rows: list[dict[str, Any]] = []
    transition_losses: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad():
        for raw_batch in eval_loader:
            batch = move_seq2seq_batch(collate_seq2seq_examples(raw_batch, tokenizer, config), device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss_value = float(outputs.loss.detach().cpu())
            losses.append(loss_value)
            for item in raw_batch:
                transition_losses[item["transition"]].append(loss_value)
            tf_accuracy = teacher_forced_seq2seq_accuracy(outputs.logits.detach().cpu(), batch["labels"].detach().cpu())
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=config.generation_max_new_tokens,
                num_beams=config.num_beams,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
            )
            predictions = tokenizer.batch_decode(
                generated.detach().cpu(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for row_index, (item, prediction, generated_ids) in enumerate(zip(raw_batch, predictions, generated.cpu().tolist())):
                metrics = lexical_metrics(prediction, item["target_text"], item["context_text"], item["original_text"])
                metrics["tf_delta_accuracy"] = tf_accuracy
                metrics.update(plan_overlap_metrics(item.get("semantic_plan", "none"), item.get("oracle_plan", "none")))
                rows.append(
                    {
                        "sample_id": item["sample_id"],
                        "schedule": item["schedule"],
                        "condition_id": item.get("condition_id"),
                        "transition": item["transition"],
                        "span_index": item.get("span_index"),
                        "span_start": item.get("span_start"),
                        "span_end": item.get("span_end"),
                        "span_length": item.get("span_length"),
                        "left_anchor_position": item.get("left_anchor_position"),
                        "right_anchor_position": item.get("right_anchor_position"),
                        "left_anchor_distance": item.get("left_anchor_distance"),
                        "right_anchor_distance": item.get("right_anchor_distance"),
                        "input": item["input_text"],
                        "context": item["context_text"],
                        "semantic_plan": item.get("semantic_plan", "none"),
                        "oracle_plan": item.get("oracle_plan", "none"),
                        "delta_positions": item["delta_positions"],
                        "target": item["target_text"],
                        "target_state": item["target_state_text"],
                        "original": item["original_text"],
                        "prediction": prediction,
                        "generated_token_ids": generated_ids,
                        "source_text": source_text_for_example(item),
                        "metrics": metrics,
                    }
                )
    aggregate = aggregate_rows(rows, losses)
    by_transition = {}
    for transition in sorted(set(row["transition"] for row in rows)):
        transition_rows = [row for row in rows if row["transition"] == transition]
        by_transition[transition] = aggregate_rows(transition_rows, transition_losses[transition])
    return {"metrics": aggregate, "by_transition": by_transition, "samples": rows}


def state_pairs_from_state(state: dict[str, Any]) -> dict[int, int]:
    return {int(position): int(token_id) for position, token_id in zip(state["positions"], state["ids"])}


def decode_state_pairs(tokenizer, pairs: dict[int, int]) -> str:
    return decode_ids(tokenizer, [token_id for _position, token_id in sorted(pairs.items())])


def clean_generated_delta_ids(tokenizer, generated_ids: list[int], expected_count: int) -> list[int]:
    special_ids = {int(item) for item in tokenizer.all_special_ids}
    cleaned: list[int] = []
    for raw_token_id in generated_ids:
        token_id = int(raw_token_id)
        if token_id == int(tokenizer.eos_token_id):
            break
        if token_id in special_ids:
            continue
        cleaned.append(token_id)
        if len(cleaned) >= expected_count:
            break
    if len(cleaned) < expected_count:
        cleaned.extend([int(tokenizer.pad_token_id)] * (expected_count - len(cleaned)))
    return cleaned[:expected_count]


def rollout_state_metrics(prediction: str, target: str, original_text: str) -> dict[str, float]:
    predicted_words = word_tokens(prediction)
    target_words = word_tokens(target)
    target_content = dedupe(content_words(target))[:16]
    original_content = dedupe(content_words(original_text))[:16]
    target_entities = dedupe(surface_entities(target) or surface_entities(original_text))[:8]
    predicted_counter = Counter(predicted_words)
    target_counter = Counter(target_words)
    overlap = sum((predicted_counter & target_counter).values())
    lcs = lcs_length(predicted_words, target_words)
    predicted_set = set(predicted_words)
    final_content_recall = (
        sum(1 for item in target_content if item in predicted_set) / len(target_content) if target_content else 0.0
    )
    final_original_content_recall = (
        sum(1 for item in original_content if item in predicted_set) / len(original_content) if original_content else 0.0
    )
    final_entity_recall = (
        sum(1 for item in target_entities if item in predicted_set) / len(target_entities) if target_entities else 0.0
    )
    final_token_f1 = f1_from_counts(overlap, len(predicted_words), len(target_words))
    final_rouge_l = f1_from_counts(lcs, len(predicted_words), len(target_words))
    nonempty = 1.0 if prediction.strip() else 0.0
    semantic_proxy_score = (
        0.35 * final_content_recall
        + 0.25 * final_original_content_recall
        + 0.20 * final_entity_recall
        + 0.15 * final_rouge_l
        + 0.05 * nonempty
    )
    return {
        "final_token_f1": final_token_f1,
        "final_rouge_l": final_rouge_l,
        "final_content_recall": final_content_recall,
        "final_original_content_recall": final_original_content_recall,
        "final_entity_recall": final_entity_recall,
        "repetition_rate": repetition_rate(predicted_words),
        "semantic_proxy_score": semantic_proxy_score,
        "semantic_drift_proxy": max(0.0, 1.0 - semantic_proxy_score),
        "nonempty": nonempty,
        "length_ratio": len(predicted_words) / max(1, len(target_words)),
    }


def score_for_teacher_forced(metrics: dict[str, float]) -> float:
    overlap = (
        1.50 * metrics.get("tf_delta_accuracy", 0.0)
        + 0.50 * metrics.get("delta_token_f1", 0.0)
        + 0.50 * metrics.get("delta_rouge_l_f1", 0.0)
    )
    semantic = (
        0.45 * metrics.get("delta_content_recall", 0.0)
        + 0.25 * metrics.get("original_content_recall", 0.0)
        + 0.20 * metrics.get("entity_recall", 0.0)
        + 0.10 * metrics.get("nonempty", 0.0)
    )
    loss_bonus = 1.0 / (1.0 + max(0.0, metrics.get("eval_loss", 1e9)))
    copy_penalty = 0.1 * metrics.get("context_copy_rate", 0.0)
    repetition_penalty = 0.1 * metrics.get("repetition_rate", 0.0)
    artifact_penalty = 0.05 * metrics.get("artifact_rate", 0.0)
    return overlap + semantic + loss_bonus - repetition_penalty - copy_penalty - artifact_penalty


def score_for_rollout(metrics: dict[str, float]) -> float:
    overlap = 0.75 * metrics.get("final_token_f1", 0.0) + 0.50 * metrics.get("final_rouge_l", 0.0)
    semantic = (
        0.45 * metrics.get("final_content_recall", 0.0)
        + 0.25 * metrics.get("final_original_content_recall", 0.0)
        + 0.20 * metrics.get("final_entity_recall", 0.0)
        + 0.10 * metrics.get("nonempty", 0.0)
    )
    drift_penalty = 0.15 * metrics.get("semantic_drift_proxy", 1.0)
    repetition_penalty = 0.10 * metrics.get("repetition_rate", 0.0)
    degradation_penalty = 0.10 * metrics.get("stepwise_degradation", 0.0)
    return overlap + semantic - drift_penalty - repetition_penalty - degradation_penalty


def aggregate_rollout_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = [
        "final_token_f1",
        "final_rouge_l",
        "final_content_recall",
        "final_original_content_recall",
        "final_entity_recall",
        "repetition_rate",
        "semantic_proxy_score",
        "semantic_drift_proxy",
        "stepwise_degradation",
        "nonempty",
        "length_ratio",
    ]
    return {name: mean([row["metrics"].get(name, 0.0) for row in rows]) for name in metric_names}


def make_rollout_items(
    config: V2S5Config,
    tokenizer,
    base: dict[str, Any],
    schedule_name: str,
    rollout_index: int,
    transition_index: int,
    from_ratio: float,
    to_ratio: float,
    current_pairs: dict[int, int],
) -> list[dict[str, Any]]:
    target_schedule = target_schedule_for(schedule_name)
    target_state = base["states"][target_schedule][to_ratio]
    target_position_to_id = {
        int(position): int(token_id) for position, token_id in zip(target_state["positions"], target_state["ids"])
    }
    current_positions = set(current_pairs)
    delta_positions = [int(position) for position in target_state["positions"] if int(position) not in current_positions]
    if not delta_positions:
        return []
    delta_pairs = [(position, target_position_to_id[position]) for position in delta_positions]
    input_state = {
        "ids": [int(token_id) for _position, token_id in sorted(current_pairs.items())],
        "positions": [int(position) for position, _token_id in sorted(current_pairs.items())],
        "text": decode_state_pairs(tokenizer, current_pairs),
        "ratio": from_ratio,
    }
    items: list[dict[str, Any]] = []
    for span_index, span_pairs in enumerate(split_contiguous_spans(delta_pairs, config.max_span_length)):
        encoder_fields = span_encoder_fields(tokenizer, schedule_name, input_state, span_pairs)
        span_positions = [position for position, _token_id in span_pairs]
        span_ids = [token_id for _position, token_id in span_pairs]
        target_text = decode_ids(tokenizer, span_ids)
        plan_text, oracle_plan_text = plan_for_condition(
            config,
            tokenizer,
            base,
            schedule_name,
            to_ratio,
            span_positions,
            target_text,
            encoder_fields["context_text"],
            encoder_fields["left_anchor_position"],
            encoder_fields["right_anchor_position"],
            transition_index,
            span_index,
        )
        items.append(
            {
                "rollout_index": rollout_index,
                "sample_id": base["sample_id"],
                "schedule": schedule_name,
                "condition_id": CONDITION_IDS[schedule_name],
                "transition_index": transition_index,
                "transition": f"{from_ratio:.2f}->{to_ratio:.2f}",
                "from_ratio": from_ratio,
                "to_ratio": to_ratio,
                "span_index": span_index,
                "span_start": int(span_positions[0]),
                "span_end": int(span_positions[-1]),
                "span_length": encoder_fields["span_length"],
                "left_anchor_position": encoder_fields["left_anchor_position"],
                "right_anchor_position": encoder_fields["right_anchor_position"],
                "left_anchor_distance": encoder_fields["left_anchor_distance"],
                "right_anchor_distance": encoder_fields["right_anchor_distance"],
                "input_ids": encoder_fields["input_ids"],
                "positions": encoder_fields["positions"],
                "roles": encoder_fields["roles"],
                "input_text": input_state["text"],
                "context_text": encoder_fields["context_text"],
                "semantic_plan": plan_text,
                "oracle_plan": oracle_plan_text,
                "delta_positions": span_positions,
                "target_ids": ids_with_eos(span_ids, tokenizer.eos_token_id, config.target_max_length),
                "target_text": target_text,
                "target_state_text": str(target_state["text"]),
                "original_text": base["original_text"],
            }
        )
    return items


def evaluate_rollout(config, tokenizer, model, device, eval_bases: list[dict[str, Any]], schedule_name: str):
    model.eval()
    ratios = list(config.ratios)
    start_ratio = ratios[0]
    target_schedule = target_schedule_for(schedule_name)
    current_pairs_by_index = [state_pairs_from_state(base["states"][schedule_name][start_ratio]) for base in eval_bases]
    step_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    step_scores_by_index: dict[int, list[float]] = defaultdict(list)

    with torch.no_grad():
        for transition_index, (from_ratio, to_ratio) in enumerate(zip(ratios[:-1], ratios[1:])):
            rollout_items = []
            for rollout_index, base in enumerate(eval_bases):
                rollout_items.extend(
                    make_rollout_items(
                        config,
                        tokenizer,
                        base,
                        schedule_name,
                        rollout_index,
                        transition_index,
                        from_ratio,
                        to_ratio,
                        current_pairs_by_index[rollout_index],
                    )
                )
            rollout_loader = make_loader(rollout_items, config.eval_batch_size, shuffle=False)
            for raw_batch in rollout_loader:
                batch = move_seq2seq_batch(collate_seq2seq_examples(raw_batch, tokenizer, config), device)
                generated = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=config.generation_max_new_tokens,
                    num_beams=config.num_beams,
                    no_repeat_ngram_size=config.no_repeat_ngram_size,
                ).detach().cpu()
                predictions = tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for item, prediction in zip(raw_batch, predictions):
                    cleaned_ids = generated_text_to_delta_ids(tokenizer, prediction, len(item["delta_positions"]))
                    current_pairs = current_pairs_by_index[int(item["rollout_index"])]
                    for position, token_id in zip(item["delta_positions"], cleaned_ids):
                        current_pairs[int(position)] = int(token_id)

            for rollout_index, base in enumerate(eval_bases):
                target_state = base["states"][target_schedule][to_ratio]
                prediction = decode_state_pairs(tokenizer, current_pairs_by_index[rollout_index])
                metrics = rollout_state_metrics(prediction, str(target_state["text"]), base["original_text"])
                step_score = score_for_rollout(metrics)
                step_scores_by_index[rollout_index].append(step_score)
                step_rows.append(
                    {
                        "sample_id": base["sample_id"],
                        "schedule": schedule_name,
                        "transition": f"{from_ratio:.2f}->{to_ratio:.2f}",
                        "prediction_state": prediction,
                        "target_state": str(target_state["text"]),
                        "original": base["original_text"],
                        "metrics": metrics,
                    }
                )

    final_ratio = ratios[-1]
    for rollout_index, base in enumerate(eval_bases):
        target_state = base["states"][target_schedule][final_ratio]
        prediction = decode_state_pairs(tokenizer, current_pairs_by_index[rollout_index])
        metrics = rollout_state_metrics(prediction, str(target_state["text"]), base["original_text"])
        step_scores = step_scores_by_index.get(rollout_index, [])
        drops = [max(0.0, left - right) for left, right in zip(step_scores[:-1], step_scores[1:])]
        metrics["stepwise_degradation"] = mean(drops)
        final_rows.append(
            {
                "sample_id": base["sample_id"],
                "schedule": schedule_name,
                "start_state": str(base["states"][schedule_name][start_ratio]["text"]),
                "prediction": prediction,
                "target_state": str(target_state["text"]),
                "original": base["original_text"],
                "step_scores": step_scores,
                "metrics": metrics,
            }
        )

    by_transition = {}
    for transition in sorted(set(row["transition"] for row in step_rows)):
        transition_rows = [row for row in step_rows if row["transition"] == transition]
        by_transition[transition] = aggregate_rollout_rows(transition_rows)
    return {"metrics": aggregate_rollout_rows(final_rows), "by_transition": by_transition, "samples": final_rows}


def evaluate_gates(
    teacher_metrics: dict[str, dict[str, float]],
    rollout_metrics: dict[str, dict[str, float]],
    tolerance: float,
    required_schedules: tuple[str, ...],
) -> dict[str, Any]:
    required = set(required_schedules)
    available = set(teacher_metrics) & set(rollout_metrics)
    losses_finite = all(math.isfinite(item.get("eval_loss", float("inf"))) for item in teacher_metrics.values())
    scores = {name: score_for_rollout(item) for name, item in rollout_metrics.items()}
    best_name = max(scores, key=scores.get) if scores else ""
    oracle_rollout = rollout_metrics.get("oracle_plan_schedule", {})
    predicted_rollout = rollout_metrics.get("predicted_plan_schedule", {})
    no_plan_rollout = rollout_metrics.get("no_plan_schedule", {})
    random_rollout = rollout_metrics.get("random_plan_schedule", {})
    same_position_rollout = rollout_metrics.get("same_position_random_plan_schedule", {})
    wrong_doc_rollout = rollout_metrics.get("wrong_document_plan_schedule", {})
    position_rollout = rollout_metrics.get("position_only_plan_schedule", {})
    shuffled_rollout = rollout_metrics.get("shuffled_plan_schedule", {})
    oracle_score = scores.get("oracle_plan_schedule")
    predicted_score = scores.get("predicted_plan_schedule")
    no_plan_score = scores.get("no_plan_schedule")
    random_score = scores.get("random_plan_schedule")
    same_position_score = scores.get("same_position_random_plan_schedule")
    wrong_doc_score = scores.get("wrong_document_plan_schedule")
    position_score = scores.get("position_only_plan_schedule")
    shuffled_score = scores.get("shuffled_plan_schedule")
    oracle_teacher = teacher_metrics.get("oracle_plan_schedule", {})
    predicted_teacher = teacher_metrics.get("predicted_plan_schedule", {})
    no_plan_teacher = teacher_metrics.get("no_plan_schedule", {})
    random_teacher = teacher_metrics.get("random_plan_schedule", {})
    wrong_doc_teacher = teacher_metrics.get("wrong_document_plan_schedule", {})

    def delta(left: float | None, right: float | None) -> float | None:
        if left is None or right is None:
            return None
        return left - right

    def beats(left: float | None, right: float | None, margin: float = tolerance) -> bool:
        return left is not None and right is not None and left > right + margin

    gates = {
        "S5-G-RUN": {"pass": required.issubset(available), "detail": sorted(available)},
        "S5-G-LOSS-FINITE": {
            "pass": losses_finite,
            "detail": {name: item["eval_loss"] for name, item in teacher_metrics.items()},
        },
        "S5-G-SHARED-REALIZER": {
            "pass": len(required) > 1,
            "detail": {"shared_condition_model": True, "pretrained_decoder": True, "condition_ids": CONDITION_IDS},
        },
        "S5-G-PRETRAINED-DECODER": {
            "pass": True,
            "detail": {"model_name": "t5-small", "decoder": "AutoModelForSeq2SeqLM"},
        },
        "S5-G-ORACLE-BEATS-NO-PLAN": {
            "pass": beats(oracle_score, no_plan_score),
            "detail": {
                "oracle_score": oracle_score,
                "no_plan_score": no_plan_score,
                "delta": delta(oracle_score, no_plan_score),
                "tolerance": tolerance,
            },
        },
        "S5-G-ORACLE-BEATS-RANDOM-PLAN": {
            "pass": beats(oracle_score, random_score),
            "detail": {
                "oracle_score": oracle_score,
                "random_plan_score": random_score,
                "delta": delta(oracle_score, random_score),
                "tolerance": tolerance,
            },
        },
        "S5-G-ORACLE-BEATS-SAME-POSITION-RANDOM-PLAN": {
            "pass": beats(oracle_score, same_position_score),
            "detail": {
                "oracle_score": oracle_score,
                "same_position_random_score": same_position_score,
                "delta": delta(oracle_score, same_position_score),
                "tolerance": tolerance,
            },
        },
        "S5-G-WRONG-DOC-PLAN-DROPS": {
            "pass": beats(oracle_score, wrong_doc_score),
            "detail": {
                "oracle_score": oracle_score,
                "wrong_document_plan_score": wrong_doc_score,
                "delta": delta(oracle_score, wrong_doc_score),
                "tolerance": tolerance,
            },
        },
        "S5-G-ORACLE-BEATS-POSITION-ONLY": {
            "pass": beats(oracle_score, position_score),
            "detail": {
                "oracle_score": oracle_score,
                "position_only_score": position_score,
                "delta": delta(oracle_score, position_score),
                "tolerance": tolerance,
            },
        },
        "S5-G-ORACLE-BEATS-SHUFFLED-PLAN": {
            "pass": beats(oracle_score, shuffled_score),
            "detail": {
                "oracle_score": oracle_score,
                "shuffled_plan_score": shuffled_score,
                "delta": delta(oracle_score, shuffled_score),
                "tolerance": tolerance,
            },
        },
        "S5-G-ORACLE-SPAN-CONTENT-GAIN-VS-S4G": {
            "pass": oracle_teacher.get("delta_content_recall", 0.0) > S4G_BASELINES["span_content_recall"] + 0.01,
            "detail": {
                "oracle_span_content_recall": oracle_teacher.get("delta_content_recall"),
                "no_plan_span_content_recall": no_plan_teacher.get("delta_content_recall"),
                "s4g_baseline_span_content_recall": S4G_BASELINES["span_content_recall"],
                "delta_vs_no_plan": delta(oracle_teacher.get("delta_content_recall"), no_plan_teacher.get("delta_content_recall")),
            },
        },
        "S5-G-ORACLE-SPAN-ENTITY-GAIN-VS-S4G": {
            "pass": oracle_teacher.get("entity_recall", 0.0) > S4G_BASELINES["span_entity_recall"] + 0.005,
            "detail": {
                "oracle_span_entity_recall": oracle_teacher.get("entity_recall"),
                "no_plan_span_entity_recall": no_plan_teacher.get("entity_recall"),
                "s4g_baseline_span_entity_recall": S4G_BASELINES["span_entity_recall"],
                "delta_vs_no_plan": delta(oracle_teacher.get("entity_recall"), no_plan_teacher.get("entity_recall")),
            },
        },
        "S5-G-ARTIFACT-LOWER-VS-S4G": {
            "pass": oracle_teacher.get("artifact_rate", 1.0) <= S4G_BASELINES["artifact_rate"] - 0.10,
            "detail": {
                "oracle_artifact_rate": oracle_teacher.get("artifact_rate"),
                "no_plan_artifact_rate": no_plan_teacher.get("artifact_rate"),
                "random_plan_artifact_rate": random_teacher.get("artifact_rate"),
                "wrong_document_artifact_rate": wrong_doc_teacher.get("artifact_rate"),
                "s4g_artifact_rate": S4G_BASELINES["artifact_rate"],
            },
        },
        "S5-G-PLAN-PREDICTOR-ABOVE-RANDOM": {
            "pass": predicted_teacher.get("plan_recall", 0.0) > random_teacher.get("plan_recall", 0.0) + tolerance,
            "detail": {
                "predicted_plan_recall": predicted_teacher.get("plan_recall"),
                "random_plan_recall": random_teacher.get("plan_recall"),
                "oracle_plan_recall": oracle_teacher.get("plan_recall"),
                "tolerance": tolerance,
            },
        },
        "S5-G-PREDICTED-BEATS-NO-PLAN": {
            "pass": beats(predicted_score, no_plan_score, margin=0.0),
            "detail": {
                "predicted_score": predicted_score,
                "no_plan_score": no_plan_score,
                "delta": delta(predicted_score, no_plan_score),
            },
        },
        "S5-G-ROLLOUT-NONREGRESSION-VS-S4G": {
            "pass": oracle_score is not None and oracle_score >= S4G_BASELINES["rollout_score"] - 0.08,
            "detail": {
                "oracle_score": oracle_score,
                "predicted_score": predicted_score,
                "s4g_rollout_score": S4G_BASELINES["rollout_score"],
                "oracle_final_content_recall": oracle_rollout.get("final_content_recall"),
                "predicted_final_content_recall": predicted_rollout.get("final_content_recall"),
                "no_plan_final_content_recall": no_plan_rollout.get("final_content_recall"),
            },
        },
        "S5-G-BEST-IDENTIFIED": {
            "pass": bool(best_name),
            "detail": {"best_schedule": best_name, "best_score": scores.get(best_name, 0.0), "scores": scores},
        },
    }
    gates["stage_1_oracle_plan_pass"] = bool(
        gates["S5-G-ORACLE-BEATS-NO-PLAN"]["pass"]
        and gates["S5-G-ORACLE-BEATS-RANDOM-PLAN"]["pass"]
        and gates["S5-G-ORACLE-SPAN-CONTENT-GAIN-VS-S4G"]["pass"]
        and gates["S5-G-ARTIFACT-LOWER-VS-S4G"]["pass"]
    )
    gates["stage_2_plan_prediction_pass"] = bool(gates["S5-G-PLAN-PREDICTOR-ABOVE-RANDOM"]["pass"])
    gates["stage_3_predicted_plan_rollout_pass"] = bool(gates["S5-G-PREDICTED-BEATS-NO-PLAN"]["pass"])
    gates["overall_pass"] = bool(
        gates["S5-G-RUN"]["pass"]
        and gates["S5-G-LOSS-FINITE"]["pass"]
        and gates["S5-G-SHARED-REALIZER"]["pass"]
        and gates["S5-G-PRETRAINED-DECODER"]["pass"]
        and gates["stage_1_oracle_plan_pass"]
        and gates["stage_2_plan_prediction_pass"]
        and gates["stage_3_predicted_plan_rollout_pass"]
        and gates["S5-G-ROLLOUT-NONREGRESSION-VS-S4G"]["pass"]
    )
    gates["process_ready"] = bool(gates["S5-G-RUN"]["pass"] and gates["S5-G-LOSS-FINITE"]["pass"])
    gates["structure_review_needed"] = not gates["overall_pass"]
    gates["s6_ready"] = gates["overall_pass"]
    return gates


def render_summary(result: dict[str, Any]) -> str:
    run_info = result["run_info"]
    gates = result["gates"]
    teacher_metrics = result["schedule_metrics"]
    teacher_by_transition = result["transition_metrics"]
    rollout_metrics = result["rollout_metrics"]
    rollout_by_transition = result["rollout_step_metrics"]
    lines = [
        "# LACE V2 S5 Summary",
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
        f"- shared condition model: `{run_info.get('shared_condition_model')}`",
        f"- condition ids: `{run_info.get('condition_ids')}`",
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
            f"| `overall_pass` | `{str(gates.get('overall_pass')).lower()}` | Multi-step rollout signal. |",
            f"| `process_ready` | `{str(gates.get('process_ready')).lower()}` | Whether the schedule comparison ran cleanly. |",
            f"| `stage_1_oracle_plan_pass` | `{str(gates.get('stage_1_oracle_plan_pass')).lower()}` | Whether oracle semantic plans improve span realization. |",
            f"| `stage_2_plan_prediction_pass` | `{str(gates.get('stage_2_plan_prediction_pass')).lower()}` | Whether the heuristic anchor plan beats random plan recall. |",
            f"| `stage_3_predicted_plan_rollout_pass` | `{str(gates.get('stage_3_predicted_plan_rollout_pass')).lower()}` | Whether predicted plans improve rollout over no-plan. |",
            f"| `structure_review_needed` | `{str(gates.get('structure_review_needed')).lower()}` | Whether plan bridge changes need another iteration. |",
            f"| `s6_ready` | `{str(gates.get('s6_ready')).lower()}` | Whether open-ended generation should be attempted next. |",
            "",
            "## Teacher-Forced Step Metrics",
            "",
            "| Schedule | Loss | PPL | Plan Recall | Plan F1 | TF Delta Acc | Delta F1 | ROUGE-L | Content Recall | Content F1 | Content Rate | Entity Recall | Entity F1 | Artifact | Repetition | Score |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for schedule_name in run_info["schedules"]:
        item = teacher_metrics[schedule_name]
        lines.append(
            "| `{}` | {:.4f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                schedule_name,
                item["eval_loss"],
                item["eval_ppl"],
                item["plan_recall"],
                item["plan_f1"],
                item["tf_delta_accuracy"],
                item["delta_token_f1"],
                item["delta_rouge_l_f1"],
                item["delta_content_recall"],
                item["delta_content_f1"],
                item["content_word_rate"],
                item["entity_recall"],
                item["entity_f1"],
                item["artifact_rate"],
                item["repetition_rate"],
                score_for_teacher_forced(item),
            )
        )
    lines.extend(
        [
            "",
            "## Rollout Final Metrics",
            "",
            "| Schedule | Final F1 | ROUGE-L | Content | Original Content | Entity | Repetition | Drift Proxy | Step Degradation | Length Ratio | Rollout Score |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for schedule_name in run_info["schedules"]:
        item = rollout_metrics[schedule_name]
        lines.append(
            "| `{}` | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                schedule_name,
                item["final_token_f1"],
                item["final_rouge_l"],
                item["final_content_recall"],
                item["final_original_content_recall"],
                item["final_entity_recall"],
                item["repetition_rate"],
                item["semantic_drift_proxy"],
                item["stepwise_degradation"],
                item["length_ratio"],
                score_for_rollout(item),
            )
        )
    lines.extend(["", "## Teacher-Forced Transition Metrics", ""])
    for schedule_name in run_info["schedules"]:
        lines.extend(
            [
                f"### `{schedule_name}`",
                "",
                "| Transition | Loss | Plan Recall | Plan F1 | TF Delta Acc | Delta F1 | ROUGE-L | Content Recall | Content F1 | Content Rate | Entity | Artifact | Repetition | Score |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for transition, item in teacher_by_transition[schedule_name].items():
            lines.append(
                "| `{}` | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    transition,
                    item["eval_loss"],
                    item["plan_recall"],
                    item["plan_f1"],
                    item["tf_delta_accuracy"],
                    item["delta_token_f1"],
                    item["delta_rouge_l_f1"],
                    item["delta_content_recall"],
                    item["delta_content_f1"],
                    item["content_word_rate"],
                    item["entity_recall"],
                    item["artifact_rate"],
                    item["repetition_rate"],
                    score_for_teacher_forced(item),
                )
            )
        lines.append("")
    lines.extend(["", "## Rollout Step Metrics", ""])
    for schedule_name in run_info["schedules"]:
        lines.extend(
            [
                f"### `{schedule_name}`",
                "",
                "| Transition | State F1 | ROUGE-L | Content | Original Content | Entity | Repetition | Drift Proxy | Score |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for transition, item in rollout_by_transition[schedule_name].items():
            lines.append(
                "| `{}` | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    transition,
                    item["final_token_f1"],
                    item["final_rouge_l"],
                    item["final_content_recall"],
                    item["final_original_content_recall"],
                    item["final_entity_recall"],
                    item["repetition_rate"],
                    item["semantic_drift_proxy"],
                    score_for_rollout(item),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation Guardrail",
            "",
            "S5 is a semantic-plan bridge experiment, not open-ended language generation.",
            "The key comparison is whether an explicit content plan improves generated span content without letting random, shuffled, wrong-document, or position-only controls win.",
            "The predicted-plan condition is intentionally heuristic in this first runner; it is a stage-2 diagnostic, not a final learned planner.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config: V2S5Config) -> dict[str, Any]:
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
    rollout_metrics: dict[str, dict[str, float]] = {}
    rollout_step_metrics: dict[str, dict[str, dict[str, float]]] = {}
    training_losses: dict[str, list[float]] = {}
    sample_rows: list[dict[str, Any]] = []
    rollout_sample_rows: list[dict[str, Any]] = []
    train_examples_by_schedule: dict[str, list[dict[str, Any]]] = {}
    eval_examples_by_schedule: dict[str, list[dict[str, Any]]] = {}
    combined_train_examples: list[dict[str, Any]] = []
    for schedule_name in config.schedules:
        train_examples = make_transition_examples(config, tokenizer, train_bases, schedule_name)
        eval_examples = make_transition_examples(config, tokenizer, eval_bases, schedule_name)
        train_examples = sample_examples(
            train_examples,
            config.max_train_examples_per_schedule,
            config.seed * 10_003 + CONDITION_IDS[schedule_name],
        )
        eval_examples = sample_examples(
            eval_examples,
            config.max_eval_examples_per_schedule,
            config.seed * 20_003 + CONDITION_IDS[schedule_name],
        )
        train_examples_by_schedule[schedule_name] = train_examples
        eval_examples_by_schedule[schedule_name] = eval_examples
        combined_train_examples.extend(train_examples)

    if not combined_train_examples:
        raise RuntimeError("No shared-condition train examples were produced.")

    rng.shuffle(combined_train_examples)
    rollout_eval_count = min(config.rollout_eval_samples, len(eval_bases))
    rollout_eval_bases = eval_bases[:rollout_eval_count]
    model, losses = train_model(config, tokenizer, device, combined_train_examples)
    training_losses["pretrained_decoder_shared_condition_model"] = losses
    for schedule_name in config.schedules:
        evaluation = evaluate_model(config, tokenizer, model, device, eval_examples_by_schedule[schedule_name])
        rollout_evaluation = evaluate_rollout(config, tokenizer, model, device, rollout_eval_bases, schedule_name)
        schedule_metrics[schedule_name] = evaluation["metrics"]
        transition_metrics[schedule_name] = evaluation["by_transition"]
        rollout_metrics[schedule_name] = rollout_evaluation["metrics"]
        rollout_step_metrics[schedule_name] = rollout_evaluation["by_transition"]
        for row in evaluation["samples"][: config.sample_output_count]:
            sample_rows.append(row)
        for row in rollout_evaluation["samples"][: config.sample_output_count]:
            rollout_sample_rows.append(row)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "run_info": {
            "phase": "v2_s5",
            "experiment_name": "S5-semantic plan bridge",
            "stages": (
                "stage_1_oracle_plan",
                "stage_2_plan_prediction",
                "stage_3_predicted_plan_rollout",
            ),
            "config": asdict(config),
            "text_source": text_source,
            "train_samples": train_count,
            "eval_samples": eval_count,
            "model_name": config.model_name,
            "device": device,
            "ratios": config.ratios,
            "schedules": config.schedules,
            "shared_condition_model": config.shared_condition_model,
            "pretrained_decoder": config.pretrained_decoder,
            "condition_ids": CONDITION_IDS,
            "training_example_counts": {name: len(items) for name, items in train_examples_by_schedule.items()},
            "eval_example_counts": {name: len(items) for name, items in eval_examples_by_schedule.items()},
            "combined_train_examples": len(combined_train_examples),
            "rollout_eval_samples": rollout_eval_count,
            "s4g_baselines": S4G_BASELINES,
            "s4e_baselines": S4E_BASELINES,
        },
        "schedule_metrics": schedule_metrics,
        "transition_metrics": transition_metrics,
        "rollout_metrics": rollout_metrics,
        "rollout_step_metrics": rollout_step_metrics,
        "training_losses": training_losses,
    }
    result["gates"] = evaluate_gates(schedule_metrics, rollout_metrics, config.gate_tolerance, config.schedules)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_summary(result), encoding="utf-8")
    with (output_dir / "reverse_transition_samples.jsonl").open("w", encoding="utf-8") as file:
        for row in sample_rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (output_dir / "rollout_samples.jsonl").open("w", encoding="utf-8") as file:
        for row in rollout_sample_rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def main() -> None:
    config = parse_args()
    result = run(config)
    print(render_summary(result))


if __name__ == "__main__":
    main()
