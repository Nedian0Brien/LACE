"""LACE V2 S2 Kaggle runner.

S2 is the first skeleton-to-text reconstruction training probe. It fine-tunes
the same T5-small reverse model under several input conditions, then compares
held-out reconstruction metrics and attention-model control inputs.
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
DEFAULT_TRAIN_CONDITIONS = (
    "attention_scaffold",
    "idf_scaffold",
    "random_scaffold",
    "position_prior_scaffold",
    "position_only",
)
ATTENTION_CONTROL_CONDITIONS = (
    "attention_wrong_document",
    "attention_same_position_random",
    "attention_position_only",
)


@dataclass(frozen=True)
class V2S2Config:
    model_name: str = "t5-small"
    max_train_samples: int = 768
    max_eval_samples: int = 192
    max_length: int = 128
    input_max_length: int = 160
    target_max_length: int = 128
    skeleton_batch_size: int = 16
    train_batch_size: int = 8
    eval_batch_size: int = 8
    epochs: int = 1
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    output_dir: str = "/kaggle/working/lace_v2_s2"
    input_text_file: str | None = None
    use_hf_dataset: bool = True
    hf_dataset_name: str = "wikitext"
    hf_dataset_config: str = "wikitext-2-raw-v1"
    hf_dataset_split: str = "train"
    seed: int = 42
    keep_ratio: float = 0.25
    sample_output_count: int = 24
    min_text_words: int = 6
    train_conditions: tuple[str, ...] = DEFAULT_TRAIN_CONDITIONS


class ReconstructionDataset:
    def __init__(self, examples: list[dict[str, str]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.examples[index]


def parse_conditions(value: str) -> tuple[str, ...]:
    conditions = tuple(item.strip() for item in value.split(",") if item.strip())
    if not conditions:
        raise ValueError("At least one train condition is required.")
    valid = set(DEFAULT_TRAIN_CONDITIONS)
    unknown = sorted(set(conditions) - valid)
    if unknown:
        raise ValueError(f"Unknown train conditions: {unknown}")
    return conditions


def parse_args() -> V2S2Config:
    parser = argparse.ArgumentParser(description="Run LACE V2 S2 skeleton-to-text reconstruction.")
    parser.add_argument("--model-name", default=V2S2Config.model_name)
    parser.add_argument("--max-train-samples", type=int, default=V2S2Config.max_train_samples)
    parser.add_argument("--max-eval-samples", type=int, default=V2S2Config.max_eval_samples)
    parser.add_argument("--max-length", type=int, default=V2S2Config.max_length)
    parser.add_argument("--input-max-length", type=int, default=V2S2Config.input_max_length)
    parser.add_argument("--target-max-length", type=int, default=V2S2Config.target_max_length)
    parser.add_argument("--skeleton-batch-size", type=int, default=V2S2Config.skeleton_batch_size)
    parser.add_argument("--train-batch-size", type=int, default=V2S2Config.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=V2S2Config.eval_batch_size)
    parser.add_argument("--epochs", type=int, default=V2S2Config.epochs)
    parser.add_argument("--learning-rate", type=float, default=V2S2Config.learning_rate)
    parser.add_argument("--max-grad-norm", type=float, default=V2S2Config.max_grad_norm)
    parser.add_argument("--output-dir", default=os.environ.get("LACE_OUTPUT_DIR", V2S2Config.output_dir))
    parser.add_argument("--input-text-file", default=os.environ.get("LACE_INPUT_TEXT_FILE"))
    parser.add_argument("--use-hf-dataset", action=argparse.BooleanOptionalAction, default=V2S2Config.use_hf_dataset)
    parser.add_argument("--hf-dataset-name", default=V2S2Config.hf_dataset_name)
    parser.add_argument("--hf-dataset-config", default=V2S2Config.hf_dataset_config)
    parser.add_argument("--hf-dataset-split", default=V2S2Config.hf_dataset_split)
    parser.add_argument("--seed", type=int, default=V2S2Config.seed)
    parser.add_argument("--keep-ratio", type=float, default=V2S2Config.keep_ratio)
    parser.add_argument("--sample-output-count", type=int, default=V2S2Config.sample_output_count)
    parser.add_argument("--min-text-words", type=int, default=V2S2Config.min_text_words)
    parser.add_argument("--train-conditions", default=",".join(DEFAULT_TRAIN_CONDITIONS))
    args = parser.parse_args()
    if args.keep_ratio <= 0 or args.keep_ratio > 1:
        raise ValueError("--keep-ratio must be in the interval (0, 1].")
    return V2S2Config(
        model_name=args.model_name,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_length=args.max_length,
        input_max_length=args.input_max_length,
        target_max_length=args.target_max_length,
        skeleton_batch_size=args.skeleton_batch_size,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
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
        train_conditions=parse_conditions(args.train_conditions),
    )


def set_seed(seed: int) -> None:
    import torch

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


def load_texts(config: V2S2Config) -> tuple[list[str], str]:
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


def load_skeleton_encoder(config: V2S2Config):
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    encoder = T5EncoderModel.from_pretrained(config.model_name)
    encoder.to(device)
    encoder.eval()
    return tokenizer, encoder, device


def special_token_mask(input_ids, tokenizer):
    import torch

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
    import torch

    scores = torch.zeros(attention_mask.shape, dtype=torch.float32, device=attention_mask.device)
    query_mask = attention_mask.bool()
    query_denominator = query_mask.sum(dim=1).clamp(min=1).to(torch.float32)

    for layer_attention in attentions:
        masked = layer_attention * query_mask[:, None, :, None].to(layer_attention.dtype)
        received = masked.sum(dim=2).mean(dim=1)
        received = received / query_denominator[:, None]
        scores += received.to(torch.float32)

    scores = scores / max(1, len(attentions))
    scores = scores.masked_fill(~valid_mask, float("-inf"))
    return scores.cpu()


def encode_for_skeleton(config: V2S2Config, tokenizer, encoder, device: str, texts: list[str]):
    import torch

    input_batches = []
    mask_batches = []
    score_batches = []
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
            input_batches.append(batch["input_ids"].cpu())
            mask_batches.append(batch["attention_mask"].cpu())
            score_batches.append(attention_received_scores(outputs.attentions, batch["attention_mask"], valid_mask))

    input_ids = torch.cat(input_batches, dim=0)
    attention_mask = torch.cat(mask_batches, dim=0)
    valid_mask = attention_mask.bool() & ~special_token_mask(input_ids, tokenizer)
    attention_scores = torch.cat(score_batches, dim=0)
    return input_ids, valid_mask, attention_scores


def build_token_idf(input_ids, valid_mask) -> dict[int, float]:
    document_frequency: Counter[int] = Counter()
    for row_index in range(input_ids.shape[0]):
        token_ids = [int(input_ids[row_index, position].item()) for position in active_positions(valid_mask[row_index])]
        document_frequency.update(set(token_ids))
    total_documents = max(1, input_ids.shape[0])
    return {
        token_id: math.log((total_documents + 1) / (count + 1)) + 1.0
        for token_id, count in document_frequency.items()
    }


def scores_from_token_map(input_ids, valid_mask, score_by_token: dict[int, float]):
    import torch

    scores = torch.full(input_ids.shape, float("-inf"), dtype=torch.float32)
    for row_index in range(input_ids.shape[0]):
        for position in active_positions(valid_mask[row_index]):
            token_id = int(input_ids[row_index, position].item())
            scores[row_index, position] = float(score_by_token.get(token_id, 0.0))
    return scores


def position_prior_scores(valid_mask):
    import torch

    scores = torch.full(valid_mask.shape, float("-inf"), dtype=torch.float32)
    for row_index in range(valid_mask.shape[0]):
        positions = active_positions(valid_mask[row_index])
        denominator = max(1, len(positions) - 1)
        for rank, position in enumerate(positions):
            scores[row_index, position] = 1.0 - (rank / denominator)
    return scores


def select_top_positions(scores: list[float], positions: list[int], keep_count: int) -> list[int]:
    ranked = sorted(positions, key=lambda index: (-float(scores[index]), index))
    return sorted(ranked[:keep_count])


def select_random_positions(positions: list[int], keep_count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(positions, keep_count))


def wrong_index(sample_index: int, sample_count: int, seed: int) -> int:
    if sample_count <= 1:
        return sample_index
    return (sample_index + 1 + (seed % (sample_count - 1))) % sample_count


def decode_positions(tokenizer, input_row, kept_positions: list[int]) -> str:
    token_ids = [int(input_row[position].item()) for position in kept_positions]
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


def decode_custom_ids(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


def position_bins(positions: list[int], valid_positions: list[int]) -> str:
    if not positions or not valid_positions:
        return "none"
    rank_by_position = {position: rank for rank, position in enumerate(valid_positions)}
    denominator = max(1, len(valid_positions) - 1)
    bins = []
    for position in positions:
        normalized = rank_by_position[position] / denominator
        if normalized < 0.33:
            bins.append("front")
        elif normalized < 0.66:
            bins.append("middle")
        else:
            bins.append("back")
    return " ".join(bins)


def idf_positions_for_other(valid_mask, idf_scores, sample_index: int, keep_count: int) -> list[int]:
    positions = active_positions(valid_mask[sample_index])
    if not positions:
        return []
    return select_top_positions(idf_scores[sample_index].tolist(), positions, min(keep_count, len(positions)))


def same_position_wrong_text(tokenizer, input_ids, valid_mask, sample_index: int, positions: list[int], seed: int) -> str:
    other_index = wrong_index(sample_index, input_ids.shape[0], seed)
    other_positions = set(active_positions(valid_mask[other_index]))
    other_valid = active_positions(valid_mask[other_index])
    token_ids = []
    for offset, position in enumerate(positions):
        if position in other_positions:
            token_ids.append(int(input_ids[other_index, position].item()))
        elif other_valid:
            token_ids.append(int(input_ids[other_index, other_valid[offset % len(other_valid)]].item()))
    return decode_custom_ids(tokenizer, token_ids)


def input_template(condition_name: str, skeleton_text: str, positions_text: str) -> str:
    skeleton = skeleton_text if skeleton_text else "none"
    positions = positions_text if positions_text else "none"
    return f"restore text | condition: {condition_name} | positions: {positions} | skeleton: {skeleton}"


def build_condition_payloads(config, tokenizer, input_ids, valid_mask, score_matrices) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for sample_index in range(input_ids.shape[0]):
        positions = active_positions(valid_mask[sample_index])
        if not positions:
            continue
        keep_count = target_keep_count(len(positions), config.keep_ratio)
        idf_scores = score_matrices["idf"][sample_index].tolist()
        attention_scores = score_matrices["attention_received"][sample_index].tolist()
        position_scores = score_matrices["position_prior"][sample_index].tolist()

        idf_positions = select_top_positions(idf_scores, positions, keep_count)
        attention_positions = select_top_positions(attention_scores, positions, keep_count)
        random_positions = select_random_positions(positions, keep_count, config.seed + sample_index * 997)
        prior_positions = select_top_positions(position_scores, positions, keep_count)
        other_index = wrong_index(sample_index, input_ids.shape[0], config.seed)
        wrong_positions = idf_positions_for_other(valid_mask, score_matrices["attention_received"], other_index, keep_count)

        attention_text = decode_positions(tokenizer, input_ids[sample_index], attention_positions)
        idf_text = decode_positions(tokenizer, input_ids[sample_index], idf_positions)
        random_text = decode_positions(tokenizer, input_ids[sample_index], random_positions)
        prior_text = decode_positions(tokenizer, input_ids[sample_index], prior_positions)
        position_text = position_bins(attention_positions, positions)
        wrong_text = decode_positions(tokenizer, input_ids[other_index], wrong_positions)
        same_position_text = same_position_wrong_text(tokenizer, input_ids, valid_mask, sample_index, attention_positions, config.seed)

        payloads.append(
            {
                "sample_id": sample_index,
                "keep_count": keep_count,
                "conditions": {
                    "attention_scaffold": {
                        "input": input_template("attention_scaffold", attention_text, position_text),
                        "skeleton": attention_text,
                    },
                    "idf_scaffold": {
                        "input": input_template("idf_scaffold", idf_text, position_bins(idf_positions, positions)),
                        "skeleton": idf_text,
                    },
                    "random_scaffold": {
                        "input": input_template("random_scaffold", random_text, position_bins(random_positions, positions)),
                        "skeleton": random_text,
                    },
                    "position_prior_scaffold": {
                        "input": input_template("position_prior_scaffold", prior_text, position_bins(prior_positions, positions)),
                        "skeleton": prior_text,
                    },
                    "position_only": {
                        "input": input_template("position_only", "", position_text),
                        "skeleton": "",
                    },
                    "attention_wrong_document": {
                        "input": input_template("attention_wrong_document", wrong_text, position_text),
                        "skeleton": wrong_text,
                    },
                    "attention_same_position_random": {
                        "input": input_template("attention_same_position_random", same_position_text, position_text),
                        "skeleton": same_position_text,
                    },
                    "attention_position_only": {
                        "input": input_template("attention_position_only", "", position_text),
                        "skeleton": "",
                    },
                },
            }
        )
    return payloads


def make_examples(texts: list[str], payloads: list[dict[str, Any]], indices: list[int], condition_name: str) -> list[dict[str, str]]:
    examples = []
    for index in indices:
        condition = payloads[index]["conditions"][condition_name]
        examples.append(
            {
                "source": condition["input"],
                "target": texts[index],
                "skeleton": condition["skeleton"],
                "sample_id": str(index),
            }
        )
    return examples


def collate_batch(tokenizer, config: V2S2Config, device: str, batch: list[dict[str, str]]) -> dict[str, Any]:
    import torch

    sources = [item["source"] for item in batch]
    targets = [item["target"] for item in batch]
    tokenized_sources = tokenizer(
        sources,
        padding=True,
        truncation=True,
        max_length=config.input_max_length,
        return_tensors="pt",
    )
    tokenized_targets = tokenizer(
        text_target=targets,
        padding=True,
        truncation=True,
        max_length=config.target_max_length,
        return_tensors="pt",
    )
    labels = tokenized_targets["input_ids"]
    labels = labels.masked_fill(labels == tokenizer.pad_token_id, -100)
    return {
        "input_ids": tokenized_sources["input_ids"].to(device),
        "attention_mask": tokenized_sources["attention_mask"].to(device),
        "labels": labels.to(device),
        "sources": sources,
        "targets": targets,
        "skeletons": [item["skeleton"] for item in batch],
        "sample_ids": [item["sample_id"] for item in batch],
    }


def make_loader(examples: list[dict[str, str]], batch_size: int, shuffle: bool):
    from torch.utils.data import DataLoader

    dataset = ReconstructionDataset(examples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda items: items)


def train_condition(config: V2S2Config, tokenizer, device: str, train_examples: list[dict[str, str]]):
    import torch
    from torch.optim import AdamW
    from transformers import T5ForConditionalGeneration

    model = T5ForConditionalGeneration.from_pretrained(config.model_name)
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    loader = make_loader(train_examples, config.train_batch_size, shuffle=True)
    losses: list[float] = []

    for _epoch in range(config.epochs):
        for raw_batch in loader:
            batch = collate_batch(tokenizer, config, device, raw_batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_dict=True,
                )
                loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: {float(loss.detach().cpu())}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
    return model, losses


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
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
    token_f1 = f1_from_counts(overlap, len(predicted_words), len(target_words))

    lcs = lcs_length(predicted_words, target_words)
    rouge_l_f1 = f1_from_counts(lcs, len(predicted_words), len(target_words))

    target_keywords = list(dict.fromkeys(content_words(target)))[:8]
    predicted_set = set(predicted_words)
    keyword_recall = (
        sum(1 for item in target_keywords if item in predicted_set) / len(target_keywords)
        if target_keywords
        else 0.0
    )

    skeleton_keywords = list(dict.fromkeys(content_words(skeleton)))
    skeleton_coverage = (
        sum(1 for item in skeleton_keywords if item in predicted_set) / len(skeleton_keywords)
        if skeleton_keywords
        else 0.0
    )
    length_ratio = len(predicted_words) / max(1, len(target_words))
    return {
        "token_f1": token_f1,
        "rouge_l_f1": rouge_l_f1,
        "keyword_recall": keyword_recall,
        "skeleton_coverage": skeleton_coverage,
        "length_ratio": length_ratio,
        "nonempty": 1.0 if prediction.strip() else 0.0,
    }


def mean(items: list[float]) -> float:
    return sum(items) / max(1, len(items))


def evaluate_model(config: V2S2Config, tokenizer, model, device: str, eval_examples: list[dict[str, str]]) -> dict[str, Any]:
    import torch

    model.eval()
    loader = make_loader(eval_examples, config.eval_batch_size, shuffle=False)
    losses: list[float] = []
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = collate_batch(tokenizer, config, device, raw_batch)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                return_dict=True,
            )
            losses.append(float(outputs.loss.detach().cpu()))
            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=config.target_max_length,
                num_beams=1,
            )
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for sample_id, source, target, skeleton, prediction in zip(
                batch["sample_ids"],
                batch["sources"],
                batch["targets"],
                batch["skeletons"],
                predictions,
            ):
                metrics = lexical_metrics(prediction, target, skeleton)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "source": source,
                        "target": target,
                        "skeleton": skeleton,
                        "prediction": prediction,
                        "metrics": metrics,
                    }
                )

    metric_names = ["token_f1", "rouge_l_f1", "keyword_recall", "skeleton_coverage", "length_ratio", "nonempty"]
    aggregate = {name: mean([row["metrics"][name] for row in rows]) for name in metric_names}
    aggregate["eval_loss"] = mean(losses)
    aggregate["eval_ppl"] = math.exp(min(20.0, aggregate["eval_loss"])) if math.isfinite(aggregate["eval_loss"]) else float("inf")
    return {"metrics": aggregate, "samples": rows}


def evaluate_gates(result: dict[str, Any], config: V2S2Config) -> dict[str, Any]:
    metrics = result["condition_metrics"]
    controls = result["attention_control_metrics"]
    attention = metrics.get("attention_scaffold", {})
    random_metrics = metrics.get("random_scaffold", {})
    position = metrics.get("position_only", {})
    wrong = controls.get("attention_wrong_document", {})

    main_conditions = [metrics[name] for name in config.train_conditions if name in metrics]
    losses_finite = all(math.isfinite(item.get("eval_loss", float("inf"))) for item in main_conditions)
    nonempty_ok = all(item.get("nonempty", 0.0) >= 0.8 for item in main_conditions)
    attention_score = max(attention.get("token_f1", 0.0), attention.get("rouge_l_f1", 0.0))
    random_score = max(random_metrics.get("token_f1", 0.0), random_metrics.get("rouge_l_f1", 0.0))
    position_score = max(position.get("token_f1", 0.0), position.get("rouge_l_f1", 0.0))
    wrong_score = max(wrong.get("token_f1", 0.0), wrong.get("rouge_l_f1", 0.0))

    gates = {
        "S2-G-RUN": {
            "pass": bool(metrics),
            "detail": f"{len(metrics)} train conditions and {len(controls)} attention controls evaluated.",
        },
        "S2-G-LOSS-FINITE": {"pass": losses_finite, "detail": {name: metrics[name]["eval_loss"] for name in metrics}},
        "S2-G-NONEMPTY-GENERATION": {
            "pass": nonempty_ok,
            "detail": {name: metrics[name]["nonempty"] for name in metrics},
        },
        "S2-G-ATTENTION-BEATS-RANDOM": {
            "pass": attention_score > random_score,
            "detail": {"attention_score": attention_score, "random_score": random_score},
        },
        "S2-G-ATTENTION-BEATS-POSITION": {
            "pass": attention_score > position_score,
            "detail": {"attention_score": attention_score, "position_only_score": position_score},
        },
        "S2-G-WRONG-DOC-DROPS": {
            "pass": attention_score > wrong_score,
            "detail": {"attention_score": attention_score, "wrong_document_score": wrong_score},
        },
    }
    gates["overall_pass"] = bool(
        gates["S2-G-RUN"]["pass"]
        and gates["S2-G-LOSS-FINITE"]["pass"]
        and gates["S2-G-NONEMPTY-GENERATION"]["pass"]
        and gates["S2-G-ATTENTION-BEATS-RANDOM"]["pass"]
        and gates["S2-G-ATTENTION-BEATS-POSITION"]["pass"]
        and gates["S2-G-WRONG-DOC-DROPS"]["pass"]
    )
    gates["next_ready"] = gates["overall_pass"]
    return gates


def render_summary(result: dict[str, Any]) -> str:
    run_info = result["run_info"]
    gates = result["gates"]
    metrics = result["condition_metrics"]
    controls = result["attention_control_metrics"]
    lines = [
        "# LACE V2 S2 Summary",
        "",
        "## Run Info",
        "",
        f"- phase: `{run_info['phase']}`",
        f"- model: `{run_info['model_name']}`",
        f"- data: `{run_info['text_source']}`",
        f"- train samples: `{run_info['train_samples']}`",
        f"- eval samples: `{run_info['eval_samples']}`",
        f"- device: `{run_info['device']}`",
        f"- keep ratio: `{run_info['keep_ratio']}`",
        f"- epochs: `{run_info['epochs']}`",
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
            f"| `overall_pass` | `{str(gates.get('overall_pass')).lower()}` | S2 reconstruction readiness, not open-ended generation success. |",
            f"| `next_ready` | `{str(gates.get('next_ready')).lower()}` | Whether to move to the next phase. |",
            "",
            "## Train Condition Metrics",
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
            "## Attention Model Control Metrics",
            "",
            "| Condition | Loss | PPL | Token F1 | ROUGE-L | Keyword Recall | Skeleton Coverage | Nonempty |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition_name in sorted(controls):
        item = controls[condition_name]
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
            "S2 checks short skeleton-to-text reconstruction training.",
            "It does not prove open-ended generation quality.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config: V2S2Config) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts, text_source = load_texts(config)
    rng = random.Random(config.seed)
    rng.shuffle(texts)
    train_count = min(config.max_train_samples, max(1, len(texts) - config.max_eval_samples))
    eval_count = min(config.max_eval_samples, len(texts) - train_count)
    if eval_count <= 0:
        raise ValueError("At least one eval sample is required.")
    train_indices = list(range(train_count))
    eval_indices = list(range(train_count, train_count + eval_count))

    skeleton_tokenizer, skeleton_encoder, device = load_skeleton_encoder(config)
    input_ids, valid_mask, attention_scores = encode_for_skeleton(config, skeleton_tokenizer, skeleton_encoder, device, texts)
    del skeleton_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    token_idf = build_token_idf(input_ids, valid_mask)
    score_matrices = {
        "idf": scores_from_token_map(input_ids, valid_mask, token_idf),
        "attention_received": attention_scores,
        "position_prior": position_prior_scores(valid_mask),
    }
    payloads = build_condition_payloads(config, skeleton_tokenizer, input_ids, valid_mask, score_matrices)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    condition_metrics: dict[str, dict[str, float]] = {}
    attention_control_metrics: dict[str, dict[str, float]] = {}
    sample_rows: list[dict[str, Any]] = []
    training_losses: dict[str, list[float]] = {}

    attention_model = None
    for condition_name in config.train_conditions:
        train_examples = make_examples(texts, payloads, train_indices, condition_name)
        eval_examples = make_examples(texts, payloads, eval_indices, condition_name)
        model, losses = train_condition(config, tokenizer, device, train_examples)
        training_losses[condition_name] = losses
        evaluation = evaluate_model(config, tokenizer, model, device, eval_examples)
        condition_metrics[condition_name] = evaluation["metrics"]
        for row in evaluation["samples"][: config.sample_output_count]:
            row["condition"] = condition_name
            sample_rows.append(row)
        if condition_name == "attention_scaffold":
            attention_model = model
        else:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if attention_model is not None:
        for control_name in ATTENTION_CONTROL_CONDITIONS:
            eval_examples = make_examples(texts, payloads, eval_indices, control_name)
            evaluation = evaluate_model(config, tokenizer, attention_model, device, eval_examples)
            attention_control_metrics[control_name] = evaluation["metrics"]
            for row in evaluation["samples"][: config.sample_output_count]:
                row["condition"] = control_name
                sample_rows.append(row)
        del attention_model

    result = {
        "run_info": {
            "phase": "v2_s2",
            "config": asdict(config),
            "text_source": text_source,
            "train_samples": train_count,
            "eval_samples": eval_count,
            "model_name": config.model_name,
            "device": device,
            "keep_ratio": config.keep_ratio,
            "epochs": config.epochs,
            "train_conditions": config.train_conditions,
            "attention_control_conditions": ATTENTION_CONTROL_CONDITIONS,
        },
        "condition_metrics": condition_metrics,
        "attention_control_metrics": attention_control_metrics,
        "training_losses": training_losses,
    }
    result["gates"] = evaluate_gates(result, config)

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
