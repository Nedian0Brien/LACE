"""LACE V2 S0 Kaggle runner.

S0 tests whether an importance-guided token skeleton is a plausible terminal
state for the v2 semantic-compression forward process. It intentionally does
not train a reverse model. The runner builds several same-budget skeletons,
compares semantic preservation against random/uniform baselines, and saves
human-readable samples for qualitative audit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
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


@dataclass(frozen=True)
class V2S0Config:
    model_name: str = "t5-small"
    max_samples: int = 1024
    max_length: int = 128
    encode_batch_size: int = 16
    output_dir: str = "/kaggle/working/lace_v2_s0"
    input_text_file: str | None = None
    use_hf_dataset: bool = True
    hf_dataset_name: str = "wikitext"
    hf_dataset_config: str = "wikitext-2-raw-v1"
    hf_dataset_split: str = "train"
    seed: int = 42
    keep_ratios: tuple[float, ...] = (0.50, 0.25, 0.125)
    semantic_eval_samples: int = 256
    sample_output_count: int = 24
    min_text_words: int = 6


def parse_float_tuple(value: str) -> tuple[float, ...]:
    ratios = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not ratios:
        raise ValueError("At least one keep ratio is required.")
    if any(ratio <= 0 or ratio > 1 for ratio in ratios):
        raise ValueError("Keep ratios must be in the interval (0, 1].")
    return ratios


def parse_args() -> V2S0Config:
    parser = argparse.ArgumentParser(description="Run LACE V2 S0 semantic skeleton sanity check.")
    parser.add_argument("--model-name", default=V2S0Config.model_name)
    parser.add_argument("--max-samples", type=int, default=V2S0Config.max_samples)
    parser.add_argument("--max-length", type=int, default=V2S0Config.max_length)
    parser.add_argument("--encode-batch-size", type=int, default=V2S0Config.encode_batch_size)
    parser.add_argument("--output-dir", default=os.environ.get("LACE_OUTPUT_DIR", V2S0Config.output_dir))
    parser.add_argument("--input-text-file", default=os.environ.get("LACE_INPUT_TEXT_FILE"))
    parser.add_argument("--use-hf-dataset", action=argparse.BooleanOptionalAction, default=V2S0Config.use_hf_dataset)
    parser.add_argument("--hf-dataset-name", default=V2S0Config.hf_dataset_name)
    parser.add_argument("--hf-dataset-config", default=V2S0Config.hf_dataset_config)
    parser.add_argument("--hf-dataset-split", default=V2S0Config.hf_dataset_split)
    parser.add_argument("--seed", type=int, default=V2S0Config.seed)
    parser.add_argument("--keep-ratios", default=",".join(str(item) for item in V2S0Config.keep_ratios))
    parser.add_argument("--semantic-eval-samples", type=int, default=V2S0Config.semantic_eval_samples)
    parser.add_argument("--sample-output-count", type=int, default=V2S0Config.sample_output_count)
    parser.add_argument("--min-text-words", type=int, default=V2S0Config.min_text_words)
    args = parser.parse_args()

    return V2S0Config(
        model_name=args.model_name,
        max_samples=args.max_samples,
        max_length=args.max_length,
        encode_batch_size=args.encode_batch_size,
        output_dir=args.output_dir,
        input_text_file=args.input_text_file,
        use_hf_dataset=args.use_hf_dataset,
        hf_dataset_name=args.hf_dataset_name,
        hf_dataset_config=args.hf_dataset_config,
        hf_dataset_split=args.hf_dataset_split,
        seed=args.seed,
        keep_ratios=parse_float_tuple(args.keep_ratios),
        semantic_eval_samples=args.semantic_eval_samples,
        sample_output_count=args.sample_output_count,
        min_text_words=args.min_text_words,
    )


def normalize_text_line(text: str) -> str:
    return " ".join(text.strip().split())


def word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def load_texts(config: V2S0Config) -> tuple[list[str], str]:
    if config.input_text_file:
        path = Path(config.input_text_file)
        texts = [normalize_text_line(line) for line in path.read_text(encoding="utf-8").splitlines()]
        texts = [text for text in texts if word_count(text) >= config.min_text_words]
        if not texts:
            raise ValueError(f"No usable texts found in {path}.")
        return texts[: config.max_samples], f"file:{path}"

    if config.use_hf_dataset:
        try:
            from datasets import load_dataset

            dataset = load_dataset(config.hf_dataset_name, config.hf_dataset_config, split=config.hf_dataset_split)
            texts: list[str] = []
            for row in dataset:
                text = normalize_text_line(str(row.get("text", "")))
                if word_count(text) >= config.min_text_words:
                    texts.append(text)
                if len(texts) >= config.max_samples:
                    break
            if texts:
                return texts, f"hf:{config.hf_dataset_name}/{config.hf_dataset_config}:{config.hf_dataset_split}"
        except Exception as exc:  # pragma: no cover - Kaggle dependency/network fallback.
            print(f"[warn] Falling back to built-in texts after dataset load failure: {exc}", file=sys.stderr)

    texts = [text for text in FALLBACK_TEXTS if word_count(text) >= config.min_text_words]
    if config.max_samples <= len(texts):
        return texts[: config.max_samples], "fallback"
    repeated: list[str] = []
    while len(repeated) < config.max_samples:
        repeated.extend(texts)
    return repeated[: config.max_samples], "fallback-repeated"


def load_encoder(config: V2S0Config):
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


def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp(min=1.0)
    return (hidden_states * mask).sum(dim=1) / denominator


def attention_received_scores(attentions, attention_mask, valid_mask):
    import torch

    scores = torch.zeros(attention_mask.shape, dtype=torch.float32, device=attention_mask.device)
    query_mask = attention_mask.bool()
    query_denominator = query_mask.sum(dim=1).clamp(min=1).to(torch.float32)

    for layer_attention in attentions:
        # [batch, heads, query, key] -> [batch, key]
        masked = layer_attention * query_mask[:, None, :, None].to(layer_attention.dtype)
        received = masked.sum(dim=2).mean(dim=1)
        received = received / query_denominator[:, None]
        scores += received.to(torch.float32)

    scores = scores / max(1, len(attentions))
    scores = scores.masked_fill(~valid_mask, float("-inf"))
    return scores.cpu()


def encode_texts(config: V2S0Config, tokenizer, encoder, device: str, texts: list[str], return_attentions: bool):
    import torch

    embedding_batches = []
    input_batches = []
    mask_batches = []
    score_batches = []

    with torch.no_grad():
        for start in range(0, len(texts), config.encode_batch_size):
            batch_texts = texts[start : start + config.encode_batch_size]
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
                output_attentions=return_attentions,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            embeddings = mean_pool(hidden_states, batch["attention_mask"])
            embedding_batches.append(embeddings.cpu())
            input_batches.append(batch["input_ids"].cpu())
            mask_batches.append(batch["attention_mask"].cpu())

            if return_attentions:
                valid_mask = batch["attention_mask"].bool() & ~special_token_mask(batch["input_ids"], tokenizer)
                score_batches.append(attention_received_scores(outputs.attentions, batch["attention_mask"], valid_mask))

    embeddings = torch.cat(embedding_batches, dim=0)
    input_ids = torch.cat(input_batches, dim=0)
    attention_mask = torch.cat(mask_batches, dim=0)
    attention_scores = torch.cat(score_batches, dim=0) if score_batches else None
    return embeddings, input_ids, attention_mask, attention_scores


def build_valid_mask(input_ids, attention_mask, tokenizer):
    return attention_mask.bool() & ~special_token_mask(input_ids, tokenizer)


def active_positions(valid_row) -> list[int]:
    return [index for index, is_active in enumerate(valid_row.tolist()) if bool(is_active)]


def target_keep_count(active_count: int, keep_ratio: float) -> int:
    if active_count <= 0:
        return 0
    return max(1, min(active_count, int(round(active_count * keep_ratio))))


def select_top_positions(scores: list[float], positions: list[int], keep_count: int) -> list[int]:
    ranked = sorted(positions, key=lambda index: (-float(scores[index]), index))
    return sorted(ranked[:keep_count])


def select_uniform_positions(positions: list[int], keep_count: int) -> list[int]:
    if keep_count >= len(positions):
        return list(positions)
    if keep_count <= 1:
        return [positions[len(positions) // 2]]
    selected = []
    for item_index in range(keep_count):
        source_index = round(item_index * (len(positions) - 1) / (keep_count - 1))
        selected.append(positions[source_index])
    return sorted(set(selected))


def select_random_positions(positions: list[int], keep_count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(positions, keep_count))


def normalize_word(word: str) -> str:
    return word.strip("'").lower()


def words_for_text(text: str) -> list[str]:
    return [normalize_word(match.group(0)) for match in WORD_PATTERN.finditer(text)]


def build_word_idf(texts: list[str]) -> dict[str, float]:
    document_frequency: Counter[str] = Counter()
    for text in texts:
        document_frequency.update(set(word for word in words_for_text(text) if word))
    total_documents = max(1, len(texts))
    return {word: math.log((total_documents + 1) / (count + 1)) + 1.0 for word, count in document_frequency.items()}


def extract_keywords(text: str, word_idf: dict[str, float], limit: int = 8) -> list[str]:
    candidates = []
    for word in words_for_text(text):
        if not word or word in STOPWORDS:
            continue
        if len(word) < 4 and not any(char.isdigit() for char in word):
            continue
        score = word_idf.get(word, 1.0) + min(len(word), 12) * 0.03
        candidates.append((score, word))
    seen: set[str] = set()
    unique = []
    for score, word in sorted(candidates, reverse=True):
        if word not in seen:
            seen.add(word)
            unique.append((score, word))
    return [word for _score, word in unique[:limit]]


def extract_entities(text: str) -> list[str]:
    entities = []
    for match in WORD_PATTERN.finditer(text):
        raw = match.group(0)
        stripped = raw.strip("'")
        if any(char.isdigit() for char in stripped):
            entities.append(normalize_word(stripped))
        elif len(stripped) > 1 and stripped[0].isupper() and stripped.lower() not in STOPWORDS:
            entities.append(normalize_word(stripped))
        elif stripped.isupper() and len(stripped) > 1:
            entities.append(normalize_word(stripped))
    return sorted(set(entities))


def recall(reference_items: list[str], skeleton_text: str) -> float | None:
    if not reference_items:
        return None
    skeleton_words = set(words_for_text(skeleton_text))
    hits = sum(1 for item in reference_items if item in skeleton_words)
    return hits / len(reference_items)


def safe_mean(values: list[float]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not valid:
        return None
    return sum(valid) / len(valid)


def cosine_similarity_mean(reference_embeddings, candidate_embeddings) -> float:
    import torch
    import torch.nn.functional as F

    if reference_embeddings.numel() == 0 or candidate_embeddings.numel() == 0:
        return 0.0
    count = min(reference_embeddings.shape[0], candidate_embeddings.shape[0])
    similarities = F.cosine_similarity(reference_embeddings[:count], candidate_embeddings[:count], dim=1)
    return float(similarities.mean().item())


def build_token_statistics(input_ids, valid_mask) -> tuple[dict[int, int], dict[int, float], dict[int, float]]:
    token_counts: Counter[int] = Counter()
    token_document_frequency: Counter[int] = Counter()

    for row_index in range(input_ids.shape[0]):
        ids = [int(input_ids[row_index, position].item()) for position in active_positions(valid_mask[row_index])]
        token_counts.update(ids)
        token_document_frequency.update(set(ids))

    total_documents = max(1, input_ids.shape[0])
    token_idf = {
        token_id: math.log((total_documents + 1) / (document_count + 1)) + 1.0
        for token_id, document_count in token_document_frequency.items()
    }
    token_frequency = {token_id: math.log(count + 1.0) for token_id, count in token_counts.items()}
    return dict(token_counts), token_idf, token_frequency


def scores_from_token_map(input_ids, valid_mask, score_by_token: dict[int, float]):
    import torch

    scores = torch.full(input_ids.shape, float("-inf"), dtype=torch.float32)
    for row_index in range(input_ids.shape[0]):
        for position in active_positions(valid_mask[row_index]):
            token_id = int(input_ids[row_index, position].item())
            scores[row_index, position] = float(score_by_token.get(token_id, 0.0))
    return scores


def position_prior_scores(input_ids, valid_mask):
    import torch

    del input_ids
    scores = torch.full(valid_mask.shape, float("-inf"), dtype=torch.float32)
    for row_index in range(valid_mask.shape[0]):
        positions = active_positions(valid_mask[row_index])
        denominator = max(1, len(positions) - 1)
        for rank, position in enumerate(positions):
            scores[row_index, position] = 1.0 - (rank / denominator)
    return scores


def position_mean(positions: list[int], valid_positions: list[int]) -> float | None:
    if not positions or not valid_positions:
        return None
    rank_by_position = {position: rank for rank, position in enumerate(valid_positions)}
    denominator = max(1, len(valid_positions) - 1)
    return sum(rank_by_position[position] / denominator for position in positions) / len(positions)


def summarize_score_stats(score_name: str, scores, valid_mask) -> dict[str, Any]:
    rows = []
    correlations = []
    for row_index in range(scores.shape[0]):
        positions = active_positions(valid_mask[row_index])
        if len(positions) < 2:
            continue
        values = [float(scores[row_index, position].item()) for position in positions]
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            continue
        mean_value = sum(finite_values) / len(finite_values)
        variance = sum((value - mean_value) ** 2 for value in finite_values) / len(finite_values)
        std_value = math.sqrt(variance)
        normalized_positions = [rank / (len(positions) - 1) for rank in range(len(positions))]
        position_mean_value = sum(normalized_positions) / len(normalized_positions)
        position_std = math.sqrt(sum((value - position_mean_value) ** 2 for value in normalized_positions) / len(normalized_positions))
        if std_value > 0 and position_std > 0:
            covariance = sum(
                (score - mean_value) * (position - position_mean_value)
                for score, position in zip(finite_values, normalized_positions)
            ) / len(finite_values)
            correlations.append(covariance / (std_value * position_std))
        rows.append({"mean": mean_value, "std": std_value, "min": min(finite_values), "max": max(finite_values)})

    std_values = [row["std"] for row in rows]
    return {
        "score_name": score_name,
        "sample_count": len(rows),
        "mean_score_std": safe_mean(std_values),
        "mean_position_correlation": safe_mean(correlations),
        "max_abs_position_correlation": max((abs(value) for value in correlations), default=None),
    }


def decode_positions(tokenizer, input_row, kept_positions: list[int]) -> str:
    token_ids = [int(input_row[position].item()) for position in kept_positions]
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


def run_skeleton_evaluation(config: V2S0Config) -> dict[str, Any]:
    import torch

    random.seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts, text_source = load_texts(config)
    tokenizer, encoder, device = load_encoder(config)
    original_embeddings, input_ids, attention_mask, attention_scores = encode_texts(
        config,
        tokenizer,
        encoder,
        device,
        texts,
        return_attentions=True,
    )
    if attention_scores is None:
        raise RuntimeError("Attention scores were not returned by the encoder.")

    valid_mask = build_valid_mask(input_ids, attention_mask, tokenizer)
    token_counts, token_idf, token_frequency = build_token_statistics(input_ids, valid_mask)
    word_idf = build_word_idf(texts)

    score_matrices = {
        "attention_received": attention_scores,
        "idf": scores_from_token_map(input_ids, valid_mask, token_idf),
        "frequency": scores_from_token_map(input_ids, valid_mask, token_frequency),
        "position_prior": position_prior_scores(input_ids, valid_mask),
    }
    scorer_names = ("random", "uniform", "frequency", "idf", "position_prior", "attention_received")

    keywords_by_sample = [extract_keywords(text, word_idf) for text in texts]
    entities_by_sample = [extract_entities(text) for text in texts]
    semantic_eval_count = max(0, min(config.semantic_eval_samples, len(texts)))

    aggregate: dict[str, dict[str, Any]] = {}
    skeleton_texts: dict[str, list[str]] = defaultdict(list)
    skeleton_samples: list[dict[str, Any]] = []

    for sample_index, text in enumerate(texts):
        valid_positions = active_positions(valid_mask[sample_index])
        if not valid_positions:
            continue
        sample_payload: dict[str, Any] | None = None
        if len(skeleton_samples) < config.sample_output_count:
            sample_payload = {
                "sample_id": sample_index,
                "text": text,
                "active_token_count": len(valid_positions),
                "keywords": keywords_by_sample[sample_index],
                "entities": entities_by_sample[sample_index],
                "skeletons": {},
            }

        for keep_ratio in config.keep_ratios:
            keep_count = target_keep_count(len(valid_positions), keep_ratio)
            ratio_key = f"{keep_ratio:.3f}".rstrip("0").rstrip(".")
            if sample_payload is not None:
                sample_payload["skeletons"].setdefault(ratio_key, {})

            for scorer_name in scorer_names:
                if scorer_name == "random":
                    kept_positions = select_random_positions(
                        valid_positions,
                        keep_count,
                        config.seed + sample_index * 1009 + int(keep_ratio * 1000),
                    )
                elif scorer_name == "uniform":
                    kept_positions = select_uniform_positions(valid_positions, keep_count)
                else:
                    score_row = score_matrices[scorer_name][sample_index].tolist()
                    kept_positions = select_top_positions(score_row, valid_positions, keep_count)

                skeleton_text = decode_positions(tokenizer, input_ids[sample_index], kept_positions)
                condition_key = f"{scorer_name}@{ratio_key}"
                skeleton_texts[condition_key].append(skeleton_text)

                key = aggregate.setdefault(
                    condition_key,
                    {
                        "scorer": scorer_name,
                        "keep_ratio": keep_ratio,
                        "sample_count": 0,
                        "target_kept_tokens": [],
                        "actual_kept_tokens": [],
                        "keyword_recall": [],
                        "entity_recall": [],
                        "position_mean": [],
                        "empty_skeleton_count": 0,
                    },
                )
                key["sample_count"] += 1
                key["target_kept_tokens"].append(keep_count)
                key["actual_kept_tokens"].append(len(kept_positions))
                keyword_recall = recall(keywords_by_sample[sample_index], skeleton_text)
                entity_recall = recall(entities_by_sample[sample_index], skeleton_text)
                if keyword_recall is not None:
                    key["keyword_recall"].append(keyword_recall)
                if entity_recall is not None:
                    key["entity_recall"].append(entity_recall)
                kept_position_mean = position_mean(kept_positions, valid_positions)
                if kept_position_mean is not None:
                    key["position_mean"].append(kept_position_mean)
                if not skeleton_text:
                    key["empty_skeleton_count"] += 1

                if sample_payload is not None:
                    score_values = []
                    if scorer_name in score_matrices:
                        score_values = [float(score_matrices[scorer_name][sample_index, position].item()) for position in kept_positions]
                    sample_payload["skeletons"][ratio_key][scorer_name] = {
                        "text": skeleton_text,
                        "kept_indices": kept_positions,
                        "kept_token_count": len(kept_positions),
                        "scores": score_values,
                    }

        if sample_payload is not None:
            skeleton_samples.append(sample_payload)

    metrics: dict[str, Any] = {}
    for condition_key, values in aggregate.items():
        target_counts = values.pop("target_kept_tokens")
        actual_counts = values.pop("actual_kept_tokens")
        keyword_recalls = values.pop("keyword_recall")
        entity_recalls = values.pop("entity_recall")
        position_means = values.pop("position_mean")
        target_mean = safe_mean(target_counts)
        actual_mean = safe_mean(actual_counts)
        values["target_kept_tokens_mean"] = target_mean
        values["actual_kept_tokens_mean"] = actual_mean
        values["count_gap_mean"] = None if target_mean is None or actual_mean is None else abs(actual_mean - target_mean)
        values["keyword_recall_mean"] = safe_mean(keyword_recalls)
        values["entity_recall_mean"] = safe_mean(entity_recalls)
        values["position_mean"] = safe_mean(position_means)
        values["empty_skeleton_rate"] = values["empty_skeleton_count"] / max(1, values["sample_count"])
        metrics[condition_key] = values

    if semantic_eval_count > 0:
        reference_embeddings = original_embeddings[:semantic_eval_count]
        for condition_key, condition_texts in skeleton_texts.items():
            eval_texts = condition_texts[:semantic_eval_count]
            candidate_embeddings, _ids, _mask, _scores = encode_texts(
                config,
                tokenizer,
                encoder,
                device,
                eval_texts,
                return_attentions=False,
            )
            metrics[condition_key]["semantic_similarity_mean"] = cosine_similarity_mean(reference_embeddings, candidate_embeddings)
            metrics[condition_key]["semantic_eval_count"] = semantic_eval_count

    score_stats = {
        name: summarize_score_stats(name, scores, valid_mask)
        for name, scores in score_matrices.items()
    }
    score_stats["token_count_unique"] = len(token_counts)
    score_stats["active_token_count"] = int(valid_mask.sum().item())

    gates = evaluate_gates(config, metrics, score_stats)
    run_info = {
        "phase": "v2_s0",
        "config": asdict(config),
        "text_source": text_source,
        "sample_count": len(texts),
        "model_name": config.model_name,
        "device": device,
        "max_length": config.max_length,
        "active_tokens": int(valid_mask.sum().item()),
        "scorers": list(scorer_names),
        "keep_ratios": list(config.keep_ratios),
    }
    result = {
        "run_info": run_info,
        "gates": gates,
        "metrics": metrics,
        "score_stats": score_stats,
    }

    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "score_stats.json").write_text(json.dumps(score_stats, indent=2, sort_keys=True), encoding="utf-8")
    with (output_dir / "skeleton_samples.jsonl").open("w", encoding="utf-8") as file:
        for item in skeleton_samples:
            file.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(render_summary(result), encoding="utf-8")
    return result


def metric_value(metrics: dict[str, Any], scorer: str, ratio: float, name: str) -> float | None:
    ratio_key = f"{ratio:.3f}".rstrip("0").rstrip(".")
    value = metrics.get(f"{scorer}@{ratio_key}", {}).get(name)
    if value is None:
        return None
    return float(value)


def best_gap_against_random(metrics: dict[str, Any], ratios: tuple[float, ...], candidate_scorers: tuple[str, ...], metric_name: str) -> float | None:
    gaps: list[float] = []
    for ratio in ratios:
        random_value = metric_value(metrics, "random", ratio, metric_name)
        if random_value is None:
            continue
        for scorer in candidate_scorers:
            candidate_value = metric_value(metrics, scorer, ratio, metric_name)
            if candidate_value is not None:
                gaps.append(candidate_value - random_value)
    return max(gaps) if gaps else None


def evaluate_gates(config: V2S0Config, metrics: dict[str, Any], score_stats: dict[str, Any]) -> dict[str, Any]:
    count_gaps = [value.get("count_gap_mean") for value in metrics.values() if value.get("count_gap_mean") is not None]
    max_count_gap = max(count_gaps, default=float("inf"))
    candidate_scorers = ("attention_received", "idf")
    keyword_gap = best_gap_against_random(metrics, config.keep_ratios, candidate_scorers, "keyword_recall_mean")
    entity_gap = best_gap_against_random(metrics, config.keep_ratios, candidate_scorers, "entity_recall_mean")
    semantic_gap = best_gap_against_random(metrics, config.keep_ratios, candidate_scorers, "semantic_similarity_mean")

    attention_stats = score_stats.get("attention_received", {})
    attention_std = attention_stats.get("mean_score_std")
    attention_corr = attention_stats.get("mean_position_correlation")
    noncollapse = bool(
        attention_std is not None
        and attention_std > 1e-8
        and (attention_corr is None or abs(float(attention_corr)) < 0.95)
    )

    gates = {
        "S0-G-RUN": {
            "pass": bool(metrics),
            "detail": f"{len(metrics)} scorer/ratio metric groups generated.",
        },
        "S0-G-COUNT-MATCH": {
            "pass": max_count_gap <= 1.0,
            "detail": f"max average kept-token gap={max_count_gap:.4f}",
        },
        "S0-G-ENTITY-GAP": {
            "pass": max(keyword_gap or float("-inf"), entity_gap or float("-inf")) > 0.0,
            "detail": {
                "best_keyword_gap_vs_random": keyword_gap,
                "best_entity_gap_vs_random": entity_gap,
            },
        },
        "S0-G-SEMANTIC-GAP": {
            "pass": semantic_gap is not None and semantic_gap > 0.0,
            "detail": {"best_semantic_similarity_gap_vs_random": semantic_gap},
        },
        "S0-G-SCORER-NONCOLLAPSE": {
            "pass": noncollapse,
            "detail": {
                "attention_mean_score_std": attention_std,
                "attention_mean_position_correlation": attention_corr,
            },
        },
    }
    gates["overall_pass"] = bool(
        gates["S0-G-RUN"]["pass"]
        and gates["S0-G-COUNT-MATCH"]["pass"]
        and (gates["S0-G-ENTITY-GAP"]["pass"] or gates["S0-G-SEMANTIC-GAP"]["pass"])
    )
    gates["s1_ready"] = gates["overall_pass"]
    return gates


def render_summary(result: dict[str, Any]) -> str:
    run_info = result["run_info"]
    gates = result["gates"]
    metrics = result["metrics"]

    lines = [
        "# LACE V2 S0 Summary",
        "",
        "## Run Info",
        "",
        f"- phase: `{run_info['phase']}`",
        f"- model: `{run_info['model_name']}`",
        f"- data: `{run_info['text_source']}`",
        f"- samples: `{run_info['sample_count']}`",
        f"- device: `{run_info['device']}`",
        f"- active tokens: `{run_info['active_tokens']}`",
        "",
        "## Gates",
        "",
        "| Gate | Pass | Detail |",
        "|---|---:|---|",
    ]
    for gate_name, gate_value in gates.items():
        if not isinstance(gate_value, dict):
            continue
        detail = gate_value.get("detail", "")
        lines.append(f"| `{gate_name}` | `{str(gate_value.get('pass')).lower()}` | `{json.dumps(detail, sort_keys=True)}` |")
    lines.extend(
        [
            f"| `overall_pass` | `{str(gates.get('overall_pass')).lower()}` | S0 pipeline readiness, not generation success. |",
            f"| `s1_ready` | `{str(gates.get('s1_ready')).lower()}` | Whether skeleton artifacts are ready for S1 controls. |",
            "",
            "## Core Metrics",
            "",
            "| Condition | Kept | Keyword Recall | Entity Recall | Semantic Similarity | Position Mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for condition_key in sorted(metrics):
        item = metrics[condition_key]
        lines.append(
            "| `{}` | {:.2f} | {} | {} | {} | {} |".format(
                condition_key,
                float(item.get("actual_kept_tokens_mean") or 0.0),
                format_optional(item.get("keyword_recall_mean")),
                format_optional(item.get("entity_recall_mean")),
                format_optional(item.get("semantic_similarity_mean")),
                format_optional(item.get("position_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "S0 only checks whether semantic skeleton construction is worth carrying into S1/S2.",
            "It does not show reverse-model skeleton use, reconstruction quality, or generation quality.",
        ]
    )
    return "\n".join(lines) + "\n"


def format_optional(value: Any) -> str:
    if value is None:
        return "`n/a`"
    return f"{float(value):.4f}"


def main() -> None:
    config = parse_args()
    result = run_skeleton_evaluation(config)
    print(render_summary(result))


if __name__ == "__main__":
    main()
