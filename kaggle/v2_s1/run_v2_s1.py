"""LACE V2 S1 Kaggle runner.

S1 checks whether a semantic skeleton can actually identify its source text
under strong controls. It does not train a generator yet. Instead, it encodes
each skeleton/control with a frozen T5 encoder and measures whether the query
retrieves the original text from a candidate pool.
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


WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9'-]*|\d+(?:\.\d+)?")


@dataclass(frozen=True)
class V2S1Config:
    model_name: str = "t5-small"
    max_samples: int = 1024
    max_length: int = 128
    encode_batch_size: int = 16
    output_dir: str = "/kaggle/working/lace_v2_s1"
    input_text_file: str | None = None
    use_hf_dataset: bool = True
    hf_dataset_name: str = "wikitext"
    hf_dataset_config: str = "wikitext-2-raw-v1"
    hf_dataset_split: str = "train"
    seed: int = 42
    keep_ratio: float = 0.25
    sample_output_count: int = 24
    min_text_words: int = 6


def parse_args() -> V2S1Config:
    parser = argparse.ArgumentParser(description="Run LACE V2 S1 skeleton-use controls.")
    parser.add_argument("--model-name", default=V2S1Config.model_name)
    parser.add_argument("--max-samples", type=int, default=V2S1Config.max_samples)
    parser.add_argument("--max-length", type=int, default=V2S1Config.max_length)
    parser.add_argument("--encode-batch-size", type=int, default=V2S1Config.encode_batch_size)
    parser.add_argument("--output-dir", default=os.environ.get("LACE_OUTPUT_DIR", V2S1Config.output_dir))
    parser.add_argument("--input-text-file", default=os.environ.get("LACE_INPUT_TEXT_FILE"))
    parser.add_argument("--use-hf-dataset", action=argparse.BooleanOptionalAction, default=V2S1Config.use_hf_dataset)
    parser.add_argument("--hf-dataset-name", default=V2S1Config.hf_dataset_name)
    parser.add_argument("--hf-dataset-config", default=V2S1Config.hf_dataset_config)
    parser.add_argument("--hf-dataset-split", default=V2S1Config.hf_dataset_split)
    parser.add_argument("--seed", type=int, default=V2S1Config.seed)
    parser.add_argument("--keep-ratio", type=float, default=V2S1Config.keep_ratio)
    parser.add_argument("--sample-output-count", type=int, default=V2S1Config.sample_output_count)
    parser.add_argument("--min-text-words", type=int, default=V2S1Config.min_text_words)
    args = parser.parse_args()
    if args.keep_ratio <= 0 or args.keep_ratio > 1:
        raise ValueError("--keep-ratio must be in the interval (0, 1].")
    return V2S1Config(
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
        keep_ratio=args.keep_ratio,
        sample_output_count=args.sample_output_count,
        min_text_words=args.min_text_words,
    )


def normalize_text_line(text: str) -> str:
    return " ".join(text.strip().split())


def word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def load_texts(config: V2S1Config) -> tuple[list[str], str]:
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


def load_encoder(config: V2S1Config):
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


def active_positions(valid_row) -> list[int]:
    return [index for index, is_active in enumerate(valid_row.tolist()) if bool(is_active)]


def target_keep_count(active_count: int, keep_ratio: float) -> int:
    if active_count <= 0:
        return 0
    return max(1, min(active_count, int(round(active_count * keep_ratio))))


def encode_texts(
    config: V2S1Config,
    tokenizer,
    encoder,
    device: str,
    texts: list[str],
    return_attentions: bool,
):
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
            embedding_batches.append(mean_pool(outputs.last_hidden_state, batch["attention_mask"]).cpu())
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


def build_valid_mask(input_ids, attention_mask, tokenizer):
    return attention_mask.bool() & ~special_token_mask(input_ids, tokenizer)


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
        return "위치 없음"
    rank_by_position = {position: rank for rank, position in enumerate(valid_positions)}
    denominator = max(1, len(valid_positions) - 1)
    bins = []
    for position in positions:
        normalized = rank_by_position[position] / denominator
        if normalized < 0.33:
            bins.append("앞")
        elif normalized < 0.66:
            bins.append("중간")
        else:
            bins.append("뒤")
    return " ".join(bins)


def shuffled_text(tokenizer, input_row, kept_positions: list[int], seed: int) -> str:
    rng = random.Random(seed)
    token_ids = [int(input_row[position].item()) for position in kept_positions]
    rng.shuffle(token_ids)
    return decode_custom_ids(tokenizer, token_ids)


def same_position_wrong_text(tokenizer, input_ids, valid_mask, sample_index: int, positions: list[int], seed: int) -> str:
    other_index = wrong_index(sample_index, input_ids.shape[0], seed)
    other_positions = set(active_positions(valid_mask[other_index]))
    token_ids = []
    other_valid = active_positions(valid_mask[other_index])
    for offset, position in enumerate(positions):
        if position in other_positions:
            token_ids.append(int(input_ids[other_index, position].item()))
        elif other_valid:
            token_ids.append(int(input_ids[other_index, other_valid[offset % len(other_valid)]].item()))
    return decode_custom_ids(tokenizer, token_ids)


def remove_top_low_texts(tokenizer, input_row, kept_positions: list[int], score_row: list[float]) -> tuple[str, str]:
    if len(kept_positions) <= 1:
        text = decode_positions(tokenizer, input_row, kept_positions)
        return text, text
    ranked = sorted(kept_positions, key=lambda index: float(score_row[index]), reverse=True)
    remove_count = max(1, len(kept_positions) // 2)
    top_removed = set(ranked[:remove_count])
    low_removed = set(ranked[-remove_count:])
    top_remaining = [position for position in kept_positions if position not in top_removed]
    low_remaining = [position for position in kept_positions if position not in low_removed]
    return decode_positions(tokenizer, input_row, top_remaining), decode_positions(tokenizer, input_row, low_remaining)


def build_condition_texts(config, tokenizer, input_ids, valid_mask, score_matrices) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    condition_texts: dict[str, list[str]] = defaultdict(list)
    samples: list[dict[str, Any]] = []

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
        position_prior_positions = select_top_positions(position_scores, positions, keep_count)
        random_positions = select_random_positions(positions, keep_count, config.seed + sample_index * 997)
        wrong_doc_index = wrong_index(sample_index, input_ids.shape[0], config.seed)

        condition_payload = {
            "idf_correct": decode_positions(tokenizer, input_ids[sample_index], idf_positions),
            "attention_correct": decode_positions(tokenizer, input_ids[sample_index], attention_positions),
            "random_same_count": decode_positions(tokenizer, input_ids[sample_index], random_positions),
            "position_prior": decode_positions(tokenizer, input_ids[sample_index], position_prior_positions),
            "position_only": position_bins(idf_positions, positions),
            "shuffled_correct": shuffled_text(tokenizer, input_ids[sample_index], idf_positions, config.seed + sample_index),
            "wrong_document": decode_positions(tokenizer, input_ids[wrong_doc_index], idf_positions_for_other(valid_mask, score_matrices["idf"], wrong_doc_index, keep_count)),
            "same_position_random": same_position_wrong_text(tokenizer, input_ids, valid_mask, sample_index, idf_positions, config.seed),
        }
        top_removed, low_removed = remove_top_low_texts(tokenizer, input_ids[sample_index], idf_positions, idf_scores)
        condition_payload["remove_topk"] = top_removed
        condition_payload["remove_lowk"] = low_removed

        for condition_name, text in condition_payload.items():
            condition_texts[condition_name].append(text if text else "빈 골격")

        if len(samples) < config.sample_output_count:
            samples.append(
                {
                    "sample_id": sample_index,
                    "active_token_count": len(positions),
                    "keep_count": keep_count,
                    "wrong_doc_index": wrong_doc_index,
                    "idf_positions": idf_positions,
                    "attention_positions": attention_positions,
                    "conditions": condition_payload,
                }
            )

    return condition_texts, samples


def idf_positions_for_other(valid_mask, idf_scores, sample_index: int, keep_count: int) -> list[int]:
    positions = active_positions(valid_mask[sample_index])
    if not positions:
        return []
    return select_top_positions(idf_scores[sample_index].tolist(), positions, min(keep_count, len(positions)))


def normalize_embeddings(embeddings):
    import torch.nn.functional as F

    return F.normalize(embeddings.float(), dim=1)


def retrieval_metrics(query_embeddings, original_embeddings) -> dict[str, float]:
    import torch

    queries = normalize_embeddings(query_embeddings)
    originals = normalize_embeddings(original_embeddings)
    scores = queries @ originals.T
    sample_count = scores.shape[0]
    ranks = []
    own_scores = []
    margins = []
    for index in range(sample_count):
        row = scores[index]
        sorted_indices = torch.argsort(row, descending=True)
        rank = int((sorted_indices == index).nonzero(as_tuple=False)[0].item()) + 1
        ranks.append(rank)
        own_score = float(row[index].item())
        own_scores.append(own_score)
        if sample_count > 1:
            masked = row.clone()
            masked[index] = -1e9
            margins.append(own_score - float(masked.max().item()))
    hit1 = sum(1 for rank in ranks if rank <= 1) / sample_count
    hit5 = sum(1 for rank in ranks if rank <= 5) / sample_count
    mrr = sum(1.0 / rank for rank in ranks) / sample_count
    return {
        "hit_at_1": hit1,
        "hit_at_5": hit5,
        "mrr": mrr,
        "own_similarity_mean": sum(own_scores) / sample_count,
        "margin_to_best_wrong_mean": sum(margins) / max(1, len(margins)),
    }


def evaluate_gates(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    best_correct = max(metrics["idf_correct"]["hit_at_1"], metrics["attention_correct"]["hit_at_1"])
    random_hit = metrics["random_same_count"]["hit_at_1"]
    wrong_hit = metrics["wrong_document"]["hit_at_1"]
    position_only_hit = metrics["position_only"]["hit_at_1"]
    same_position_hit = metrics["same_position_random"]["hit_at_1"]
    top_hit = metrics["remove_topk"]["hit_at_1"]
    low_hit = metrics["remove_lowk"]["hit_at_1"]
    top_similarity = metrics["remove_topk"]["own_similarity_mean"]
    low_similarity = metrics["remove_lowk"]["own_similarity_mean"]
    best_correct_similarity = max(metrics["idf_correct"]["own_similarity_mean"], metrics["attention_correct"]["own_similarity_mean"])
    shuffled_hit = metrics["shuffled_correct"]["hit_at_1"]
    correct_name = "idf_correct" if metrics["idf_correct"]["hit_at_1"] >= metrics["attention_correct"]["hit_at_1"] else "attention_correct"

    removal_drops = (
        (top_hit < best_correct or top_similarity < best_correct_similarity)
        and (low_hit < best_correct or low_similarity < best_correct_similarity)
    )
    top_order = top_hit < low_hit or top_similarity < low_similarity

    gates = {
        "S1-G-RUN": {"pass": bool(metrics), "detail": f"{len(metrics)} condition groups generated."},
        "S1-G-CORRECT-BEATS-RANDOM": {
            "pass": best_correct > random_hit,
            "detail": {"best_correct": best_correct, "random_same_count": random_hit, "best_correct_name": correct_name},
        },
        "S1-G-WRONG-DOC-DROPS": {
            "pass": best_correct > wrong_hit,
            "detail": {"best_correct": best_correct, "wrong_document": wrong_hit},
        },
        "S1-G-POSITION-ONLY-DROPS": {
            "pass": best_correct > position_only_hit and best_correct > same_position_hit,
            "detail": {
                "best_correct": best_correct,
                "position_only": position_only_hit,
                "same_position_random": same_position_hit,
            },
        },
        "S1-G-REMOVAL-DROPS": {
            "pass": removal_drops,
            "detail": {
                "remove_topk_hit": top_hit,
                "remove_lowk_hit": low_hit,
                "best_correct_hit": best_correct,
                "remove_topk_similarity": top_similarity,
                "remove_lowk_similarity": low_similarity,
                "best_correct_similarity": best_correct_similarity,
            },
        },
        "S1-G-TOPK-ORDER": {
            "pass": top_order,
            "detail": {
                "remove_topk_hit": top_hit,
                "remove_lowk_hit": low_hit,
                "remove_topk_similarity": top_similarity,
                "remove_lowk_similarity": low_similarity,
            },
        },
        "S1-G-SHUFFLE-SENSITIVE": {
            "pass": best_correct > shuffled_hit,
            "detail": {"best_correct": best_correct, "shuffled_correct": shuffled_hit},
        },
    }
    gates["overall_pass"] = bool(
        gates["S1-G-RUN"]["pass"]
        and gates["S1-G-CORRECT-BEATS-RANDOM"]["pass"]
        and gates["S1-G-WRONG-DOC-DROPS"]["pass"]
        and gates["S1-G-POSITION-ONLY-DROPS"]["pass"]
        and gates["S1-G-REMOVAL-DROPS"]["pass"]
    )
    gates["s2_ready"] = gates["overall_pass"]
    return gates


def render_summary(result: dict[str, Any]) -> str:
    run_info = result["run_info"]
    gates = result["gates"]
    metrics = result["metrics"]
    lines = [
        "# LACE V2 S1 Summary",
        "",
        "## Run Info",
        "",
        f"- phase: `{run_info['phase']}`",
        f"- model: `{run_info['model_name']}`",
        f"- data: `{run_info['text_source']}`",
        f"- samples: `{run_info['sample_count']}`",
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
            lines.append(f"| `{gate_name}` | `{str(gate_value.get('pass')).lower()}` | `{json.dumps(gate_value.get('detail'), sort_keys=True)}` |")
    lines.extend(
        [
            f"| `overall_pass` | `{str(gates.get('overall_pass')).lower()}` | S1 readiness, not generation success. |",
            f"| `s2_ready` | `{str(gates.get('s2_ready')).lower()}` | Whether to move to skeleton-to-text training. |",
            "",
            "## Retrieval Metrics",
            "",
            "| Condition | Hit@1 | Hit@5 | MRR | Own Sim | Margin |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for condition_name in sorted(metrics):
        item = metrics[condition_name]
        lines.append(
            "| `{}` | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                condition_name,
                item["hit_at_1"],
                item["hit_at_5"],
                item["mrr"],
                item["own_similarity_mean"],
                item["margin_to_best_wrong_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "S1 checks retrieval-style skeleton use. It does not prove generation quality.",
            "If position or wrong-document controls stay strong, S2 needs tighter controls before training claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config: V2S1Config) -> dict[str, Any]:
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
        raise RuntimeError("Attention scores were not returned.")
    valid_mask = build_valid_mask(input_ids, attention_mask, tokenizer)
    token_idf = build_token_idf(input_ids, valid_mask)
    score_matrices = {
        "idf": scores_from_token_map(input_ids, valid_mask, token_idf),
        "attention_received": attention_scores,
        "position_prior": position_prior_scores(input_ids, valid_mask),
    }

    condition_texts, samples = build_condition_texts(config, tokenizer, input_ids, valid_mask, score_matrices)
    metrics: dict[str, dict[str, float]] = {}
    for condition_name, query_texts in condition_texts.items():
        query_embeddings, _ids, _mask, _scores = encode_texts(
            config,
            tokenizer,
            encoder,
            device,
            query_texts,
            return_attentions=False,
        )
        metrics[condition_name] = retrieval_metrics(query_embeddings, original_embeddings[: len(query_texts)])

    gates = evaluate_gates(metrics)
    result = {
        "run_info": {
            "phase": "v2_s1",
            "config": asdict(config),
            "text_source": text_source,
            "sample_count": len(texts),
            "model_name": config.model_name,
            "device": device,
            "keep_ratio": config.keep_ratio,
            "conditions": sorted(condition_texts.keys()),
        },
        "gates": gates,
        "metrics": metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "summary.md").write_text(render_summary(result), encoding="utf-8")
    with (output_dir / "retrieval_samples.jsonl").open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
    return result


def main() -> None:
    config = parse_args()
    result = run(config)
    print(render_summary(result))


if __name__ == "__main__":
    main()
