from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE1_PATH = ROOT / "kaggle" / "phase1" / "run_phase1.py"


def load_phase1_module():
    spec = importlib.util.spec_from_file_location("phase1_runner", PHASE1_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["phase1_runner"] = module
    spec.loader.exec_module(module)
    return module


class Phase1RunnerContractTest(unittest.TestCase):
    def test_stage_tokens_parse_to_ordered_int_tuple(self) -> None:
        phase1 = load_phase1_module()

        self.assertEqual(phase1.parse_stage_tokens("64,32,16"), (64, 32, 16))

    def test_stage_pairs_build_reverse_expansion_targets(self) -> None:
        phase1 = load_phase1_module()

        pairs = phase1.build_stage_pairs(stage_tokens=(64, 32, 16), max_length=128)

        self.assertEqual(
            pairs,
            [
                ("z3_to_z2", 16, 32),
                ("z2_to_z1", 32, 64),
                ("z1_to_h0", 64, 128),
            ],
        )

    def test_default_dataset_source_is_wikitext2(self) -> None:
        phase1 = load_phase1_module()

        config = phase1.Phase1Config()

        self.assertTrue(config.use_hf_dataset)
        self.assertEqual(config.hf_dataset_name, "wikitext")
        self.assertEqual(config.hf_dataset_config, "wikitext-2-raw-v1")
        self.assertEqual(config.hf_dataset_split, "train")

    def test_extract_texts_filters_short_and_heading_rows(self) -> None:
        phase1 = load_phase1_module()
        rows = [
            {"text": ""},
            {"text": " = A Wiki Heading = "},
            {"text": "too short"},
            {"text": "This is a long enough WikiText row that should be kept for latent encoding."},
        ]

        texts = phase1.extract_texts_from_rows(rows, max_samples=4)

        self.assertEqual(texts, ["This is a long enough WikiText row that should be kept for latent encoding."])

    def test_gate_summary_marks_phase1_pass_when_core_metrics_improve(self) -> None:
        phase1 = load_phase1_module()
        metrics = {
            "train": {"initial_loss": 1.0, "final_loss": 0.62},
            "validation": {"initial_loss": 1.0, "final_loss": 0.7},
            "stages": {
                "z1": {"mse": 0.2, "cosine": 0.8, "variance_ratio": 0.7},
                "z2": {"mse": 0.3, "cosine": 0.7, "variance_ratio": 0.6},
                "z3": {"mse": 0.45, "cosine": 0.55, "variance_ratio": 0.4},
            },
            "perturbation": {"sensitivity": 0.12},
        }

        gates = phase1.evaluate_phase1_gates(metrics)

        self.assertTrue(gates["overall_pass"])
        self.assertTrue(gates["P1-G1"]["pass"])
        self.assertTrue(gates["P1-G2"]["pass"])
        self.assertTrue(gates["P1-G3"]["pass"])
        self.assertTrue(gates["P1-G4"]["pass"])


if __name__ == "__main__":
    unittest.main()
