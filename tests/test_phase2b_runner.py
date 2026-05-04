from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE2B_PATH = ROOT / "kaggle" / "phase2b" / "run_phase2b.py"


def load_phase2b_module():
    spec = importlib.util.spec_from_file_location("phase2b_runner", PHASE2B_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["phase2b_runner"] = module
    spec.loader.exec_module(module)
    return module


class Phase2BRunnerContractTest(unittest.TestCase):
    def test_defaults_are_validation_scale_not_smoke_scale(self) -> None:
        phase2b = load_phase2b_module()

        config = phase2b.Phase2BConfig()

        self.assertEqual(config.output_dir, "/kaggle/working/lace_phase2b")
        self.assertGreaterEqual(config.max_samples, 2048)
        self.assertGreaterEqual(config.epochs, 4)
        self.assertEqual(config.sigma_values, (0.05, 0.10, 0.15))
        self.assertTrue(config.use_matched_gaussian)
        self.assertTrue(config.enable_token_head_bridge)

    def test_resolve_stage_sigmas_prefers_matched_calibration_values(self) -> None:
        phase2b = load_phase2b_module()
        metadata = {
            "calibration": {
                "matched_sigmas": {
                    "z1": {"sigma_name": "sigma_0.05"},
                    "z2": {"sigma_name": "sigma_0.1"},
                    "z3": {"sigma_name": "sigma_0.15"},
                }
            }
        }

        sigmas = phase2b.resolve_stage_sigmas(
            forward_metadata=metadata,
            stage_names=("z1", "z2", "z3"),
            fallback_sigmas=(0.10, 0.20, 0.40),
            use_matched=True,
        )

        self.assertEqual(sigmas, {"z1": 0.05, "z2": 0.10, "z3": 0.15})

    def test_strict_gates_report_mse_cosine_nll_and_latent_use_separately(self) -> None:
        phase2b = load_phase2b_module()

        def stage(mse: float, cosine: float, decoder_delta: float, head_delta: float) -> dict:
            return {
                "validation": {"final_loss": mse},
                "final": {
                    "mse": mse,
                    "cosine": cosine,
                    "latent_use": {
                        "relative_perturbation_sensitivity": 0.02,
                        "ablation_delta_mse": 0.003,
                        "swap_delta_mse": 0.003,
                    },
                    "generation_bridge": {
                        "delta_token_nll_vs_h0": decoder_delta,
                        "delta_token_head_nll_vs_h0": head_delta,
                    },
                    "generation_quality": {"meaningful_sample_rate": 0.25},
                },
            }

        metrics = {
            "results": {
                "average_pool": {
                    "z1": stage(0.10, 0.90, 0.10, 0.08),
                    "z2": stage(0.11, 0.85, 0.12, 0.09),
                    "z3": stage(0.12, 0.80, 0.14, 0.10),
                },
                "random_select": {
                    "z1": stage(0.08, 0.60, 0.50, 0.30),
                    "z2": stage(0.09, 0.55, 0.55, 0.35),
                    "z3": stage(0.10, 0.50, 0.60, 0.40),
                },
                "gaussian_noise": {
                    "z1": stage(0.07, 0.50, 0.60, 0.35),
                    "z2": stage(0.08, 0.45, 0.65, 0.40),
                    "z3": stage(0.09, 0.40, 0.70, 0.45),
                },
            },
            "h0_decoder_control": {
                "generation_quality": {"meaningful_sample_rate": 0.50},
                "oracle_token_nll": 1.0,
            },
        }

        gates = phase2b.evaluate_phase2b_gates(metrics)

        self.assertFalse(gates["P2B-G-MSE"]["pass"])
        self.assertTrue(gates["P2B-G-COS"]["pass"])
        self.assertTrue(gates["P2B-G-DECODER-NLL"]["pass"])
        self.assertTrue(gates["P2B-G-TOKEN-HEAD-NLL"]["pass"])
        self.assertTrue(gates["P2B-G-USE"]["pass"])
        self.assertTrue(gates["phase3_candidate"])

    def test_generation_quality_summary_counts_only_meaningful_text(self) -> None:
        phase2b = load_phase2b_module()

        quality = phase2b.summarize_generation_samples(
            [
                {"generated": ""},
                {"generated": ",,,,,,"},
                {"generated": "A usable generated sentence with lexical content."},
            ]
        )

        self.assertEqual(quality["sample_count"], 3)
        self.assertAlmostEqual(quality["nonempty_sample_rate"], 2 / 3)
        self.assertAlmostEqual(quality["meaningful_sample_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
