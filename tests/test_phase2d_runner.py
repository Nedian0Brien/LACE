from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE2D_PATH = ROOT / "kaggle" / "phase2d" / "run_phase2d.py"


def load_phase2d_module():
    spec = importlib.util.spec_from_file_location("phase2d_runner", PHASE2D_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["phase2d_runner"] = module
    spec.loader.exec_module(module)
    return module


class Phase2DRunnerContractTest(unittest.TestCase):
    def test_defaults_are_phase3_bridge_positional_confounds(self) -> None:
        phase2d = load_phase2d_module()

        config = phase2d.Phase2DConfig()

        self.assertEqual(config.output_dir, "/kaggle/working/lace_phase2d")
        self.assertIn("average_pool_abs_pos", config.forward_conditions)
        self.assertIn("average_pool_rel_pos", config.forward_conditions)
        self.assertIn("average_pool_abs_rel_pos", config.forward_conditions)
        self.assertIn("position_only_abs_rel_pos", config.forward_conditions)
        self.assertIn("gaussian_noise_abs_pos", config.forward_conditions)
        self.assertEqual(config.wrong_position_shift, 1)
        self.assertEqual(config.wrong_position_shifts, (1, 2, 4, 8, 16))

    def test_condition_helpers_separate_base_condition_and_position_mode(self) -> None:
        phase2d = load_phase2d_module()

        self.assertEqual(phase2d.base_condition_name("average_pool_abs_rel_pos"), "average_pool")
        self.assertEqual(phase2d.base_condition_name("position_only_rel_pos"), "average_pool")
        self.assertEqual(phase2d.base_condition_name("gaussian_noise_abs_pos"), "gaussian_noise")
        self.assertEqual(phase2d.condition_position_mode("average_pool_abs_pos"), "absolute")
        self.assertEqual(phase2d.condition_position_mode("average_pool_rel_pos"), "relative")
        self.assertEqual(phase2d.condition_position_mode("average_pool_abs_rel_pos"), "absolute_relative")
        self.assertTrue(phase2d.condition_is_position_only("position_only_abs_rel_pos"))

    def test_relative_position_features_repeat_inside_pooling_blocks(self) -> None:
        phase2d = load_phase2d_module()

        features = phase2d.relative_position_features(max_length=8, target_tokens=2, feature_size=4)

        self.assertEqual(tuple(features.shape), (8, 4))
        self.assertTrue((features[0] == features[4]).all())
        self.assertFalse((features[0] == features[1]).all())

    def test_phase2d_gates_mark_bridge_ready_from_confound_diagnostics(self) -> None:
        phase2d = load_phase2d_module()

        def stage(mse: float, cosine: float, decoder_delta: float, head_delta: float, shuffled_head: float = 0.2) -> dict:
            return {
                "validation": {"final_loss": mse},
                "final": {
                    "mse": mse,
                    "cosine": cosine,
                    "latent_use": {
                        "relative_perturbation_sensitivity": 0.02,
                        "ablation_delta_mse": 0.003,
                        "swap_delta_mse": 0.004,
                    },
                    "position_diagnostics": {
                        "wrong_position_delta_mse": 0.002,
                        "wrong_position_delta_cosine": -0.002,
                        "wrong_position_sweep_delta_mse": {"1": 0.001, "2": 0.002, "4": 0.003, "8": 0.004, "16": 0.005},
                    },
                    "generation_bridge": {
                        "delta_token_nll_vs_h0": decoder_delta,
                        "delta_token_head_nll_vs_h0": head_delta,
                        "shuffled_delta_token_nll": 0.2,
                        "shuffled_delta_token_head_nll": shuffled_head,
                    },
                    "generation_quality": {"meaningful_sample_rate": 0.0},
                },
            }

        metrics = {
            "strict_gate_min_evidence": 2,
            "results": {
                "average_pool": {"z1": stage(0.50, 0.30, 8.0, 1.4)},
                "average_pool_abs_pos": {"z1": stage(0.20, 0.45, 6.0, 0.8)},
                "average_pool_rel_pos": {"z1": stage(0.25, 0.50, 5.8, 0.7)},
                "average_pool_abs_rel_pos": {"z1": stage(0.18, 0.60, 5.0, 0.5, 0.3)},
                "position_only_abs_pos": {"z1": stage(0.30, 0.20, 7.0, 0.9, 0.05)},
                "position_only_rel_pos": {"z1": stage(0.35, 0.25, 7.0, 0.9, 0.05)},
                "position_only_abs_rel_pos": {"z1": stage(0.32, 0.35, 6.5, 0.8, 0.05)},
                "gaussian_noise": {"z1": stage(0.45, 0.36, 8.5, 0.9)},
                "gaussian_noise_abs_pos": {"z1": stage(0.15, 0.70, 4.0, 0.2)},
            },
        }

        gates = phase2d.evaluate_phase2d_gates(metrics)

        self.assertTrue(gates["P2D-G-RUN"]["pass"])
        self.assertTrue(gates["P2D-G-ABSREL-CONTENT"]["pass"])
        self.assertTrue(gates["P2D-G-SHUFFLED-LABEL"]["pass"])
        self.assertTrue(gates["phase3_bridge_ready"])


if __name__ == "__main__":
    unittest.main()
