from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE3A_PATH = ROOT / "kaggle" / "phase3a" / "run_phase3a.py"


def load_phase3a_module():
    spec = importlib.util.spec_from_file_location("phase3a_runner", PHASE3A_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["phase3a_runner"] = module
    spec.loader.exec_module(module)
    return module


class Phase3ARunnerContractTest(unittest.TestCase):
    def test_defaults_define_generation_aware_objective_arms(self) -> None:
        phase3a = load_phase3a_module()

        config = phase3a.Phase3AConfig()

        self.assertEqual(config.output_dir, "/kaggle/working/lace_phase3a")
        self.assertIn("average_pool_rel_pos_recon", config.forward_conditions)
        self.assertIn("average_pool_rel_pos_tok005", config.forward_conditions)
        self.assertIn("average_pool_rel_pos_tok010", config.forward_conditions)
        self.assertIn("average_pool_rel_pos_tok020", config.forward_conditions)
        self.assertIn("position_only_rel_pos_tok010", config.forward_conditions)
        self.assertIn("gaussian_noise_abs_pos_tok010", config.forward_conditions)
        self.assertEqual(config.lambda_token, 0.0)

    def test_condition_helpers_split_source_condition_and_token_weight(self) -> None:
        phase3a = load_phase3a_module()
        config = phase3a.Phase3AConfig(lambda_token=0.03)

        self.assertEqual(phase3a.source_condition_name("average_pool_rel_pos_tok010"), "average_pool_rel_pos")
        self.assertEqual(phase3a.source_condition_name("gaussian_noise_abs_pos_recon"), "gaussian_noise_abs_pos")
        self.assertEqual(phase3a.base_condition_name("average_pool_rel_pos_tok010"), "average_pool")
        self.assertEqual(phase3a.base_condition_name("position_only_abs_rel_pos_tok010"), "average_pool")
        self.assertEqual(phase3a.base_condition_name("gaussian_noise_abs_pos_tok010"), "gaussian_noise")
        self.assertEqual(phase3a.condition_position_mode("average_pool_rel_pos_tok010"), "relative")
        self.assertEqual(phase3a.condition_position_mode("average_pool_abs_rel_pos_tok010"), "absolute_relative")
        self.assertTrue(phase3a.condition_is_position_only("position_only_rel_pos_tok010"))
        self.assertAlmostEqual(phase3a.condition_token_loss_weight("average_pool_rel_pos_tok005", config), 0.05)
        self.assertAlmostEqual(phase3a.condition_token_loss_weight("average_pool_rel_pos", config), 0.03)

    def test_relative_position_features_repeat_inside_pooling_blocks(self) -> None:
        phase3a = load_phase3a_module()

        features = phase3a.relative_position_features(max_length=8, target_tokens=2, feature_size=4)

        self.assertEqual(tuple(features.shape), (8, 4))
        self.assertTrue((features[0] == features[4]).all())
        self.assertFalse((features[0] == features[1]).all())

    def test_phase3a_gates_detect_token_objective_success(self) -> None:
        phase3a = load_phase3a_module()

        def stage(
            mse: float,
            cosine: float,
            decoder_delta: float,
            head_delta: float,
            meaningful: float,
            shuffled_head: float = 0.5,
        ) -> dict:
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
                        "shuffled_delta_token_nll": 0.5,
                        "shuffled_delta_token_head_nll": shuffled_head,
                    },
                    "generation_quality": {"meaningful_sample_rate": meaningful},
                },
            }

        metrics = {
            "strict_gate_min_evidence": 2,
            "results": {
                "average_pool_rel_pos_recon": {"z1": stage(0.25, 0.50, 5.8, 0.70, 0.0)},
                "average_pool_rel_pos_tok005": {"z1": stage(0.24, 0.54, 5.7, 0.55, 0.25)},
                "average_pool_rel_pos_tok010": {"z1": stage(0.23, 0.58, 5.6, 0.45, 0.25, 0.7)},
                "average_pool_rel_pos_tok020": {"z1": stage(0.30, 0.45, 6.2, 0.80, 0.0)},
                "average_pool_abs_rel_pos_tok010": {"z1": stage(0.22, 0.57, 5.7, 0.50, 0.25)},
                "position_only_rel_pos_tok010": {"z1": stage(0.32, 0.25, 6.5, 0.40, 0.0, 0.1)},
                "position_only_abs_rel_pos_tok010": {"z1": stage(0.31, 0.30, 6.3, 0.42, 0.0, 0.1)},
                "gaussian_noise_abs_pos_recon": {"z1": stage(0.15, 0.70, 4.0, 0.20, 0.25)},
                "gaussian_noise_abs_pos_tok010": {"z1": stage(0.14, 0.72, 4.1, 0.22, 0.25)},
            },
        }

        gates = phase3a.evaluate_phase3a_gates(metrics)

        self.assertEqual(gates["diagnostics"]["best_average_pool_rel_token_arm"], "average_pool_rel_pos_tok010")
        self.assertTrue(gates["P3A-G-RUN"]["pass"])
        self.assertTrue(gates["P3A-G-TOKEN-HEAD"]["pass"])
        self.assertTrue(gates["P3A-G-LATENT-USE"]["pass"])
        self.assertTrue(gates["P3A-G-CONTENT-CONTROL"]["pass"])
        self.assertTrue(gates["P3A-G-GENERATION"]["pass"])
        self.assertTrue(gates["phase3a_success"])


if __name__ == "__main__":
    unittest.main()
