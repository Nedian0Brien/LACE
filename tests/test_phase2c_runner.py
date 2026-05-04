from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE2C_PATH = ROOT / "kaggle" / "phase2c" / "run_phase2c.py"


def load_phase2c_module():
    spec = importlib.util.spec_from_file_location("phase2c_runner", PHASE2C_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["phase2c_runner"] = module
    spec.loader.exec_module(module)
    return module


class Phase2CRunnerContractTest(unittest.TestCase):
    def test_defaults_include_positional_and_position_only_controls(self) -> None:
        phase2c = load_phase2c_module()

        config = phase2c.Phase2CConfig()

        self.assertEqual(config.output_dir, "/kaggle/working/lace_phase2c")
        self.assertIn("average_pool", config.forward_conditions)
        self.assertIn("average_pool_positional", config.forward_conditions)
        self.assertIn("position_only", config.forward_conditions)
        self.assertIn("gaussian_noise_positional", config.forward_conditions)
        self.assertEqual(config.positional_feature_size, 64)
        self.assertGreater(config.wrong_position_shift, 0)

    def test_condition_helpers_map_positional_conditions_to_base_latents(self) -> None:
        phase2c = load_phase2c_module()

        self.assertEqual(phase2c.base_condition_name("average_pool_positional"), "average_pool")
        self.assertEqual(phase2c.base_condition_name("gaussian_noise_positional"), "gaussian_noise")
        self.assertEqual(phase2c.base_condition_name("position_only"), "average_pool")
        self.assertTrue(phase2c.condition_uses_positional_encoding("average_pool_positional"))
        self.assertTrue(phase2c.condition_uses_positional_encoding("position_only"))
        self.assertFalse(phase2c.condition_uses_positional_encoding("average_pool"))

    def test_sinusoidal_position_features_have_expected_shape(self) -> None:
        phase2c = load_phase2c_module()

        features = phase2c.sinusoidal_position_features(max_length=8, feature_size=6)

        self.assertEqual(tuple(features.shape), (8, 6))
        self.assertAlmostEqual(float(features[0, 0]), 0.0)
        self.assertAlmostEqual(float(features[0, 1]), 1.0)

    def test_phase2c_gates_compare_positional_average_pool_against_controls(self) -> None:
        phase2c = load_phase2c_module()

        def stage(mse: float, cosine: float, decoder_delta: float, head_delta: float, meaningful: float = 0.0) -> dict:
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
                        "uses_positional_encoding": True,
                        "wrong_position_delta_mse": 0.002,
                        "wrong_position_delta_cosine": -0.002,
                    },
                    "generation_bridge": {
                        "delta_token_nll_vs_h0": decoder_delta,
                        "delta_token_head_nll_vs_h0": head_delta,
                    },
                    "generation_quality": {"meaningful_sample_rate": meaningful},
                },
            }

        metrics = {
            "strict_gate_min_evidence": 2,
            "results": {
                "average_pool": {"z1": stage(0.50, 0.30, 8.0, 1.4)},
                "average_pool_positional": {"z1": stage(0.40, 0.36, 7.5, 1.2, 0.25)},
                "position_only": {"z1": stage(0.90, 0.05, 12.0, 4.0)},
                "random_select": {"z1": stage(0.60, 0.10, 8.4, 1.5)},
                "gaussian_noise": {"z1": stage(0.44, 0.34, 7.8, 1.1)},
                "gaussian_noise_positional": {"z1": stage(0.43, 0.35, 7.7, 1.05)},
            },
            "h0_decoder_control": {"generation_quality": {"meaningful_sample_rate": 0.50}},
        }

        gates = phase2c.evaluate_phase2c_gates(metrics)

        self.assertTrue(gates["P2C-G-RUN"]["pass"])
        self.assertTrue(gates["P2C-G-POS-MSE"]["pass"])
        self.assertTrue(gates["P2C-G-POS-COS"]["pass"])
        self.assertTrue(gates["P2C-G-POS-DECODER-NLL"]["pass"])
        self.assertTrue(gates["P2C-G-POS-CONTROL"]["pass"])
        self.assertTrue(gates["phase3_candidate"])


if __name__ == "__main__":
    unittest.main()
