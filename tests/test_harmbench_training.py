import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune_cond_data_harmbench import (
    PromptLoader,
    SafetyGFNReward,
    prepare_conditioning,
    reward_inverse_temperature,
    sample_conditioned,
    split_prompts,
    training_loss,
)


class Config(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)


class Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __init__(self):
        self.text = None

    def __call__(self, text):
        self.text = text
        return SimpleNamespace(input_ids=[10, 11, 12])

    def batch_decode(self, tokens, skip_special_tokens):
        return [" ".join(map(str, row)) for row in tokens]


class FakeGFN:
    device = "cpu"
    prior_model = SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(length=8)))

    def sample_fwd(self, shape, **kwargs):
        tokens = torch.full(shape, 20, dtype=torch.long)
        projected = kwargs["projector"](tokens)
        torch.testing.assert_allclose(projected[:, :3], torch.tensor([[10, 11, 12]]).repeat(shape[0], 1))
        # Match the real sampler's unprojected final output.
        return {"x": tokens}


class PromptTests(unittest.TestCase):
    def test_harmbench_download_when_no_local_dataset(self):
        csv_bytes = (
            "Behavior,BehaviorID,FunctionalCategory,SemanticCategory,ContextString\n"
            "sample instruction,h1,standard,test,\n"
        ).encode()
        loader = PromptLoader(num_prompts_harmbench=1)
        with patch.object(Path, "is_file", return_value=False), patch(
            "finetune_cond_data_harmbench.urlopen", return_value=io.BytesIO(csv_bytes),
        ) as download:
            records = loader.load_harmbench()
        download.assert_called_once_with(PromptLoader.HARMBENCH_URL, timeout=60)
        self.assertEqual(records[0]["id"], "harmbench:h1")
        self.assertIs(records[0]["is_harmful"], True)

    def test_download_errors_propagate_and_explicit_missing_paths_are_not_replaced(self):
        with patch.object(Path, "is_file", return_value=False), patch(
            "finetune_cond_data_harmbench.urlopen", side_effect=URLError("offline"),
        ), self.assertRaises(URLError):
            PromptLoader().load_harmbench()
        with tempfile.TemporaryDirectory() as directory, patch(
            "finetune_cond_data_harmbench.urlopen",
        ) as download, self.assertRaises(FileNotFoundError):
            PromptLoader(harmbench_path=Path(directory) / "missing.csv").load_harmbench()
        download.assert_not_called()

    def test_loading_labels_context_metadata_and_safe_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            harmbench = Path(directory) / "harmbench.csv"
            xstest = Path(directory) / "xstest.csv"
            with harmbench.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "Behavior", "BehaviorID", "FunctionalCategory", "SemanticCategory", "ContextString",
                ])
                writer.writeheader()
                writer.writerow(dict(
                    Behavior="sample instruction", BehaviorID="h1", FunctionalCategory="standard",
                    SemanticCategory="test", ContextString="context",
                ))
            with xstest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "prompt", "type", "label"])
                writer.writeheader()
                writer.writerow(dict(id="s1", prompt="benign instruction", type="test", label="safe"))
                writer.writerow(dict(id="u1", prompt="excluded instruction", type="test", label="unsafe"))
            loader = PromptLoader(
                harmbench, xstest, num_prompts_harmbench=1, num_prompts_xstest=1,
            )
            harmful, benign = loader.load()
            self.assertEqual(harmful["prompt"], "context\n\nsample instruction")
            self.assertEqual(harmful["metadata"]["BehaviorID"], "h1")
            self.assertIs(harmful["is_harmful"], True)
            self.assertIs(benign["is_harmful"], False)
            self.assertEqual(benign["id"], "xstest:s1")
            loader.num_prompts_xstest = 2
            with self.assertRaisesRegex(ValueError, "only 1"):
                loader.load()
            xstest.write_text("id,prompt,type,label\ns1,test,test,unknown\n")
            with self.assertRaisesRegex(ValueError, "Unknown XSTest label"):
                loader.load()

    def test_split_is_deterministic_disjoint_and_has_both_labels(self):
        records = [
            {"id": str(index), "prompt": f"prompt {index}", "is_harmful": index < 5}
            for index in range(10)
        ]
        train, held_out = split_prompts(records, 0.2, 42)
        self.assertEqual((len(train), len(held_out)), (8, 2))
        self.assertEqual((train, held_out), split_prompts(list(reversed(records)), 0.2, 42))
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in held_out})
        self.assertEqual({row["is_harmful"] for row in held_out}, {True, False})
        invalid = [dict(record) for record in records]
        invalid[-1]["is_harmful"] = None
        with self.assertRaisesRegex(ValueError, "boolean"):
            split_prompts(invalid, 0.2, 42)

    def test_safety_only_split_and_disabled_benign_download(self):
        records = [
            {"id": str(i), "prompt": f"behavior {i}", "is_harmful": True}
            for i in range(4)
        ]
        with patch.object(PromptLoader, "load_harmbench", return_value=records), patch.object(
            PromptLoader, "load_xstest",
        ) as benign:
            self.assertEqual(PromptLoader(num_prompts_xstest=0).load(), records)
        benign.assert_not_called()
        train, held_out = split_prompts(records, 0.25, 42)
        self.assertEqual((len(train), len(held_out)), (3, 1))
        self.assertFalse({r["id"] for r in train} & {r["id"] for r in held_out})
        invalid = [dict(record) for record in records]
        invalid[-1]["prompt"] = invalid[0]["prompt"]
        with self.assertRaisesRegex(ValueError, "Duplicate prompt"):
            split_prompts(invalid, 0.2, 42)


class ConditioningTests(unittest.TestCase):
    def test_format_shape_and_projector_preserve_prompt_without_mutation(self):
        tokenizer = Tokenizer()
        length, shape, projector = prepare_conditioning("hello", tokenizer, 2, 5, 8, "cpu")
        self.assertEqual(tokenizer.text, "Instruction:\nhello\n\nResponse:\n")
        self.assertEqual((length, shape), (3, (2, 8)))
        tokens = torch.full(shape, 20, dtype=torch.long)
        projected = projector(tokens)
        torch.testing.assert_allclose(projected[:, 3:], tokens[:, 3:])
        self.assertTrue((tokens == 20).all())
        self.assertEqual(projected[0, :3].tolist(), [10, 11, 12])

    def test_context_limit_and_invalid_response_lengths(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            prepare_conditioning("hello", Tokenizer(), 1, 6, 8, "cpu")
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                prepare_conditioning("hello", Tokenizer(), 1, value, 8, "cpu")

    def test_final_projection_and_response_only_reward_integration(self):
        tokenizer = Tokenizer()
        result, prefix = sample_conditioned(FakeGFN(), "hello", tokenizer, 2, 5, False, 0.7)
        self.assertEqual(result["x"][0, :prefix].tolist(), [10, 11, 12])

        def scorer(prompts, responses):
            self.assertEqual(prompts, ["hello", "hello"])
            self.assertEqual(responses, ["20 20 20 20 20"] * 2)
            return torch.ones(2)

        reward = SafetyGFNReward(scorer, scorer, scorer, device="cpu")
        values = reward(result["x"], ["hello"] * 2, tokenizer, prefix, [True, False])
        torch.testing.assert_allclose(values, torch.tensor([-1.0, -1.0]))


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(
            reward_invtemp_start=0.9, reward_invtemp_end=1.2, reward_temp_sched_steps=10,
        )

    def test_temperature_endpoints_and_zero_duration(self):
        self.assertAlmostEqual(reward_inverse_temperature(self.cfg, 0), 0.9)
        self.assertAlmostEqual(reward_inverse_temperature(self.cfg, 5), 1.05)
        self.assertAlmostEqual(reward_inverse_temperature(self.cfg, 10), 1.2)
        self.assertAlmostEqual(reward_inverse_temperature(self.cfg, 20), 1.2)
        self.cfg.reward_temp_sched_steps = 0
        self.assertAlmostEqual(reward_inverse_temperature(self.cfg, 0), 1.2)

    def test_same_reward_scale_in_both_losses(self):
        result = {
            "posterior_log_probs": torch.zeros(2, 1),
            "prior_log_probs": torch.zeros(2, 1),
        }
        reward = torch.tensor([1.0, -1.0])
        for loss_type in ("tb", "vargrad"):
            self.assertAlmostEqual(training_loss(result, reward, 2, loss_type, 0, 0).item(), 2.0)
        with self.assertRaises(ValueError):
            training_loss(result, reward, 1, "invalid", 0, 0)

    def test_new_config_parameters_are_consumed_and_legacy_config_is_not_loaded(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "finetune_cond_data_harmbench.py").read_text()
        config = (root / "configs/ft_harmbench.yaml").read_text()
        keys = [line.split(":")[0] for line in config.splitlines() if line and not line.startswith("#")]
        for key in keys:
            if "_scorer_" not in key:
                self.assertIn(key, source)
        self.assertIn('"configs/ft_harmbench.yaml"', source)
        self.assertNotIn('"configs/ft.yaml"', source)
        for key in ("likelihood_model", "buffer_size", "reward_type", "num_prompts_alpaca", "reward_beta"):
            self.assertNotIn(key + ":", config)


if __name__ == "__main__":
    unittest.main()
