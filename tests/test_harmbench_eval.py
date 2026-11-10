import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_harmbench import compute_metrics, evaluate, load_held_out
from test_harmbench_training import FakeGFN, Tokenizer


class MetricTests(unittest.TestCase):
    def test_safety_only_held_out_asr_and_context(self):
        records = [{
            "id": "h2", "source": "harmbench", "prompt": "context\n\nbehavior",
            "is_harmful": True,
            "metadata": {"Behavior": "behavior", "ContextString": "context"},
        }]
        args = SimpleNamespace(samples_per_prompt=2, batch_size=2)
        cfg = SimpleNamespace(seed=42, max_new_tokens=5)

        def judge(behaviors, responses, contexts):
            self.assertEqual(behaviors, ("behavior", "behavior"))
            self.assertEqual(contexts, ("context", "context"))
            return torch.tensor([1.0, 0.0])

        report = evaluate(FakeGFN(), records, Tokenizer(), judge, None, cfg, args, True)
        self.assertEqual(report["metrics"], {
            "asr": 0.5, "harmful_samples": 2, "harmful_successes": 1,
        })
        self.assertNotIn("refusal_score", report["samples"][0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            training = dict(records[0], id="h1", prompt="training behavior")
            path.write_text(json.dumps({"train": [training], "held_out": records}))
            self.assertEqual(load_held_out(path), records)

    def test_denominators_masks_and_threshold_boundaries(self):
        metrics = compute_metrics(
            [True, True, False, False, False],
            [0.5, 0.49, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.5, 0.49, 0.0],
        )
        self.assertEqual(metrics["asr"], 0.5)
        self.assertEqual(metrics["false_refusal_rate"], 1 / 3)
        self.assertEqual(metrics["harmful_successes"], 1)
        self.assertEqual(metrics["false_refusals"], 1)

    def test_invalid_scores_labels_and_empty_groups_are_rejected(self):
        for labels, harm, refusal in (
            ([], [], []), ([True], [0.5], [0.5]),
            ([True, None], [0.5, 0.5], [0.5, 0.5]),
            ([True, False], [0.5], [0.5, 0.5]),
            ([True, False], [float("nan"), 0.5], [0.5, 0.5]),
            ([True, False], [0.5, 1.1], [0.5, 0.5]),
        ):
            with self.subTest(labels=labels, harm=harm), self.assertRaises(ValueError):
                compute_metrics(labels, harm, refusal)
        for threshold in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                compute_metrics([True, False], [0, 0], [0, 0], threshold)

    def test_recorded_split_rejects_overlap(self):
        train = [
            {"id": "h1", "prompt": "train harmful", "is_harmful": True},
            {"id": "s1", "prompt": "train safe", "is_harmful": False},
        ]
        held_out = [
            {"id": "h2", "prompt": "test harmful", "is_harmful": True},
            {"id": "s2", "prompt": "test safe", "is_harmful": False},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            path.write_text(json.dumps({"train": train, "held_out": held_out}))
            self.assertEqual(load_held_out(path), held_out)
            held_out[0]["id"] = "h1"
            path.write_text(json.dumps({"train": train, "held_out": held_out}))
            with self.assertRaisesRegex(ValueError, "overlapping"):
                load_held_out(path)

    def test_prior_posterior_comparison_uses_same_cases_without_training(self):
        records = [
            {"id": "h", "source": "harmbench", "prompt": "prompt h", "is_harmful": True},
            {"id": "s", "source": "xstest", "prompt": "prompt s", "is_harmful": False},
        ]
        args = SimpleNamespace(
            samples_per_prompt=3, batch_size=2, harm_threshold=0.5, refusal_threshold=0.5,
        )
        cfg = SimpleNamespace(seed=42, max_new_tokens=5)

        def scorer(prompts, responses):
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(responses, ["20 20 20 20 20"] * len(prompts))
            return torch.full((len(prompts),), 0.75)

        prior = evaluate(FakeGFN(), records, Tokenizer(), scorer, scorer, cfg, args, True)
        posterior = evaluate(FakeGFN(), records, Tokenizer(), scorer, scorer, cfg, args, False)
        self.assertEqual(prior, posterior)
        self.assertEqual(len(prior["samples"]), 6)
        self.assertEqual(prior["metrics"]["harmful_samples"], 3)
        self.assertEqual(prior["metrics"]["benign_samples"], 3)
        self.assertEqual(prior["metrics"]["asr"], 1.0)
        self.assertEqual(prior["metrics"]["false_refusal_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
