import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune_cond_data_harmbench import HuggingFacePairScorer, SafetyGFNReward


class PairTokenizer:
    model_max_length = 512

    def __call__(self, prompts, responses, **kwargs):
        assert prompts == ["prompt"] * len(responses)
        assert not kwargs["truncation"] and kwargs["padding"]
        return {"input_ids": torch.ones((len(prompts), 2), dtype=torch.long)}


class SequenceClassifier:
    def __init__(self, logits):
        self.logits = logits
        self.is_training = True
        self.frozen = False
        self.config = types.SimpleNamespace(max_position_embeddings=512)

    def to(self, device):
        return self

    def eval(self):
        self.is_training = False

    def requires_grad_(self, value):
        self.frozen = not value

    def __call__(self, **inputs):
        return type("Output", (), {"logits": self.logits.expand(inputs["input_ids"].shape[0], -1)})()


class ModelLoader:
    model = None

    @classmethod
    def from_pretrained(cls, model_name):
        return cls.model


class TokenizerLoader:
    tokenizer = None

    @classmethod
    def from_pretrained(cls, model_name):
        return cls.tokenizer


class Tokenizer:
    def __init__(self):
        self.tokens = []

    def batch_decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        self.tokens = tokens
        return ["response"] * len(tokens)


class Scorer(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))
        self.batches = []

    def forward(self, prompts, responses):
        assert not torch.is_grad_enabled()
        assert not self.training
        self.batches.append((prompts, responses))
        return self.value.expand(len(prompts))


class SafetyRewardTests(unittest.TestCase):
    def setUp(self):
        self.harm, self.help, self.refusal = Scorer(0.25), Scorer(0.8), Scorer(0.1)
        self.reward = SafetyGFNReward(
            self.harm, self.help, self.refusal, lambda_harm=2,
            mu_refusal=3, beta=2, batch_size=2, device="cpu",
        )
        self.tokens = torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]])
        self.tokenizer = Tokenizer()

    def call_reward(self, **overrides):
        kwargs = dict(
            input_ids=self.tokens, prompts=["a", "b", "c"], tokenizer=self.tokenizer,
            prefix_lengths=[1, 2, 3], is_harmful=[True, False, False],
        )
        kwargs.update(overrides)
        return self.reward(**kwargs)

    def test_formula_decoding_freezing_and_batching(self):
        result = self.call_reward()
        torch.testing.assert_allclose(result, torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6, rtol=0)
        self.assertEqual(self.tokenizer.tokens, [[11, 12], [22], []])
        self.assertEqual([len(p) for p, _ in self.harm.batches], [2, 1])
        self.assertFalse(result.requires_grad)
        self.assertEqual(result.shape, (3,))
        self.assertEqual(result.device, self.tokens.device)
        for scorer in (self.harm, self.help, self.refusal):
            self.assertFalse(scorer.value.requires_grad)
            self.assertFalse(scorer.training)

    def test_scalar_prefix(self):
        self.call_reward(prefix_lengths=1)
        self.assertEqual(self.tokenizer.tokens, [[11, 12], [21, 22], [31, 32]])

    def test_invalid_inputs(self):
        for override in (
            dict(is_harmful=[True, None, False]), dict(prompts=["a"]),
            dict(prefix_lengths=-1), dict(prefix_lengths=True),
            dict(prefix_lengths=[1, 2]), dict(prefix_lengths=4),
            dict(input_ids=self.tokens.float()), dict(prompts="abc"),
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.call_reward(**override)

    def test_invalid_scorer_outputs(self):
        for scorer in (
            lambda p, r: torch.zeros(len(p), 1),
            lambda p, r: torch.full((len(p),), float("nan")),
            lambda p, r: [0.0] * len(p),
        ):
            self.reward.harm_scorer = scorer
            with self.assertRaises(ValueError):
                self.call_reward()
        self.reward.harm_scorer = self.harm
        self.reward.refusal_scorer = lambda p, r: torch.full((len(p),), 1.1)
        with self.assertRaises(ValueError):
            self.call_reward()

    def test_invalid_configuration(self):
        for kwargs in (
            dict(beta=-1), dict(lambda_harm=float("inf")), dict(mu_refusal=float("nan")),
            dict(batch_size=0), dict(batch_size=True),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SafetyGFNReward(self.harm, self.help, self.refusal, device="cpu", **kwargs)


class HuggingFacePairScorerTests(unittest.TestCase):
    def make_scorer(self, logits, mode="classification", label=1):
        model = SequenceClassifier(torch.tensor([logits], dtype=torch.float32))
        ModelLoader.model = model
        TokenizerLoader.tokenizer = PairTokenizer()
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = TokenizerLoader
        transformers.AutoModelForSequenceClassification = ModelLoader
        with patch.dict(sys.modules, {"transformers": transformers}):
            scorer = HuggingFacePairScorer("test-model", "cpu", mode=mode, label=label)
        return scorer, model

    def test_classification_returns_selected_class_probability_and_freezes_model(self):
        scorer, model = self.make_scorer([0.0, 2.0])
        scores = scorer(["prompt", "prompt"], ["response", "response"])
        torch.testing.assert_allclose(scores, torch.tensor([0.880797, 0.880797]), atol=1e-6, rtol=0)
        self.assertFalse(model.is_training)
        self.assertTrue(model.frozen)

    def test_regression_returns_scalar_logit(self):
        scorer, _ = self.make_scorer([1.25], mode="regression")
        torch.testing.assert_allclose(
            scorer(["prompt"], ["response"]), torch.tensor([1.25]), atol=1e-6, rtol=0
        )

    def test_single_logit_class_labels(self):
        positive, _ = self.make_scorer([2.0], label=1)
        negative, _ = self.make_scorer([2.0], label=0)
        torch.testing.assert_allclose(
            positive(["prompt"], ["response"]) + negative(["prompt"], ["response"]),
            torch.ones(1),
        )
        invalid, _ = self.make_scorer([2.0], label=2)
        with self.assertRaisesRegex(ValueError, "label 0 or 1"):
            invalid(["prompt"], ["response"])

    def test_invalid_model_output_is_rejected(self):
        scorer, _ = self.make_scorer([0.0, 1.0], mode="regression")
        with self.assertRaises(ValueError):
            scorer(["prompt"], ["response"])

    def test_scoring_never_silently_truncates(self):
        scorer, _ = self.make_scorer([0.0, 1.0])
        scorer.max_length = 1
        with self.assertRaisesRegex(ValueError, "context limit"):
            scorer(["prompt"], ["response"])

    def test_nonfinite_logits_are_rejected(self):
        scorer, _ = self.make_scorer([float("inf"), 1.0])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            scorer(["prompt"], ["response"])

    def test_response_decoding_stops_at_eos_and_removes_padding(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.eos_token_id = 99
        self.tokenizer.pad_token_id = 0
        reward = SafetyGFNReward(
            Scorer(0.25), Scorer(0.8), Scorer(0.1), device="cpu",
        )
        reward(
            torch.tensor([[10, 0, 11, 99, 12]]), ["prompt"], self.tokenizer, 1, [False],
        )
        self.assertEqual(self.tokenizer.tokens, [[11]])


if __name__ == "__main__":
    unittest.main()
