import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmbench_reward import (
    HarmBenchClassifier,
    HarmBenchSafetyReward,
    VALIDATION_CLASSIFIER,
    TEST_CLASSIFIER,
    behavior_and_context,
    classifier_prompt,
)


def record():
    return {
        "id": "harmbench:h1", "source": "harmbench", "is_harmful": True,
        "prompt": "context\n\nbehavior",
        "metadata": {"Behavior": "behavior", "ContextString": "context"},
    }


class Tokenizer:
    pad_token_id = 0
    answers = None
    width = 3

    def __init__(self):
        self.prompts = []

    def __call__(self, prompts, **kwargs):
        assert kwargs["truncation"] is False
        self.prompts.extend(prompts)
        return {
            "input_ids": torch.ones(len(prompts), self.width, dtype=torch.long),
            "attention_mask": torch.ones(len(prompts), self.width, dtype=torch.long),
        }

    def batch_decode(self, tokens, **kwargs):
        return self.answers[:len(tokens)]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(max_position_embeddings=32)

    def generate(self, input_ids, **kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs["do_sample"] is False and kwargs["max_new_tokens"] == 1
        return torch.cat([input_ids, torch.ones(len(input_ids), 1, dtype=torch.long)], dim=1)


class HarmBenchRewardTests(unittest.TestCase):
    def make_classifier(self, answers=("Yes", "No")):
        tokenizer, model = Tokenizer(), Model()
        tokenizer.answers = answers
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: model),
        )
        with patch.dict(sys.modules, {"transformers": transformers}):
            classifier = HarmBenchClassifier(VALIDATION_CLASSIFIER, "cpu", 32, 2)
        return classifier

    def test_judge_is_frozen_and_binary_labels_have_correct_direction(self):
        classifier = self.make_classifier()
        self.assertFalse(classifier.model.training)
        self.assertFalse(classifier.model.weight.requires_grad)
        scores = classifier(["behavior"] * 2, ["a", "b"], contexts=["context", ""])
        self.assertEqual(scores.tolist(), [1.0, 0.0])
        self.assertIn("[CONTEXT]:\ncontext", classifier.tokenizer.prompts[0])

    def test_invalid_judge_output_and_overlength_input_fail_explicitly(self):
        classifier = self.make_classifier(("Maybe",))
        with self.assertRaisesRegex(ValueError, "invalid label"):
            classifier(["b"], ["r"])
        classifier.tokenizer.width = 32
        with self.assertRaisesRegex(ValueError, "context limit"):
            classifier(["b"], ["r"])
        with self.assertRaisesRegex(ValueError, "behavior and context"):
            classifier(["b"], [])

    def test_reward_uses_fixed_weight_and_preserves_behavior_context(self):
        calls = []

        def judge(behaviors, responses, contexts):
            calls.append((behaviors, responses, contexts))
            return torch.tensor([1.0, 0.0])

        reward = HarmBenchSafetyReward(judge, 2.0)
        self.assertEqual(reward(["a", "b"], [record(), record()]).tolist(), [-2.0, 0.0])
        self.assertEqual(calls[0], (("behavior", "behavior"), ["a", "b"], ("context", "context")))
        for value in (0, -1, True, float("nan")):
            with self.assertRaises(ValueError):
                HarmBenchSafetyReward(judge, value)
        with self.assertRaises(ValueError):
            reward([], [])
        with self.assertRaises(ValueError):
            behavior_and_context(dict(record(), is_harmful=False))
        invalid_reward = HarmBenchSafetyReward(lambda *a, **k: torch.tensor([0.5]), 1)
        with self.assertRaisesRegex(ValueError, "binary label"):
            invalid_reward(["r"], [record()])

    def test_templates_match_model_families_and_separate_ids(self):
        self.assertNotEqual(VALIDATION_CLASSIFIER, TEST_CLASSIFIER)
        mistral = classifier_prompt("b", "r", "", "mistral")
        llama = classifier_prompt("b", "r", "", "llama")
        contextual = classifier_prompt("b", "r", "c", "llama")
        self.assertTrue(mistral.startswith("[INST] You are"))
        self.assertTrue(llama.startswith("[INST] <<SYS>>"))
        self.assertIn("Behavior:\nb", llama)
        self.assertIn("Benign or supportive", llama)
        self.assertIn("[CONTEXT]:\nc", contextual)
        self.assertTrue(contextual.endswith("Answer: [/INST]"))


if __name__ == "__main__":
    unittest.main()
