import copy
import csv
import gzip
import io
import json
import math
import pickle
import random
from numbers import Real
from pathlib import Path
from typing import Callable, Sequence, Union
from urllib.request import urlopen

import numpy as np
import torch
from harmbench_reward import HarmBenchClassifier, HarmBenchSafetyReward


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HuggingFacePairScorer:
    """Score paired prompt/response inputs with a frozen classification model."""

    def __init__(self, model_name, device, mode="classification", label=1, max_length=512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if mode not in {"classification", "regression"}:
            raise ValueError("mode must be 'classification' or 'regression'")
        if isinstance(label, bool) or not isinstance(label, int) or label < 0:
            raise ValueError("label must be a nonnegative integer")
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        self.device = torch.device(device)
        self.mode = mode
        self.label = label
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def __call__(self, prompts, responses):
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")
        if not prompts:
            return torch.empty(0, device=self.device)
        encoded = self.tokenizer(
            list(prompts), list(responses), padding=True, truncation=False,
            return_tensors="pt",
        )
        limits = [self.max_length]
        for limit in (
            self.tokenizer.model_max_length,
            getattr(self.model.config, "max_position_embeddings", None),
        ):
            if isinstance(limit, int) and limit > 0:
                limits.append(limit)
        if encoded["input_ids"].shape[1] > min(limits):
            raise ValueError("Prompt/response pair exceeds scorer context limit; scoring would truncate text")
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            if logits.ndim != 2 or logits.shape[0] != len(prompts):
                raise ValueError("Scoring model must return logits with shape [B, classes]")
            if not torch.isfinite(logits).all():
                raise ValueError("Scoring model returned nonfinite logits")
            if self.mode == "regression":
                if logits.shape[1] != 1:
                    raise ValueError("Regression scorer must return exactly one logit per sample")
                return logits[:, 0].float()
            if logits.shape[1] == 1:
                if self.label not in (0, 1):
                    raise ValueError("A single-logit classifier requires label 0 or 1")
                positive = logits[:, 0].sigmoid().float()
                return positive if self.label == 1 else 1 - positive
            if self.label >= logits.shape[1]:
                raise ValueError(f"label index {self.label} is outside the model's {logits.shape[1]} classes")
            return logits.softmax(dim=-1)[:, self.label].float()


def decode_responses(input_ids, tokenizer, prefix_lengths):
    """Exclude fixed prefixes, EOS, padding, and special tokens."""
    responses = []
    for tokens, length in zip(input_ids, prefix_lengths):
        tokens = tokens[length:].detach().cpu().tolist()
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None and eos in tokens:
            tokens = tokens[:tokens.index(eos)]
        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is not None:
            tokens = [token for token in tokens if token != pad]
        responses.append(tokens)
    return tokenizer.batch_decode(responses, skip_special_tokens=True)


class SafetyGFNReward:
    """Return beta * (benign helpfulness - harmfulness - benign refusal) log rewards."""

    def __init__(
        self,
        harm_scorer: Callable,
        helpfulness_scorer: Callable,
        refusal_scorer: Callable,
        lambda_harm: float = 1.0,
        mu_refusal: float = 1.0,
        beta: float = 1.0,
        batch_size: int = 8,
        device: str = "cuda",
    ):
        for name, value in (("lambda_harm", lambda_harm), ("mu_refusal", mu_refusal), ("beta", beta)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.device = torch.device(device)
        for name, scorer in (
            ("harm_scorer", harm_scorer),
            ("helpfulness_scorer", helpfulness_scorer),
            ("refusal_scorer", refusal_scorer),
        ):
            if not callable(scorer):
                raise TypeError(f"{name} must be callable")
            if isinstance(scorer, torch.nn.Module):
                scorer.to(self.device)
                scorer.eval()
                scorer.requires_grad_(False)
            setattr(self, name, scorer)
        self.lambda_harm = float(lambda_harm)
        self.mu_refusal = float(mu_refusal)
        self.beta = float(beta)
        self.batch_size = batch_size

    def __call__(
        self,
        input_ids: torch.Tensor,
        prompts: Sequence[str],
        tokenizer,
        prefix_lengths: Union[int, Sequence[int]],
        is_harmful: Sequence[bool],
    ) -> torch.Tensor:
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must be a tensor with shape [B, L]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs")
        batch_size, sequence_length = input_ids.shape
        if batch_size == 0:
            raise ValueError("input_ids must contain at least one sample")
        if isinstance(prompts, str) or len(prompts) != batch_size or any(
            not isinstance(prompt, str) for prompt in prompts
        ):
            raise ValueError("prompts must contain one string per sample")
        if len(is_harmful) != batch_size or any(type(label) is not bool for label in is_harmful):
            raise ValueError("is_harmful must contain one boolean per sample")
        if isinstance(prefix_lengths, int) and not isinstance(prefix_lengths, bool):
            lengths = [prefix_lengths] * batch_size
        else:
            if isinstance(prefix_lengths, (bool, Real)):
                raise ValueError("prefix_lengths must be an integer or a sequence of integers")
            lengths = list(prefix_lengths)
        if len(lengths) != batch_size or any(
            isinstance(length, bool) or not isinstance(length, int) or not 0 <= length <= sequence_length
            for length in lengths
        ):
            raise ValueError("prefix_lengths must contain one integer in [0, L] per sample")

        responses = decode_responses(input_ids, tokenizer, lengths)
        if len(responses) != batch_size or any(not isinstance(response, str) for response in responses):
            raise ValueError("tokenizer must decode one response string per sample")

        def score(scorer, name, batch_prompts, batch_responses):
            if isinstance(scorer, torch.nn.Module):
                scorer.eval()
            values = scorer(batch_prompts, batch_responses)
            if not isinstance(values, torch.Tensor) or values.shape != (len(batch_prompts),):
                raise ValueError(f"{name} must return a tensor with shape [B]")
            values = values.detach().to(device=input_ids.device, dtype=torch.float32)
            if not torch.isfinite(values).all():
                raise ValueError(f"{name} returned nonfinite scores")
            if name == "refusal_scorer" and ((values < 0) | (values > 1)).any():
                raise ValueError("refusal_scorer must return scores in [0, 1]")
            return values

        rewards = []
        with torch.no_grad():
            for start in range(0, batch_size, self.batch_size):
                end = min(start + self.batch_size, batch_size)
                batch_prompts = list(prompts[start:end])
                batch_responses = responses[start:end]
                harm = score(self.harm_scorer, "harm_scorer", batch_prompts, batch_responses)
                helpfulness = score(self.helpfulness_scorer, "helpfulness_scorer", batch_prompts, batch_responses)
                refusal = score(self.refusal_scorer, "refusal_scorer", batch_prompts, batch_responses)
                benign = torch.tensor(
                    [not label for label in is_harmful[start:end]], device=input_ids.device, dtype=torch.bool
                )
                reward = self.beta * (
                    torch.where(benign, helpfulness, torch.zeros_like(helpfulness))
                    - self.lambda_harm * harm
                    - self.mu_refusal * torch.where(benign, refusal, torch.zeros_like(refusal))
                )
                if not torch.isfinite(reward).all():
                    raise ValueError("Combined log rewards are nonfinite")
                rewards.append(reward)
        return torch.cat(rewards).detach()


class PromptLoader:
    """Load HarmBench and explicitly safe XSTest prompts with source IDs."""

    HARMBENCH_URL = (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
        "data/behavior_datasets/harmbench_behaviors_text_all.csv"
    )
    XSTEST_URL = "https://huggingface.co/datasets/Paul/XSTest/resolve/main/xstest_prompts.csv"

    def __init__(
        self, harmbench_path=None, xstest_source=None,
        num_prompts_harmbench=400, num_prompts_xstest=100,
    ):
        for count in (num_prompts_harmbench, num_prompts_xstest):
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 0
            ):
                raise ValueError("Prompt counts must be nonnegative integers or None")
        self.num_prompts_harmbench = num_prompts_harmbench
        self.num_prompts_xstest = num_prompts_xstest
        self.harmbench_path = harmbench_path
        self.xstest_source = self.XSTEST_URL if xstest_source is None else xstest_source

    @staticmethod
    def _select(examples, num_prompts, source):
        if not examples:
            raise ValueError(f"{source} dataset is empty")
        if num_prompts is not None and len(examples) < num_prompts:
            raise ValueError(f"{source}: requested {num_prompts} prompts but only {len(examples)} are available")
        return examples[:num_prompts]

    @staticmethod
    def _read_csv(text, required, source, id_field):
        reader = csv.DictReader(io.StringIO(text))
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing {source} columns: {sorted(missing)}")
        examples = list(reader)
        for row_number, example in enumerate(examples, start=2):
            if any(not example.get(key) or not example[key].strip() for key in required - {"ContextString"}):
                raise ValueError(f"Missing {source} field in CSV row {row_number}")
            if "ContextString" in required and example["ContextString"] is None:
                raise ValueError(f"Missing {source} context in CSV row {row_number}")
        if len({example[id_field] for example in examples}) != len(examples):
            raise ValueError(f"{source} contains duplicate IDs")
        return examples

    def load_harmbench(self):
        if self.harmbench_path is None:
            relative = Path("data/harmbench_behaviors_text_all.csv")
            path = next(
                (parent / relative for parent in Path(__file__).resolve().parents
                 if (parent / relative).is_file()),
                None,
            )
        else:
            path = Path(self.harmbench_path).expanduser()
        if path is None:
            print(f"Downloading HarmBench prompts from {self.HARMBENCH_URL}", flush=True)
            with urlopen(self.HARMBENCH_URL, timeout=60) as response:
                text = response.read().decode("utf-8-sig")
        else:
            text = path.read_text(encoding="utf-8-sig")
        required = {"Behavior", "BehaviorID", "FunctionalCategory", "SemanticCategory", "ContextString"}
        examples = self._read_csv(text, required, "HarmBench", "BehaviorID")
        records = []
        examples = [example for example in examples if example["FunctionalCategory"] in {"standard", "contextual"}]
        for example in self._select(examples, self.num_prompts_harmbench, "HarmBench"):
            context = example["ContextString"].strip()
            behavior = example["Behavior"].strip()
            records.append({
                "id": f"harmbench:{example['BehaviorID']}",
                "source": "harmbench",
                "prompt": f"{context}\n\n{behavior}" if context else behavior,
                "is_harmful": True,
                "metadata": example,
            })
        return records

    def load_xstest(self):
        source = str(self.xstest_source)
        if source.startswith(("https://", "http://")):
            with urlopen(source, timeout=60) as response:
                text = response.read().decode("utf-8-sig")
        else:
            text = Path(source).expanduser().read_text(encoding="utf-8-sig")
        examples = self._read_csv(text, {"id", "prompt", "type", "label"}, "XSTest", "id")
        for example in examples:
            if example["label"] not in {"safe", "unsafe"}:
                raise ValueError(f"Unknown XSTest label: {example['label']!r}")
        examples = [example for example in examples if example["label"] == "safe"]
        return [{
            "id": f"xstest:{example['id']}",
            "source": "xstest",
            "prompt": example["prompt"],
            "is_harmful": False,
            "metadata": example,
        } for example in self._select(examples, self.num_prompts_xstest, "XSTest (safe)")]

    def load(self):
        return self.load_harmbench() + (self.load_xstest() if self.num_prompts_xstest != 0 else [])


def load_prompt_splits(cfg):
    records = PromptLoader(
        harmbench_path=cfg.harmbench_path, xstest_source=cfg.xstest_source,
        num_prompts_harmbench=cfg.num_prompts_harmbench,
        num_prompts_xstest=cfg.num_prompts_xstest,
    ).load()
    return split_prompts(records, cfg.eval_fraction, cfg.seed)


def split_prompts(records, eval_fraction, seed):
    if isinstance(eval_fraction, bool) or not isinstance(eval_fraction, Real) or not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1")
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Prompt IDs must be unique")
    if len({record["prompt"].strip() for record in records}) != len(records):
        raise ValueError("Duplicate prompt text must be removed before splitting")
    if any(type(record["is_harmful"]) is not bool for record in records):
        raise ValueError("All prompts must have known boolean safety labels")
    rng = random.Random(seed)
    train, held_out = [], []
    for label in (True, False):
        group = sorted(
            (record for record in records if record["is_harmful"] == label),
            key=lambda record: record["id"],
        )
        if not group:
            continue
        if len(group) < 2:
            raise ValueError("At least two harmful and two benign prompts are required")
        rng.shuffle(group)
        count = max(1, min(len(group) - 1, int(len(group) * eval_fraction)))
        held_out.extend(group[:count])
        train.extend(group[count:])
    return train, held_out


def make_scorer(cfg, name):
    model_name = cfg[f"{name}_scorer_model"]
    if not model_name:
        raise ValueError(f"{name}_scorer_model must be a compatible sequence-classification model ID or path")
    return HuggingFacePairScorer(
        model_name, cfg.device, mode=cfg[f"{name}_scorer_mode"],
        label=cfg[f"{name}_scorer_label"], max_length=cfg.scorer_max_length,
    )


def prepare_conditioning(prompt, tokenizer, batch_size, max_new_tokens, context_limit, device):
    for name, value in (
        ("batch_size", batch_size), ("max_new_tokens", max_new_tokens), ("context_limit", context_limit),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    prefix_ids = tokenizer(f"Instruction:\n{prompt}\n\nResponse:\n").input_ids
    sequence_length = len(prefix_ids) + max_new_tokens
    if sequence_length > context_limit:
        raise ValueError(
            f"Prompt ({len(prefix_ids)} tokens) + response ({max_new_tokens}) exceeds "
            f"model context limit ({context_limit}); reduce max_new_tokens or shorten the prompt"
        )
    prefix = torch.tensor(prefix_ids, device=device)[None].repeat(batch_size, 1)

    def projector(tokens):
        tokens = tokens.clone()
        tokens[:, :len(prefix_ids)] = prefix
        return tokens

    return len(prefix_ids), (batch_size, sequence_length), projector


def sample_conditioned(gfn, prompt, tokenizer, batch_size, max_new_tokens, sample_from_prior, detach_prob):
    prefix_length, shape, projector = prepare_conditioning(
        prompt, tokenizer, batch_size, max_new_tokens,
        gfn.prior_model.config.model.length, gfn.device,
    )
    results = gfn.sample_fwd(
        shape, sample_from_prior=sample_from_prior, detach_prob=detach_prob,
        tokenizer=tokenizer, projector=projector,
    )
    # sample_fwd projects before each transition, but not after its final denoising step.
    results["x"] = projector(results["x"])
    return results, prefix_length


def reward_inverse_temperature(cfg, step):
    for name in ("reward_invtemp_start", "reward_invtemp_end"):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    steps = cfg.reward_temp_sched_steps
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("reward_temp_sched_steps must be a nonnegative integer")
    fraction = min(step / steps, 1.0) if steps else 1.0
    return cfg.reward_invtemp_start + (cfg.reward_invtemp_end - cfg.reward_invtemp_start) * fraction


def training_loss(results, logreward, inverse_temperature, loss_type, log_z, cutoff):
    term = (
        results["posterior_log_probs"].sum(1)
        - results["prior_log_probs"].sum(1)
        - inverse_temperature * logreward
    )
    if loss_type == "tb":
        residual = term + log_z
    elif loss_type == "vargrad":
        residual = term + term.mean()
    else:
        raise ValueError("loss_type must be 'tb' or 'vargrad'")
    return 0.5 * (residual.square() - cutoff).relu().mean()


def main():
    import wandb
    from gfn import PosteriorPriorGFN
    from load_model import load_model
    from omegaconf import OmegaConf
    from tqdm import tqdm
    from transformers import GPT2TokenizerFast, get_constant_schedule_with_warmup

    conf = OmegaConf.load(Path(__file__).parent / "configs/ft_harmbench.yaml")
    OmegaConf.set_struct(conf, True)
    cfg = OmegaConf.merge(conf, OmegaConf.from_cli())
    if cfg.loss_type != "tb":
        raise ValueError("The safety-only pilot requires trajectory balance (loss_type=tb)")
    if cfg.num_prompts_xstest != 0:
        raise ValueError("The safety-only pilot trains on HarmBench only; set num_prompts_xstest=0")
    if cfg.reward_invtemp_start != cfg.reward_invtemp_end:
        raise ValueError("The safety-only pilot requires a fixed reward inverse temperature")
    if cfg.harm_scorer_model == cfg.test_scorer_model:
        raise ValueError("Training and test classifiers must be separate")
    for name in ("num_steps", "batch_size", "grad_accumulation", "sampling_len", "reward_batch_size"):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    reward_inverse_temperature(cfg, 0)
    output_dir = Path(cfg.save_dir).expanduser() / cfg.exp_name
    if any((output_dir / name).exists() for name in ("run_config.yaml", "prompt_split.json")):
        raise FileExistsError(f"Run directory already contains a recorded run: {output_dir}; choose a new exp_name")
    seed_everything(cfg.seed)
    data, held_out = load_prompt_splits(cfg)
    reward_model = HarmBenchSafetyReward(
        HarmBenchClassifier(
            cfg.harm_scorer_model, cfg.device, cfg.scorer_max_length, cfg.reward_batch_size,
        ),
        safety_weight=cfg.safety_weight,
    )
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    device = torch.device(cfg.device)
    prior_model, graph, noise = load_model(cfg.model_name, device)
    for example in data + held_out:
        prepare_conditioning(
            example["prompt"], tokenizer, 1, cfg.max_new_tokens,
            prior_model.config.model.length, device,
        )
    posterior_model = copy.deepcopy(prior_model)
    prior_model.eval()
    prior_model.requires_grad_(False)
    posterior_model.eval()
    gfn = PosteriorPriorGFN(
        prior_model, posterior_model, reward_model, graph, noise,
        "analytic", 0, cfg.sampling_len, 1e-5, device,
    )
    params = [{"params": posterior_model.parameters(), "lr": cfg.learning_rate}]
    if cfg.loss_type == "tb":
        params.append({"params": [gfn.logZ], "lr": cfg.learning_rate_Z})
    opt = torch.optim.Adam(params)
    scheduler = get_constant_schedule_with_warmup(opt, cfg.warmup_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "run_config.yaml")
    (output_dir / "prompt_split.json").write_text(json.dumps({
        "train": data, "held_out": held_out,
    }, indent=2), encoding="utf-8")
    wandb.init(project="discrete_diff_ft", config=OmegaConf.to_container(cfg, resolve=True), mode=cfg.wandb_mode)

    opt.zero_grad()
    progress = tqdm(range(cfg.num_steps))
    for step in progress:
        example = data[np.random.choice(len(data))]
        results, prefix_length = sample_conditioned(
            gfn, example["prompt"], tokenizer, cfg.batch_size, cfg.max_new_tokens,
            sample_from_prior=step % 5 == 0, detach_prob=0.7,
        )
        responses = decode_responses(results["x"], tokenizer, [prefix_length] * cfg.batch_size)
        logreward = reward_model(responses, [example] * cfg.batch_size)
        inverse_temperature = reward_inverse_temperature(cfg, step)
        forward_loss = training_loss(
            results, logreward, inverse_temperature, cfg.loss_type, gfn.logZ, cfg.cutoff,
        )
        if not torch.isfinite(forward_loss):
            raise ValueError(f"Nonfinite training loss at step {step}")
        with torch.no_grad():
            residuals = (
                results["posterior_log_probs"].sum(1)
                - results["prior_log_probs"].sum(1)
                + gfn.logZ
                - inverse_temperature * logreward
            )
        (forward_loss / cfg.grad_accumulation).backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(posterior_model.parameters(), 0.1)
        with (output_dir / "rollouts.jsonl").open("a", encoding="utf-8") as handle:
            for index, response in enumerate(responses):
                # The fixed binary reward is log R = -safety_weight * harmful_label.
                harmful = bool((-logreward[index] / reward_model.safety_weight).item())
                diagnostic = {
                    "step": step, "sample_index": index, "prompt_id": example["id"],
                    "prompt": example["prompt"], "response": response,
                    "judge_model": cfg.harm_scorer_model,
                    "judge_label": "Yes" if harmful else "No",
                    "log_reward": logreward[index].item(),
                    "tb_residual": residuals[index].item(),
                    "loss": forward_loss.item(),
                    "gradient_norm_before_clip": gradient_norm.item(),
                    "sample_from_prior": step % 5 == 0,
                    "sampling_len": cfg.sampling_len,
                    "max_new_tokens": cfg.max_new_tokens,
                }
                line = json.dumps(diagnostic, allow_nan=False)
                handle.write(line + "\n")
                progress.write(line)
        if (step + 1) % cfg.grad_accumulation == 0:
            opt.step()
            opt.zero_grad()
            scheduler.step()
        mean_reward = logreward.mean().item()
        progress.set_description(f"Loss: {forward_loss.item():.3f}, Log Reward: {mean_reward:.3f}")
        wandb.log({
            "fwd_loss": forward_loss.item(), "fwd_logreward": mean_reward,
            "reward_invtemp": inverse_temperature,
        })
        if step % 50 == 0 or step + 1 == cfg.num_steps:
            with gzip.open(output_dir / f"posterior_{step}.pkl.gz", "wb") as handle:
                pickle.dump(posterior_model, handle)
    wandb.finish()


if __name__ == "__main__":
    main()
