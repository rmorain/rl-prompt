from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    pipeline,
)

from rlprompt.rewards import BaseReward

SUPPORTED_LEFT_TO_RIGHT_LMS = [
    "distilgpt2",
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
]
SUPPORTED_MASK_LMS = ["distilroberta-base", "roberta-base", "roberta-large"]


class SentimentRewardModel:
    """
    A subclass of `RewardModel` that calculates a reward using the default
    Hugging Face sentiment classifier.
    """

    def __init__(
        self,
        model: Optional[
            str
        ] = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device: Optional[Union[torch.device, int]] = -1,
        kwargs: Optional[Dict] = {
            "top_k": 2,
            "function_to_apply": None,
            "batch_size": 16,
        },
    ):
        """
        Initialize the sentiment reward model.

        Args:
          model (Optional[str] =
          "distilbert/distilbert-base-uncased-finetuned-sst-2-english"):
            The Hugging Face model identifier for sentiment analysis.
          device (Optional[Union[torch.device, int]] = -1):
            The device to use for the sentiment analysis pipeline. This can be
            either a `torch.device` object or an integer representing the GPU
            index (if using CUDA with PyTorch).
          kwargs (Optional[Dict]): Additional keyword arguments to be passed
            to the Hugging Face `pipeline` function.
        """
        self.reward_model = pipeline(
            "sentiment-analysis",
            model=model,
            device=device,
        )
        self.kwargs = kwargs
        self.device = device

    def __call__(
        self, input_string: Union[str, List[str]]
    ) -> List[Optional[List[float]]]:
        """
        Assigns scores for each input string.

        Args:
            input_string (Union[str, List[str]]): A List (batch size) of strings to be
                evaluated.

        Returns:
            List[Optional[List[float]]]: A List (batch size) of Lists (number of
                classes) containing scores.
        """
        prediction = self.reward_model(input_string, **self.kwargs)
        scores = self.process_sentiment_output(prediction)
        return scores

    def to(self, device: torch.device) -> None:
        """
        Set the device for the RewardModel.

        Args:
            device (torch.device): The device to use for the RewardModel.

        Returns:
            None
        """
        self.reward_model.model = self.reward_model.model.to(device)
        self.reward_model.device = device
        self.device = device
        return self

    def process_sentiment_output(self, model_output):
        return [
            [
                next(item["score"] for item in sample if item["label"] == "NEGATIVE"),
                next(item["score"] for item in sample if item["label"] == "POSITIVE"),
            ]
            for sample in model_output
        ]


class SentimentReward(BaseReward):
    def __init__(self, task_lm, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_lm = AutoModelForCausalLM.from_pretrained(task_lm).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(task_lm, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.reward_model = SentimentRewardModel().to(self.device)
        self.config = config
        self.continuation_length = 20
        self.continuation_max_str_length = 400

    def forward(self, batch, output_tokens=None, to_tensor=None, mode=None):
        rewards_log = {}
        prompts = batch["prompt"]
        prefix_prompt = []
        for out, prompt in zip(output_tokens, prompts):
            prefix = self.tokenizer.convert_tokens_to_ids(out)
            prefix = torch.tensor(prefix)
            # prefix = self.tokenizer(out, return_tensors="pt").input_ids.flatten()
            pre_pro = torch.cat((prefix, prompt), dim=0)
            prefix_prompt.append(self.tokenizer.decode(pre_pro))
        rewards, continuations = self.compute_reward(
            prefix_prompt,
            [self.task_lm],
            [self.tokenizer],
            [self.reward_model],
            self.config,
        )
        targets = torch.tensor(batch["target"])
        rewards_log["accuracy"] = (rewards.argmax(-1) == targets).float().mean().item()
        target_rewards = torch.gather(
            rewards.mean(0).mean(0), -1, targets.unsqueeze(1)
        ).squeeze()
        rewards_log["reward"] = target_rewards.mean().item()
        # target_rewards = [r for r in target_rewards]
        return target_rewards, rewards_log

    def compute_reward(
        self,
        prefix_prompt: List[torch.LongTensor],
        base_models: List[AutoModelForCausalLM],
        tokenizers: List[AutoTokenizer],
        reward_models,
        config,
    ) -> List[float]:
        """
        Compute a reward for each (prompt, continuation) pair.

        Args:
            prompts (List[str]): A list (batch size) of prompt strings.
            prefix_prompt (List[torch.LongTensor]): A list (batch size) of prefix tensors
                prepended to prompt tensors.
            base_models (AutoModelForCausalLM): A list of language models to be controlled
                by the policy model.
            tokenizers (List[AutoTokenizer]): A list of tokenizers corresponding to each
                base model.
            reward_models (List[RewardModel]): A list of reward models.
            config (TrainingConfig): The configuration object containing hyperparameters.

        Returns:
            List[float]: A list (batch size) of reward values.
        """
        continuation = self.generate_continuation(
            prefix_prompt, base_models, tokenizers, config
        )
        # base_model_perplexity = perplexity(prompts, continuation, base_models, tokenizers)
        scores = self.compute_scores_continuation_only(
            continuation,
            reward_models,
        )
        return scores, continuation

    def generate_continuation(
        self,
        prefix_prompt: List[str],
        base_models: List[AutoModelForCausalLM],
        tokenizers: List[AutoTokenizer],
        config,
    ) -> List[List[str]]:
        """
        Generates a continuation from a (prefix, prompt) pair for each base model.

        Args:
            prefix_prompt (List[str]): A list (batch size) of prefix strings
                prepended to prompt strings.
            base_models (AutoModelForCausalLM): A list of language models to be controlled
                by the policy model.
            tokenizers (List[AutoTokenizer]): A list of tokenizers corresponding to each
                base model.
            config (TrainingConfig): The configuration object containing hyperparameters.

        Returns:
            List[List[str]]: A list (len(base_models)) of lists (batch size) of
                continuation strings.
        """
        gen_kwargs = {
            "min_length": -1,
            # "top_p": 0.9,
            "top_k": 0.0,
            "do_sample": False,
            "output_scores": True,
        }
        continuations = []
        with torch.no_grad():
            for model, tokenizer in zip(base_models, tokenizers):
                inputs = tokenizer(prefix_prompt, padding=True, return_tensors="pt")
                input_ids = inputs.input_ids.to(model.device)
                attention_mask = inputs.attention_mask.to(model.device)
                prefix_prompt_continuation = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.continuation_length,
                    pad_token_id=model.config.eos_token_id,
                    **gen_kwargs,
                )
                continuation_ids = [
                    cont_ids[len(pp_ids) :]
                    for cont_ids, pp_ids in zip(prefix_prompt_continuation, input_ids)
                ]
                continuation_str = tokenizer.batch_decode(
                    continuation_ids,
                    skip_special_tokens=False,
                )
                continuation = [
                    s[: self.continuation_max_str_length] for s in continuation_str
                ]
                if not all(continuation):
                    print("ERROR: Missing continuations")
                    print(continuation)
                    print(prefix_prompt)
                continuations.append(continuation)
        return continuations

    def compute_scores_continuation_only(
        self,
        continuations: List[List[torch.LongTensor]],
        reward_models,
    ) -> List[List[float]]:
        scores = []
        for base_model_continuations in continuations:
            for model in reward_models:
                s = model(base_model_continuations)
                scores.append(s)
        scores_tensor = torch.tensor(scores).reshape(
            len(reward_models),
            len(continuations),
            len(continuations[0]),
            len(scores[0][0]),
        )
        return scores_tensor


class PromptedClassificationReward(BaseReward):
    def __init__(
        self,
        task_lm: str,
        is_mask_lm: Optional[bool],
        compute_zscore: bool,
        incorrect_coeff: float,  # lambda_1 in paper
        correct_coeff: float,  # lambda_2 in paper
        num_classes: int,
        verbalizers: List[str],
        template: Optional[str],
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_lm = task_lm
        if is_mask_lm is None:
            # If False, then treat as left-to-right LM
            self.is_mask_lm = True if "bert" in self.task_lm else False
        else:
            self.is_mask_lm = is_mask_lm
        print("Task LM:", self.task_lm)
        if self.is_mask_lm:
            assert self.task_lm in SUPPORTED_MASK_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(self.task_lm)
            self._generator = AutoModelForMaskedLM.from_pretrained(self.task_lm).to(
                self.device
            )
        else:
            assert self.task_lm in SUPPORTED_LEFT_TO_RIGHT_LMS
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.task_lm, pad_token="<|endoftext|>"
            )
            self._generator = GPT2LMHeadModel.from_pretrained(self.task_lm).to(
                self.device
            )
            self._generator.config.pad_token_id = self._tokenizer.pad_token_id

        self.compute_zscore = compute_zscore
        self.incorrect_coeff = incorrect_coeff
        self.correct_coeff = correct_coeff
        self.num_classes = num_classes
        self.verbalizers = verbalizers
        print("Verbalizers:", self.verbalizers)
        self.verbalizer_ids = [
            self._tokenizer.convert_tokens_to_ids(v) for v in self.verbalizers
        ]
        if template is None:
            self.template = self.load_default_template()  # prompt templates
        else:
            self.template = template
        self._counter = 0

    def load_default_template(self) -> str:
        if self.is_mask_lm:
            mask_token = self._tokenizer.mask_token
            template = f"{{sentence_1}} {{prompt}} {mask_token} ."
        else:
            # Template for left-to-right LMs like GPT-2
            template = "{sentence_1} {prompt}"
        return template

    def forward(
        self,
        source_texts: List[str],
        class_labels: List[int],
        output_tokens: List[List[str]],
        to_tensor: bool,
        mode: str,
    ) -> Tuple[Union[List[float], torch.Tensor], Dict[str, Any]]:
        assert mode in ["train", "infer"]

        if mode == "train":
            self._counter += 1

        # Process prompts and verbalizer indices
        prompt_tokens = output_tokens
        prompt_strings = self._convert_tokens_to_string(prompt_tokens)
        batch_size = len(source_texts)

        rewards: List[torch.Tensor] = []
        input_rewards: Dict[str, List[float]] = defaultdict(list)
        quantities_to_log: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for i, prompt in enumerate(prompt_strings):
            # Compute LM logits
            current_prompts = [prompt for _ in source_texts]
            formatted_templates = self._format_prompts(source_texts, current_prompts)
            all_logits = self._get_logits(formatted_templates)
            # [batch_size, vocab_size]
            class_probs = torch.softmax(all_logits[:, self.verbalizer_ids], -1)
            # [batch_size, num_classes]

            # Get label and maximum not-label probabilities
            label_probs = class_probs[range(batch_size), class_labels]
            # [batch_size, 1]
            not_label_probs = torch.where(
                class_probs == label_probs.unsqueeze(1),
                torch.Tensor([-1]).to(self.device),
                class_probs,
            )
            # [batch_size, num_classes]
            max_not_label_probs, _ = torch.max(not_label_probs, -1)
            # [batch_size, 1]

            # Compute piecewise gap reward
            gap = label_probs - max_not_label_probs
            correct = (gap > 0).long()
            gap_rewards = gap * (
                self.correct_coeff * correct + self.incorrect_coeff * (1 - correct)
            )
            reward = gap_rewards.mean().detach()

            # Log quantities such as accuracy and class-wise reward
            acc = correct.float().mean()
            quantities_to_log["acc"] = acc
            for c in range(self.num_classes):
                class_idx = np.array(class_labels) == c
                class_rewards = gap_rewards[class_idx]
                quantities_to_log[f"gap_reward_class_{c}"].append(
                    class_rewards.mean().item()
                )
            quantities_to_log["gap_reward"].append(reward.item())
            rewards.append(reward)

            # keep track of rewards for z-score normalization
            input_rewards["z"] += [reward.item()]

            # Print examples
            print_strs = [self._counter, "|", prompt, "\n"]
            for c in range(self.num_classes):
                class_example_idx = np.where(np.array(class_labels) == c)[0][0]
                class_example = formatted_templates[class_example_idx]
                class_example_probs = class_probs[class_example_idx, :].tolist()
                class_example_probs = [round(prob, 2) for prob in class_example_probs]
                print_strs += [
                    "Class",
                    c,
                    "Example:",
                    class_example,
                    "|",
                    "Probs:",
                    class_example_probs,
                    "\n",
                ]
            print_strs += [
                "Accuracy:",
                acc.item(),
                "|",
                "Reward:",
                round(reward.item(), 2),
            ]
            print(*print_strs)
        rewards_tensor = torch.stack(rewards)

        # z-score normalization (2nd stage)
        if mode == "train" and self.compute_zscore:
            input_reward_means = {k: np.mean(v) for k, v in input_rewards.items()}
            input_reward_stds = {k: np.std(v) for k, v in input_rewards.items()}
            # not source strings
            idx_means = torch.tensor(input_reward_means["z"]).float()
            idx_stds = torch.tensor(input_reward_stds["z"]).float()
            rewards_tensor = (rewards_tensor - idx_means) / (idx_stds + 1e-4)
            for i in range(rewards_tensor.size(0)):
                quantities_to_log["resized_reward"].append(rewards_tensor[i].item())
        elif mode == "infer":  # Optional: Predict Val Prompts
            score = rewards_tensor.mean().item()
            print("Our Prompt:")
            print(prompt_strings, score)

        rewards_log = dict(
            (reward_key, torch.mean(torch.tensor(reward_vals)))
            for reward_key, reward_vals in quantities_to_log.items()
        )

        if to_tensor is True:
            return rewards_tensor, rewards_log
        else:
            return rewards_tensor.tolist(), rewards_log

    # Adapted from
    # https://huggingface.co/docs/transformers/v4.21.1/en/task_summary#masked-language-modeling
    def _get_mask_token_index(self, input_ids: torch.Tensor) -> np.ndarray:
        mask_token_index = torch.where(input_ids == self._tokenizer.mask_token_id)[1]
        return mask_token_index

    def ensure_exactly_one_mask_token(
        self, model_inputs: Dict[str, torch.Tensor]
    ) -> None:
        for input_ids in model_inputs["input_ids"]:
            masked_index = self._get_mask_token_index(input_ids)
            numel = np.prod(masked_index.shape)
            assert numel == 1

    @torch.no_grad()
    def _get_logits(self, texts: List[str]) -> torch.Tensor:
        # for MLM, add mask token
        batch_size = len(texts)
        encoded_inputs = self._tokenizer(
            texts,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        if self.is_mask_lm:
            # self.ensure_exactly_one_mask_token(encoded_inputs) TODO
            token_logits = self._generator(**encoded_inputs.to(self.device)).logits
            mask_token_indices = self._get_mask_token_index(encoded_inputs["input_ids"])
            out_logits = token_logits[range(batch_size), mask_token_indices, :]
        else:
            token_logits = self._generator(**encoded_inputs.to(self.device)).logits
            input_lengths = encoded_inputs["attention_mask"].sum(dim=1)
            out_logits = token_logits[range(batch_size), input_lengths - 1, :]

        return out_logits

    def _convert_tokens_to_string(self, tokens: List[List[str]]) -> List[str]:
        return [self._tokenizer.convert_tokens_to_string(s) for s in tokens]

    def _format_prompts(
        self,
        source_strs: List[str],
        prompt_strs: List[str],
    ) -> List[str]:
        return [
            self.template.format(sentence_1=s_1, prompt=p)
            for s_1, p in zip(source_strs, prompt_strs)
        ]
