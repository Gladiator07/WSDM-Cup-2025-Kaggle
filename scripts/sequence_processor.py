from transformers import AutoTokenizer


class SequenceProcessor:
    def __init__(
        self,
        model_name: str,
        max_length: int = 1600,
        truncation_side: str = "left",
        padding_side: str = "left",
        add_eos_token: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            # use_fast=True,
            # trust_remote_code=True,
            # from_slow=True,
            # add_prefix_space=False,
            padding_side=padding_side,
            truncation_side=truncation_side,
        )
        self.tokenizer.add_eos_token = add_eos_token
        self.model_name = model_name

        self.max_length = max_length

        # Template parts (only tokenize once during initialization)
        self.templates = {
            "start": self.tokenizer.encode("# Prompt\n", add_special_tokens=False),
            "response_a": self.tokenizer.encode("\n\n# Response A\n", add_special_tokens=False),
            "response_b": self.tokenizer.encode("\n\n# Response B\n", add_special_tokens=False),
            "question": self.tokenizer.encode("\n\n# Which response is better?", add_special_tokens=False),
            "ellipsis": self.tokenizer.encode(" [...] ", add_special_tokens=False),
        }

        # Calculate fixed template length
        self.template_length = sum(len(tokens) for tokens in self.templates.values()) - len(self.templates["ellipsis"])

    def truncate_if_needed(self, tokens, max_tokens):
        """Truncate tokens if they exceed max_tokens by keeping start and end portions."""
        if len(tokens) <= max_tokens:
            return tokens

        keep_tokens = (max_tokens - len(self.templates["ellipsis"])) // 2
        return tokens[:keep_tokens] + self.templates["ellipsis"] + tokens[-keep_tokens:]

    def tokenize(self, row, tta=False):
        if tta:
            prompt, response_a, response_b = row["prompt"], row["response_b"], row["response_a"]
        else:
            prompt, response_a, response_b = row["prompt"], row["response_a"], row["response_b"]

        # Available tokens after accounting for template and special tokens
        buffer_tokens = 3
        available_tokens = self.max_length - self.template_length - buffer_tokens

        # Tokenize all inputs at once
        enc = self.tokenizer([prompt, response_a, response_b], add_special_tokens=False)["input_ids"]
        prompt_tokens, response_a_tokens, response_b_tokens = enc[0], enc[1], enc[2]

        total_length = len(prompt_tokens) + len(response_a_tokens) + len(response_b_tokens)

        # If total length is within limit, return without truncation
        if total_length <= available_tokens:
            final_sequence = (
                [self.tokenizer.bos_token_id]
                + self.templates["start"]
                + prompt_tokens
                + self.templates["response_a"]
                + response_a_tokens
                + self.templates["response_b"]
                + response_b_tokens
                + self.templates["question"]
            )
            if "qwen" in self.model_name.lower():
                final_sequence.pop(0)
            if self.tokenizer.add_eos_token:
                final_sequence.append(self.tokenizer.eos_token_id)

            return {"input_ids": final_sequence, "attention_mask": [1] * len(final_sequence), "length": len(final_sequence)}

        # Allocate tokens based on 20-40-40 split with dynamic adjustment
        prompt_max = int(available_tokens * 0.2)  # Reserve 20% for prompt
        response_max = int(available_tokens * 0.4)  # 40% each for responses

        # If prompt needs less than its allocation, distribute the excess
        prompt_needed = min(len(prompt_tokens), prompt_max)
        excess_tokens = prompt_max - prompt_needed

        # Add half of excess to each response's budget
        response_a_max = response_max + excess_tokens // 2
        response_b_max = response_max + excess_tokens - (excess_tokens // 2)  # Account for odd number

        # Calculate actual token allocations
        prompt_max_tokens = prompt_needed
        response_a_max_tokens = min(len(response_a_tokens), response_a_max)
        response_b_max_tokens = min(len(response_b_tokens), response_b_max)

        # Truncate each section if needed
        prompt_tokens = self.truncate_if_needed(prompt_tokens, prompt_max_tokens)
        response_a_tokens = self.truncate_if_needed(response_a_tokens, response_a_max_tokens)
        response_b_tokens = self.truncate_if_needed(response_b_tokens, response_b_max_tokens)

        # Assemble final input
        final_sequence = (
            [self.tokenizer.bos_token_id]
            + self.templates["start"]
            + prompt_tokens
            + self.templates["response_a"]
            + response_a_tokens
            + self.templates["response_b"]
            + response_b_tokens
            + self.templates["question"]
        )
        if "qwen" in self.model_name.lower():
            final_sequence.pop(0)
        if self.tokenizer.add_eos_token:
            final_sequence.append(self.tokenizer.eos_token_id)
        return {"input_ids": final_sequence, "attention_mask": [1] * len(final_sequence), "length": len(final_sequence)}
