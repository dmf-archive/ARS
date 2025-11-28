import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM


class Qwen3RoPEWrapper(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        rope_theta: float,
        intermediate_size: int,
        tie_word_embeddings: bool,
    ):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.intermediate_size = intermediate_size
        self.tie_word_embeddings = tie_word_embeddings

        config = Qwen3Config(
            vocab_size=vocabulary_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            intermediate_size=intermediate_size,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=False,
            use_sliding_window=False,
        )

        self.model = Qwen3ForCausalLM(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=inputs)
        return outputs.logits

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = nn.CrossEntropyLoss(reduction='none')(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        masked_loss = loss * mask.view(-1)
        mean_loss = masked_loss.sum() / mask.sum()
        return mean_loss

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
