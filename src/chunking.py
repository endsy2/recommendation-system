"""Token-ID lyric chunking without decode/re-tokenize truncation hazards."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 200
    overlap: int = 32

    @property
    def stride(self) -> int:
        return self.chunk_size - self.overlap

    def validate(self, model_input_limit: int, special_token_count: int) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be nonnegative and smaller than chunk_size")
        if special_token_count < 0:
            raise ValueError("special_token_count cannot be negative")
        if self.chunk_size + special_token_count > model_input_limit:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) plus {special_token_count} special tokens exceeds "
                f"the model input limit ({model_input_limit})"
            )


def content_token_ids(tokenizer: object, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False, return_attention_mask=False)
    return list(encoded["input_ids"])


def token_slices(token_count: int, config: ChunkConfig) -> list[tuple[int, int]]:
    """Cover content exactly, including the final partial chunk, with no overlap-only tail."""
    if token_count < 0:
        raise ValueError("token_count cannot be negative")
    if config.chunk_size <= 0 or config.overlap < 0 or config.overlap >= config.chunk_size:
        raise ValueError("overlap must be nonnegative and smaller than chunk_size")
    if token_count == 0:
        return [(0, 0)]
    slices: list[tuple[int, int]] = []
    start = 0
    while start < token_count:
        end = min(start + config.chunk_size, token_count)
        slices.append((start, end))
        if end == token_count:
            break
        start += config.stride
    return slices


def chunk_token_ids(ids: Sequence[int], config: ChunkConfig) -> Iterable[list[int]]:
    for start, end in token_slices(len(ids), config):
        yield list(ids[start:end])


def special_token_count(tokenizer: object) -> int:
    return int(tokenizer.num_special_tokens_to_add(pair=False))


def model_inputs_for_content(tokenizer: object, content_ids: Sequence[int], input_limit: int) -> dict[str, list[int]]:
    """Add specials to already-tokenized IDs; never decode and tokenize them again."""
    features = tokenizer.prepare_for_model(
        list(content_ids), add_special_tokens=True, truncation=False, return_attention_mask=True
    )
    if len(features["input_ids"]) > input_limit:
        raise ValueError(
            f"Prepared input has {len(features['input_ids'])} tokens, exceeding model input limit {input_limit}"
        )
    return {"input_ids": list(features["input_ids"]), "attention_mask": list(features["attention_mask"])}
