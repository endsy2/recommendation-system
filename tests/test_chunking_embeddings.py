import numpy as np
import pytest

from src.chunking import ChunkConfig, model_inputs_for_content, token_slices
from src.embeddings import EmbeddingError, pool_normalized_chunks


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair=False): return 2
    def prepare_for_model(self, ids, **kwargs): return {"input_ids": [101, *ids, 102], "attention_mask": [1] * (len(ids) + 2)}


def test_boundaries_coverage_and_final_partial_chunk():
    config = ChunkConfig(200, 32)
    assert config.stride == 168
    assert token_slices(450, config) == [(0, 200), (168, 368), (336, 450)]
    assert token_slices(200, config) == [(0, 200)]
    slices = token_slices(450, config)
    assert slices[0][0] == 0 and slices[-1][1] == 450
    assert all(next_start <= end for (_, end), (next_start, _) in zip(slices, slices[1:]))


def test_invalid_overlap_and_special_token_limit():
    with pytest.raises(ValueError): ChunkConfig(10, 10).validate(20, 2)
    with pytest.raises(ValueError): ChunkConfig(200, 32).validate(201, 2)
    with pytest.raises(ValueError, match="exceeding"): model_inputs_for_content(FakeTokenizer(), list(range(200)), 201)


def test_pooling_normalizes_and_never_crosses_song_boundaries():
    left = pool_normalized_chunks([np.array([3, 0], np.float32), np.array([0, 4], np.float32)], expected_dimension=2)
    right = pool_normalized_chunks([np.array([0, 5], np.float32)], expected_dimension=2)
    assert np.allclose(left, [2 ** -0.5, 2 ** -0.5])
    assert np.allclose(right, [0, 1])
    with pytest.raises(EmbeddingError): pool_normalized_chunks([np.array([0, 0], np.float32)], expected_dimension=2)
