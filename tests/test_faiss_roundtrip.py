import json
import numpy as np
import pytest


faiss = pytest.importorskip("faiss")


def test_flat_ip_matches_normalized_dot_products_and_round_trips(tmp_path):
    vectors = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(2); index.add(vectors)
    query = np.array([[1, .2]], dtype=np.float32); query /= np.linalg.norm(query)
    scores, positions = index.search(query, 3)
    assert positions[0].tolist() == np.argsort(-(vectors @ query[0])).tolist()
    path = tmp_path / "songs.faiss"; faiss.write_index(index, str(path))
    assert faiss.read_index(str(path)).ntotal == 3
