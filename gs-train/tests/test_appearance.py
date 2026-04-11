import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vroom_core.models.decoder import AppearanceTable as Embedding


def test_embedding_returns_requested_rows():
    torch.manual_seed(7)
    embedding = Embedding(6, 4, init_std=0.01).cuda()

    indices = torch.tensor([0, 3, 5], dtype=torch.long, device="cuda")
    output = embedding(indices)

    assert output.shape == (3, 4)
    assert torch.allclose(output[1], embedding._table.weight[3])


def test_embedding_mean_matches_weight_mean():
    torch.manual_seed(11)
    embedding = Embedding(5, 3).cuda()

    assert torch.allclose(embedding.mean(dim=0), embedding._table.weight.mean(dim=0))
