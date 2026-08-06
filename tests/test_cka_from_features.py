import pytest
import torch

from cka import cka_from_features


def _random_features(n_layers=3, n_samples=16, feature_dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_layers, n_samples, feature_dim, generator=g)


class TestCkaFromFeatures:
    def test_single_pair_returns_1x1(self):
        x = torch.randn(16, 8)
        y = torch.randn(16, 4)
        result = cka_from_features(x, y)
        assert result.shape == (1, 1)

    def test_multi_layer_shape(self):
        x = _random_features(n_layers=3, seed=0)
        y = _random_features(n_layers=2, seed=1)
        result = cka_from_features(x, y)
        assert result.shape == (3, 2)

    def test_list_of_varying_dims(self):
        x = [torch.randn(16, d) for d in (8, 4, 12)]
        y = [torch.randn(16, d) for d in (6, 10)]
        result = cka_from_features(x, y)
        assert result.shape == (3, 2)

    def test_self_similarity_near_one(self):
        x = _random_features(n_layers=4, seed=42)
        result = cka_from_features(x, x)
        diag = torch.diagonal(result)
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)

    def test_values_in_range(self):
        x = _random_features(n_layers=3, seed=0)
        y = _random_features(n_layers=3, seed=1)
        result = cka_from_features(x, y)
        assert torch.all(result >= 0)
        assert torch.all(result <= 1)

    def test_n_samples_too_small_raises(self):
        x = torch.randn(3, 8)
        y = torch.randn(3, 8)
        with pytest.raises(ValueError, match=r"n > 3|n_samples > 3"):
            cka_from_features(x, y)
