import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cka import compute_cka


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


class AnotherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 15)
        self.fc2 = nn.Linear(15, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def dataloader():
    x = torch.randn(32, 10)
    dataset = TensorDataset(x)
    return DataLoader(dataset, batch_size=8)


@pytest.fixture
def model1():
    return SimpleModel()


@pytest.fixture
def model2():
    return AnotherModel()


class TestCKABasic:
    def test_different_models(self, model1, model2, dataloader):
        result = compute_cka(model1, model2, dataloader)[0]

        assert result.shape == (3, 2)
        assert torch.all((result >= 0) & (result <= 1))

    def test_same_model(self, model1, dataloader):
        result = compute_cka(model1, model1, dataloader)[0]

        assert result.shape == (3, 3)
        diagonal = torch.diagonal(result)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-5)


class TestComputeCkaFunction:
    def test_returns_list_for_multiple_dataloaders(self, model1, model2, dataloader):
        x2 = torch.randn(32, 10)
        dataset2 = TensorDataset(x2)
        dataloader2 = DataLoader(dataset2, batch_size=8)

        results = compute_cka(model1, model2, dataloader, dataloader2, progress=False)

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].shape == (3, 2)
        assert results[1].shape == (3, 2)

    def test_requires_dataloader(self, model1, model2):
        with pytest.raises(ValueError, match="at least one dataloader"):
            compute_cka(model1, model2)


class TestLayerSpecification:
    def test_string_layer_names(self, model1, model2, dataloader):
        result = compute_cka(
            model1,
            model2,
            dataloader,
            layers=["layer1", "layer3"],
            model2_layers=["fc1"],
        )[0]

        assert result.shape == (2, 1)

    def test_integer_indices(self, model1, model2, dataloader):
        result = compute_cka(
            model1,
            model2,
            dataloader,
            layers=[0, 2],
            model2_layers=[0],
        )[0]

        assert result.shape == (2, 1)

    def test_negative_indices(self, model1, model2, dataloader):
        result = compute_cka(
            model1,
            model2,
            dataloader,
            layers=[-1],
            model2_layers=[-1],
        )[0]

        assert result.shape == (1, 1)

    def test_mixed_indices(self, model1, model2, dataloader):
        result = compute_cka(
            model1,
            model2,
            dataloader,
            layers=["layer1", -1],
            model2_layers=[0, "fc2"],
        )[0]

        assert result.shape == (2, 2)


class TestLayerValidation:
    def test_invalid_layer_name(self, model1, model2, dataloader):
        with pytest.raises(ValueError, match="Layer 'nonexistent' not found"):
            compute_cka(model1, model2, dataloader, layers=["nonexistent"])

    def test_invalid_layer_index_positive(self, model1, model2, dataloader):
        with pytest.raises(IndexError, match="out of range"):
            compute_cka(model1, model2, dataloader, layers=[100])

    def test_invalid_layer_index_negative(self, model1, model2, dataloader):
        with pytest.raises(IndexError, match="out of range"):
            compute_cka(model1, model2, dataloader, layers=[-100])

    def test_invalid_layer_type(self, model1, model2, dataloader):
        with pytest.raises(TypeError, match="must be str or int"):
            compute_cka(model1, model2, dataloader, layers=[1.5])


class TestSameModelDifferentLayers:
    def test_same_model_different_layers(self, model1, dataloader):
        result = compute_cka(
            model1,
            model1,
            dataloader,
            layers=["layer1", "layer2"],
            model2_layers=["layer2", "layer3"],
        )[0]

        assert result.shape == (2, 2)


class TestLayerInheritance:
    def test_layers_applies_to_both_models(self, model1, dataloader):
        """When layers is specified without model2_layers, both models use same layers."""
        result = compute_cka(
            model1,
            model1,
            dataloader,
            layers=["layer1", "layer2"],
        )[0]

        assert result.shape == (2, 2)
        diagonal = torch.diagonal(result)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-5)

    def test_layers_none_uses_all_layers(self, model1, dataloader):
        """When layers is None, all layers are used for both models."""
        result = compute_cka(model1, model1, dataloader)[0]

        assert result.shape == (3, 3)

    def test_model2_layers_overrides_inheritance(self, model1, dataloader):
        """When model2_layers is explicitly specified, it overrides inheritance."""
        result = compute_cka(
            model1,
            model1,
            dataloader,
            layers=["layer1"],
            model2_layers=["layer2", "layer3"],
        )[0]

        assert result.shape == (1, 2)
