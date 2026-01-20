import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cka import CKA


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
        cka = CKA(model1, model2)
        result = cka(dataloader)

        assert result.shape == (3, 2)
        assert torch.all((result >= 0) & (result <= 1))

    def test_same_model(self, model1, dataloader):
        cka = CKA(model1, model1)
        result = cka(dataloader)

        assert result.shape == (3, 3)
        diagonal = torch.diagonal(result)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-5)

    def test_context_manager(self, model1, model2, dataloader):
        cka = CKA(model1, model2)
        with cka:
            result = cka.compare(dataloader)

        assert result.shape == (3, 2)

    def test_export(self, model1, model2, dataloader):
        cka = CKA(model1, model2, model1_name="Model1", model2_name="Model2")
        result = cka(dataloader)
        exported = cka.export(result)

        assert exported["model1_name"] == "Model1"
        assert exported["model2_name"] == "Model2"
        assert exported["model1_layers"] == ["layer1", "layer2", "layer3"]
        assert exported["model2_layers"] == ["fc1", "fc2"]
        assert torch.equal(exported["cka_matrix"], result)


class TestLayerSpecification:
    def test_string_layer_names(self, model1, model2, dataloader):
        cka = CKA(
            model1,
            model2,
            model1_layers=["layer1", "layer3"],
            model2_layers=["fc1"],
        )
        result = cka(dataloader)

        assert result.shape == (2, 1)

    def test_integer_indices(self, model1, model2, dataloader):
        cka = CKA(
            model1,
            model2,
            model1_layers=[0, 2],
            model2_layers=[0],
        )
        result = cka(dataloader)

        assert result.shape == (2, 1)

    def test_negative_indices(self, model1, model2, dataloader):
        cka = CKA(
            model1,
            model2,
            model1_layers=[-1],
            model2_layers=[-1],
        )
        result = cka(dataloader)

        assert result.shape == (1, 1)

    def test_mixed_indices(self, model1, model2, dataloader):
        cka = CKA(
            model1,
            model2,
            model1_layers=["layer1", -1],
            model2_layers=[0, "fc2"],
        )
        result = cka(dataloader)

        assert result.shape == (2, 2)


class TestLayerValidation:
    def test_invalid_layer_name(self, model1, model2):
        with pytest.raises(ValueError, match="Layer 'nonexistent' not found"):
            CKA(model1, model2, model1_layers=["nonexistent"])

    def test_invalid_layer_index_positive(self, model1, model2):
        with pytest.raises(IndexError, match="out of range"):
            CKA(model1, model2, model1_layers=[100])

    def test_invalid_layer_index_negative(self, model1, model2):
        with pytest.raises(IndexError, match="out of range"):
            CKA(model1, model2, model1_layers=[-100])

    def test_invalid_layer_type(self, model1, model2):
        with pytest.raises(TypeError, match="must be str or int"):
            CKA(model1, model2, model1_layers=[1.5])


class TestSameModelDifferentLayers:
    def test_same_model_different_layers(self, model1, dataloader):
        cka = CKA(
            model1,
            model1,
            model1_layers=["layer1", "layer2"],
            model2_layers=["layer2", "layer3"],
        )
        result = cka(dataloader)

        assert result.shape == (2, 2)
