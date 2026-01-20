import pytest
import torch

from cka.hsic import hsic, hsic_outer


class TestHsic:
    def test_valid_3d_inputs(self):
        grams_x = torch.randn(5, 8, 8)
        grams_y = torch.randn(3, 8, 8)

        hsic_xy, hsic_xx, hsic_yy = hsic(grams_x, grams_y)

        assert hsic_xy.shape == (5, 3)
        assert hsic_xx.shape == (5,)
        assert hsic_yy.shape == (3,)

    def test_non_3d_input_x_raises_error(self):
        grams_x = torch.randn(5, 8)
        grams_y = torch.randn(3, 8, 8)

        with pytest.raises(ValueError, match="hsic_all requires 3D tensors"):
            hsic(grams_x, grams_y)

    def test_non_3d_input_y_raises_error(self):
        grams_x = torch.randn(5, 8, 8)
        grams_y = torch.randn(3, 8, 8, 2)

        with pytest.raises(ValueError, match="hsic_all requires 3D tensors"):
            hsic(grams_x, grams_y)

    def test_non_square_matrix_x_raises_error(self):
        grams_x = torch.randn(5, 8, 6)
        grams_y = torch.randn(3, 8, 8)

        with pytest.raises(ValueError, match="hsic_all requires square matrices"):
            hsic(grams_x, grams_y)

    def test_non_square_matrix_y_raises_error(self):
        grams_x = torch.randn(5, 8, 8)
        grams_y = torch.randn(3, 8, 6)

        with pytest.raises(ValueError, match="hsic_all requires square matrices"):
            hsic(grams_x, grams_y)

    def test_mismatched_sample_dimension_raises_error(self):
        grams_x = torch.randn(5, 8, 8)
        grams_y = torch.randn(3, 10, 10)

        with pytest.raises(ValueError, match="must have same sample dimension"):
            hsic(grams_x, grams_y)

    def test_n_equals_3_raises_error(self):
        grams_x = torch.randn(5, 3, 3)
        grams_y = torch.randn(3, 3, 3)

        with pytest.raises(ValueError, match="hsic_all requires n > 3"):
            hsic(grams_x, grams_y)

    def test_n_equals_2_raises_error(self):
        grams_x = torch.randn(5, 2, 2)
        grams_y = torch.randn(3, 2, 2)

        with pytest.raises(ValueError, match="hsic_all requires n > 3"):
            hsic(grams_x, grams_y)

    def test_diagonal_is_one_for_normalized_gram(self):
        n_samples = 8
        n_layers = 4

        features = torch.randn(n_layers, n_samples, 16)
        features = features / features.norm(dim=-1, keepdim=True)
        grams = torch.bmm(features, features.transpose(1, 2))

        hsic_xy, hsic_xx, hsic_yy = hsic(grams, grams)

        diagonal = torch.diagonal(hsic_xy)
        assert torch.allclose(diagonal, hsic_xx, atol=1e-5)
        assert torch.allclose(diagonal, hsic_yy, atol=1e-5)


class TestHsicOuter:
    def test_valid_inputs_shape(self):
        grams_x = torch.randn(5, 8, 8)
        grams_y = torch.randn(3, 8, 8)

        result = hsic_outer(grams_x, grams_y)

        assert result.shape == (5, 3)

    def test_non_3d_input_x_raises_error(self):
        grams_x = torch.randn(5, 8)
        grams_y = torch.randn(3, 8, 8)

        with pytest.raises(ValueError, match="hsic_outer requires 3D tensors"):
            hsic_outer(grams_x, grams_y)

    def test_non_3d_input_y_raises_error(self):
        grams_x = torch.randn(5, 8, 8)
        grams_y = torch.randn(3, 8, 8, 2)

        with pytest.raises(ValueError, match="hsic_outer requires 3D tensors"):
            hsic_outer(grams_x, grams_y)

    def test_non_square_matrix_x_raises_error(self):
        grams_x = torch.randn(5, 8, 6)
        grams_y = torch.randn(3, 8, 8)

        with pytest.raises(ValueError, match="hsic_outer requires square matrices"):
            hsic_outer(grams_x, grams_y)

    def test_non_square_matrix_y_raises_error(self):
        grams_x = torch.randn(5, 8, 8)
        grams_y = torch.randn(3, 8, 6)

        with pytest.raises(ValueError, match="hsic_outer requires square matrices"):
            hsic_outer(grams_x, grams_y)

    def test_mismatched_sample_dimension_raises_error(self):
        grams_x = torch.randn(5, 8, 8)
        grams_y = torch.randn(3, 10, 10)

        with pytest.raises(ValueError, match="must have same sample dimension"):
            hsic_outer(grams_x, grams_y)

    def test_n_equals_3_raises_error(self):
        grams_x = torch.randn(5, 3, 3)
        grams_y = torch.randn(3, 3, 3)

        with pytest.raises(ValueError, match="hsic_outer requires n > 3"):
            hsic_outer(grams_x, grams_y)

    def test_n_equals_1_raises_error(self):
        grams_x = torch.randn(5, 1, 1)
        grams_y = torch.randn(3, 1, 1)

        with pytest.raises(ValueError, match="hsic_outer requires n > 3"):
            hsic_outer(grams_x, grams_y)

    def test_symmetry_same_inputs(self):
        grams = torch.randn(4, 8, 8)
        grams = grams + grams.transpose(-1, -2)

        result = hsic_outer(grams, grams)

        assert torch.allclose(result, result.T, atol=1e-5)

    def test_consistency_with_hsic(self):
        grams_x = torch.randn(3, 8, 8)
        grams_y = torch.randn(4, 8, 8)

        grams_x = grams_x + grams_x.transpose(-1, -2)
        grams_y = grams_y + grams_y.transpose(-1, -2)

        outer_result = hsic_outer(grams_x, grams_y)
        hsic_xy, _, _ = hsic(grams_x, grams_y)

        assert torch.allclose(outer_result, hsic_xy, atol=1e-5)
