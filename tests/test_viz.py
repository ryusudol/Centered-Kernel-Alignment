import os
import tempfile
import unittest.mock
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from cka.viz import plot_cka_comparison, plot_cka_heatmap, plot_cka_trend, save_figure


class TestPlotCkaHeatmap:
    def test_basic_with_torch_tensor(self):
        matrix = torch.rand(5, 4)
        fig, ax = plot_cka_heatmap(matrix)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_basic_with_numpy_array(self):
        matrix = np.random.rand(5, 4)
        fig, ax = plot_cka_heatmap(matrix)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_custom_axis(self):
        matrix = torch.rand(5, 4)
        external_fig, external_ax = plt.subplots()

        fig, ax = plot_cka_heatmap(matrix, ax=external_ax)

        assert ax is external_ax
        assert fig is external_fig
        plt.close(fig)

    def test_mask_upper_symmetric(self):
        matrix = torch.rand(5, 5)
        matrix = (matrix + matrix.T) / 2

        fig, ax = plot_cka_heatmap(matrix, mask_upper=True)

        assert fig is not None
        plt.close(fig)

    def test_mask_upper_non_square_ignored(self):
        matrix = torch.rand(5, 4)

        fig, ax = plot_cka_heatmap(matrix, mask_upper=True)

        assert fig is not None
        plt.close(fig)

    def test_annot_true(self):
        matrix = torch.rand(3, 3)

        fig, ax = plot_cka_heatmap(matrix, annot=True)

        texts = [child for child in ax.get_children() if hasattr(child, "get_text")]
        assert len(texts) > 0
        plt.close(fig)

    def test_annot_with_mask_upper(self):
        matrix = torch.rand(4, 4)
        matrix = (matrix + matrix.T) / 2

        fig, ax = plot_cka_heatmap(matrix, annot=True, mask_upper=True)

        assert fig is not None
        plt.close(fig)

    def test_custom_layer_names(self):
        matrix = torch.rand(3, 2)
        layers1 = ["layer1", "layer2", "layer3"]
        layers2 = ["fc1", "fc2"]

        fig, ax = plot_cka_heatmap(matrix, layers1=layers1, layers2=layers2)

        yticklabels = [t.get_text() for t in ax.get_yticklabels()]
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        assert yticklabels == layers1
        assert xticklabels == layers2
        plt.close(fig)

    def test_layer_name_depth(self):
        matrix = torch.rand(2, 2)
        layers1 = ["encoder.block.layer1", "encoder.block.layer2"]
        layers2 = ["decoder.fc1", "decoder.fc2"]

        fig, ax = plot_cka_heatmap(
            matrix, layers1=layers1, layers2=layers2, layer_name_depth=1
        )

        yticklabels = [t.get_text() for t in ax.get_yticklabels()]
        assert yticklabels == ["layer1", "layer2"]
        plt.close(fig)

    def test_layer_name_depth_2(self):
        matrix = torch.rand(2, 2)
        layers1 = ["encoder.block.layer1", "encoder.block.layer2"]
        layers2 = ["decoder.fc1", "decoder.fc2"]

        fig, ax = plot_cka_heatmap(
            matrix, layers1=layers1, layers2=layers2, layer_name_depth=2
        )

        yticklabels = [t.get_text() for t in ax.get_yticklabels()]
        assert yticklabels == ["block.layer1", "block.layer2"]
        plt.close(fig)

    def test_custom_vmin_vmax(self):
        matrix = torch.rand(4, 4)

        fig, ax = plot_cka_heatmap(matrix, vmin=0.2, vmax=0.8)

        images = ax.get_images()
        assert len(images) == 1
        assert images[0].get_clim() == (0.2, 0.8)
        plt.close(fig)

    def test_colorbar_false(self):
        matrix = torch.rand(4, 4)

        fig, ax = plot_cka_heatmap(matrix, colorbar=False)

        assert fig is not None
        plt.close(fig)

    def test_show_false_default(self):
        matrix = torch.rand(4, 4)

        fig, ax = plot_cka_heatmap(matrix, show=False)

        assert fig is not None
        plt.close(fig)

    def test_show_true(self):
        matrix = torch.rand(4, 4)

        with unittest.mock.patch.object(plt, "show"):
            fig, ax = plot_cka_heatmap(matrix, show=True)

        assert fig is not None
        plt.close(fig)

    def test_custom_model_names(self):
        matrix = torch.rand(4, 4)

        fig, ax = plot_cka_heatmap(
            matrix, model1_name="ResNet", model2_name="VGG", title=None
        )

        assert "ResNet vs VGG" in ax.get_title()
        plt.close(fig)

    def test_custom_title(self):
        matrix = torch.rand(4, 4)

        fig, ax = plot_cka_heatmap(matrix, title="My Custom Title")

        assert ax.get_title() == "My Custom Title"
        plt.close(fig)

    def test_custom_figsize(self):
        matrix = torch.rand(4, 4)

        fig, ax = plot_cka_heatmap(matrix, figsize=(12, 10))

        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 10
        plt.close(fig)

    def test_custom_cmap(self):
        matrix = torch.rand(4, 4)

        fig, ax = plot_cka_heatmap(matrix, cmap="viridis")

        images = ax.get_images()
        assert images[0].get_cmap().name == "viridis"
        plt.close(fig)

    def test_nan_values_in_annot(self):
        matrix = torch.rand(3, 3)
        matrix[0, 0] = float("nan")

        fig, ax = plot_cka_heatmap(matrix, annot=True)

        assert fig is not None
        plt.close(fig)


class TestPlotCkaTrend:
    def test_single_1d_tensor(self):
        values = torch.rand(10)

        fig, ax = plot_cka_trend(values)

        assert fig is not None
        assert ax is not None
        lines = ax.get_lines()
        assert len(lines) == 1
        plt.close(fig)

    def test_2d_tensor_multiple_lines(self):
        values = torch.rand(3, 10)

        fig, ax = plot_cka_trend(values)

        lines = ax.get_lines()
        assert len(lines) == 3
        plt.close(fig)

    def test_list_of_tensors(self):
        values = [torch.rand(10), torch.rand(10)]

        fig, ax = plot_cka_trend(values)

        lines = ax.get_lines()
        assert len(lines) == 2
        plt.close(fig)

    def test_list_of_numpy_arrays(self):
        values = [np.random.rand(10), np.random.rand(10)]

        fig, ax = plot_cka_trend(values)

        lines = ax.get_lines()
        assert len(lines) == 2
        plt.close(fig)

    def test_custom_x_values(self):
        values = torch.rand(5)
        x_values = [0, 2, 4, 6, 8]

        fig, ax = plot_cka_trend(values, x_values=x_values)

        lines = ax.get_lines()
        xdata = lines[0].get_xdata()
        assert list(xdata) == x_values
        plt.close(fig)

    def test_custom_colors(self):
        values = [torch.rand(10), torch.rand(10)]
        colors = ["red", "blue"]

        fig, ax = plot_cka_trend(values, colors=colors)

        assert fig is not None
        plt.close(fig)

    def test_custom_linestyles(self):
        values = [torch.rand(10), torch.rand(10)]
        linestyles = ["--", "-."]

        fig, ax = plot_cka_trend(values, linestyles=linestyles)

        assert fig is not None
        plt.close(fig)

    def test_custom_markers(self):
        values = [torch.rand(10), torch.rand(10)]
        markers = ["s", "^"]

        fig, ax = plot_cka_trend(values, markers=markers)

        assert fig is not None
        plt.close(fig)

    def test_labels(self):
        values = [torch.rand(10), torch.rand(10)]
        labels = ["Model A", "Model B"]

        fig, ax = plot_cka_trend(values, labels=labels, legend=True)

        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_legend_true_multiple_lines(self):
        values = [torch.rand(10), torch.rand(10)]

        fig, ax = plot_cka_trend(values, legend=True)

        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)

    def test_legend_false(self):
        values = [torch.rand(10), torch.rand(10)]

        fig, ax = plot_cka_trend(values, legend=False)

        legend = ax.get_legend()
        assert legend is None
        plt.close(fig)

    def test_legend_single_line(self):
        values = torch.rand(10)

        fig, ax = plot_cka_trend(values, legend=True)

        legend = ax.get_legend()
        assert legend is None
        plt.close(fig)

    def test_grid_false(self):
        values = torch.rand(10)

        fig, ax = plot_cka_trend(values, grid=False)

        assert fig is not None
        plt.close(fig)

    def test_custom_ax(self):
        values = torch.rand(10)
        external_fig, external_ax = plt.subplots()

        fig, ax = plot_cka_trend(values, ax=external_ax)

        assert ax is external_ax
        assert fig is external_fig
        plt.close(fig)

    def test_custom_xlabel_ylabel(self):
        values = torch.rand(10)

        fig, ax = plot_cka_trend(values, xlabel="Epoch", ylabel="Similarity")

        assert ax.get_xlabel() == "Epoch"
        assert ax.get_ylabel() == "Similarity"
        plt.close(fig)

    def test_custom_title(self):
        values = torch.rand(10)

        fig, ax = plot_cka_trend(values, title="Trend Plot")

        assert ax.get_title() == "Trend Plot"
        plt.close(fig)

    def test_ylim_set(self):
        values = torch.rand(10)

        fig, ax = plot_cka_trend(values)

        ylim = ax.get_ylim()
        assert ylim[0] == 0
        assert ylim[1] == 1.05
        plt.close(fig)

    def test_1d_numpy_array(self):
        values = np.random.rand(10)

        fig, ax = plot_cka_trend(values)

        lines = ax.get_lines()
        assert len(lines) == 1
        plt.close(fig)

    def test_show_true(self):
        values = torch.rand(10)

        with unittest.mock.patch.object(plt, "show"):
            fig, ax = plot_cka_trend(values, show=True)

        assert fig is not None
        plt.close(fig)


class TestPlotCkaComparison:
    def test_basic_comparison(self):
        matrices = [torch.rand(4, 4), torch.rand(4, 4)]
        titles = ["Matrix 1", "Matrix 2"]

        fig, axes = plot_cka_comparison(matrices, titles)

        assert fig is not None
        assert axes.shape == (1, 2)
        plt.close(fig)

    def test_share_colorbar_true(self):
        matrices = [torch.rand(4, 4), torch.rand(4, 4)]
        titles = ["Matrix 1", "Matrix 2"]

        fig, axes = plot_cka_comparison(matrices, titles, share_colorbar=True)

        assert fig is not None
        plt.close(fig)

    def test_share_colorbar_false(self):
        matrices = [torch.rand(4, 4), torch.rand(4, 4)]
        titles = ["Matrix 1", "Matrix 2"]

        fig, axes = plot_cka_comparison(matrices, titles, share_colorbar=False)

        assert fig is not None
        plt.close(fig)

    def test_custom_ncols(self):
        matrices = [torch.rand(4, 4) for _ in range(4)]
        titles = [f"Matrix {i}" for i in range(4)]

        fig, axes = plot_cka_comparison(matrices, titles, ncols=4)

        assert axes.shape == (1, 4)
        plt.close(fig)

    def test_auto_figsize(self):
        matrices = [torch.rand(4, 4), torch.rand(4, 4)]
        titles = ["Matrix 1", "Matrix 2"]

        fig, axes = plot_cka_comparison(matrices, titles, figsize=None)

        assert fig is not None
        plt.close(fig)

    def test_custom_figsize(self):
        matrices = [torch.rand(4, 4), torch.rand(4, 4)]
        titles = ["Matrix 1", "Matrix 2"]

        fig, axes = plot_cka_comparison(matrices, titles, figsize=(15, 8))

        size = fig.get_size_inches()
        assert size[0] == 15
        assert size[1] == 8
        plt.close(fig)

    def test_hide_empty_subplots(self):
        matrices = [torch.rand(4, 4) for _ in range(3)]
        titles = [f"Matrix {i}" for i in range(3)]

        fig, axes = plot_cka_comparison(matrices, titles, ncols=2)

        assert axes.shape == (2, 2)
        assert not axes[1, 1].get_visible()
        plt.close(fig)

    def test_heatmap_kwargs_passthrough(self):
        matrices = [torch.rand(4, 4), torch.rand(4, 4)]
        titles = ["Matrix 1", "Matrix 2"]

        fig, axes = plot_cka_comparison(matrices, titles, annot=True, tick_fontsize=10)

        assert fig is not None
        plt.close(fig)

    def test_custom_layers(self):
        matrices = [torch.rand(3, 3), torch.rand(3, 3)]
        titles = ["Matrix 1", "Matrix 2"]
        layers = ["layer1", "layer2", "layer3"]

        fig, axes = plot_cka_comparison(matrices, titles, layers=layers)

        assert fig is not None
        plt.close(fig)

    def test_numpy_arrays(self):
        matrices = [np.random.rand(4, 4), np.random.rand(4, 4)]
        titles = ["Matrix 1", "Matrix 2"]

        fig, axes = plot_cka_comparison(matrices, titles)

        assert fig is not None
        plt.close(fig)

    def test_mixed_torch_numpy(self):
        matrices = [torch.rand(4, 4), np.random.rand(4, 4)]
        titles = ["Matrix 1", "Matrix 2"]

        fig, axes = plot_cka_comparison(matrices, titles)

        assert fig is not None
        plt.close(fig)

    def test_single_matrix(self):
        matrices = [torch.rand(4, 4)]
        titles = ["Matrix 1"]

        fig, axes = plot_cka_comparison(matrices, titles)

        assert fig is not None
        plt.close(fig)

    def test_show_true(self):
        matrices = [torch.rand(4, 4), torch.rand(4, 4)]
        titles = ["Matrix 1", "Matrix 2"]

        with unittest.mock.patch.object(plt, "show"):
            fig, axes = plot_cka_comparison(matrices, titles, show=True)

        assert fig is not None
        plt.close(fig)


class TestSaveFigure:
    def test_basic_save(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.png")
            save_figure(fig, path)

            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_custom_dpi(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_dpi.png")
            save_figure(fig, path, dpi=300)

            assert os.path.exists(path)

    def test_custom_bbox_inches(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_bbox.png")
            save_figure(fig, path, bbox_inches="tight")

            assert os.path.exists(path)

    def test_transparent(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_transparent.png")
            save_figure(fig, path, transparent=True)

            assert os.path.exists(path)

    def test_figure_closed_after_save(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        fig_num = fig.number

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_close.png")
            save_figure(fig, path)

            assert fig_num not in plt.get_fignums()

    def test_save_pdf(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pdf")
            save_figure(fig, path)

            assert os.path.exists(path)
