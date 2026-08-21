import os
import tempfile
import unittest.mock
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from cka.viz import (
    plot_cka_comparison,
    plot_cka_heatmap,
    plot_cka_layer_trend,
    plot_cka_trend,
    plot_cka_trend_with_range,
    save_figure,
)


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

    def test_annot_true(self):
        matrix = torch.rand(3, 3)

        fig, ax = plot_cka_heatmap(matrix, annot=True)

        assert len(ax.texts) == matrix.numel()
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

        assert len(ax.texts) == matrix.numel() - 1
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

    def test_3d_values_raise(self):
        with pytest.raises(ValueError, match="values must be 1D or 2D"):
            plot_cka_trend(torch.rand(2, 3, 4))

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one series"):
            plot_cka_trend([])

    def test_list_of_scalars(self):
        fig, ax = plot_cka_trend([0.1, 0.2, 0.3, 0.4])

        assert len(ax.get_lines()) == 1
        plt.close(fig)

    def test_list_of_2d_arrays_raises(self):
        with pytest.raises(ValueError, match="Each series must be 1D"):
            plot_cka_trend([np.random.rand(2, 5), np.random.rand(2, 5)])

    def test_unsupported_values_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported values type"):
            plot_cka_trend({"a": 1})

    def test_unequal_series_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            plot_cka_trend([torch.rand(5), torch.rand(6)])

    def test_2d_numpy_x_values(self):
        values = torch.rand(2, 5)
        x_values = np.stack([np.arange(5), np.arange(5) + 10])

        fig, ax = plot_cka_trend(values, x_values=x_values)

        assert list(ax.get_lines()[1].get_xdata()) == list(x_values[1])
        plt.close(fig)

    def test_2d_numpy_x_values_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="x_values shape must match"):
            plot_cka_trend(torch.rand(2, 5), x_values=np.zeros((3, 5)))

    def test_nested_list_x_values(self):
        values = torch.rand(2, 5)
        x_values = [np.arange(5), np.arange(5) + 1]

        fig, ax = plot_cka_trend(values, x_values=x_values)

        assert list(ax.get_lines()[0].get_xdata()) == list(x_values[0])
        plt.close(fig)

    def test_nested_x_values_wrong_line_count_raises(self):
        with pytest.raises(ValueError, match="x_values must match number of lines"):
            plot_cka_trend(torch.rand(2, 5), x_values=[np.arange(5)])

    def test_nested_x_values_wrong_length_raises(self):
        with pytest.raises(ValueError, match="1D and match length"):
            plot_cka_trend(torch.rand(2, 5), x_values=[np.arange(5), np.arange(4)])

    def test_nested_x_values_not_1d_raises(self):
        with pytest.raises(ValueError, match="1D and match length"):
            plot_cka_trend(
                torch.rand(2, 5),
                x_values=[np.ones((2, 5)), np.ones((2, 5))],
            )

    def test_1d_x_values_wrong_length_raises(self):
        with pytest.raises(ValueError, match="1D and match series length"):
            plot_cka_trend(torch.rand(5), x_values=[0, 1, 2])

    def test_single_color_list_broadcast(self):
        values = torch.rand(3, 5)

        fig, ax = plot_cka_trend(values, colors=["red"])

        assert fig is not None
        plt.close(fig)

    def test_scalar_color_broadcast(self):
        values = torch.rand(3, 5)

        fig, ax = plot_cka_trend(values, colors="red")

        assert fig is not None
        plt.close(fig)

    def test_colors_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="colors length must match"):
            plot_cka_trend(torch.rand(3, 5), colors=["red", "blue"])

    def test_color_overflow_repeat(self):
        fig, ax = plot_cka_trend(torch.rand(12, 4), color_overflow="repeat")

        assert len(ax.get_lines()) == 12
        plt.close(fig)

    def test_color_overflow_tab20(self):
        fig, ax = plot_cka_trend(torch.rand(12, 4), color_overflow="tab20")

        assert len(ax.get_lines()) == 12
        plt.close(fig)

    def test_color_overflow_tab20_wrap(self):
        fig, ax = plot_cka_trend(torch.rand(21, 4), color_overflow="tab20")

        assert len(ax.get_lines()) == 21
        plt.close(fig)

    def test_color_overflow_variant(self):
        fig, ax = plot_cka_trend(torch.rand(12, 4), color_overflow="variant")

        assert len(ax.get_lines()) == 12
        plt.close(fig)

    def test_show_range_without_values_raises(self):
        with pytest.raises(ValueError, match="show_range=True requires range_values"):
            plot_cka_trend(torch.rand(5), show_range=True)

    def test_show_range_lower_upper_tuple(self):
        values = torch.rand(5)
        lower = values - 0.1
        upper = values + 0.1

        fig, ax = plot_cka_trend(
            values, show_range=True, range_values=(lower, upper)
        )

        assert len(ax.collections) > 0
        plt.close(fig)

    def test_show_range_tuple_broadcast(self):
        values = torch.rand(3, 5)
        lower = torch.zeros(5)
        upper = torch.ones(5)

        fig, ax = plot_cka_trend(
            values, show_range=True, range_values=(lower, upper)
        )

        assert len(ax.collections) == 3
        plt.close(fig)

    def test_show_range_std_broadcast(self):
        values = torch.rand(3, 5)
        std = torch.full((5,), 0.05)

        fig, ax = plot_cka_trend(values, show_range=True, range_values=std)

        assert len(ax.collections) == 3
        plt.close(fig)

    def test_show_range_std_line_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="range_values must match number of lines"):
            plot_cka_trend(
                torch.rand(2, 5),
                show_range=True,
                range_values=torch.rand(3, 5),
            )

    def test_show_range_tuple_line_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="range_values must match number of lines"):
            plot_cka_trend(
                torch.rand(3, 5),
                show_range=True,
                range_values=(torch.rand(2, 5), torch.rand(2, 5)),
            )

    def test_show_range_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="range_values length must match"):
            plot_cka_trend(
                torch.rand(5),
                show_range=True,
                range_values=(torch.rand(4), torch.rand(4)),
            )


class TestPlotCkaTrendWithRange:
    def test_single_group_2d(self):
        values = torch.rand(4, 10)

        fig, ax = plot_cka_trend_with_range(values)

        lines = ax.get_lines()
        assert len(lines) == 1
        assert len(ax.collections) > 0
        plt.close(fig)

    def test_multiple_groups_3d(self):
        values = torch.rand(3, 4, 10)

        fig, ax = plot_cka_trend_with_range(values)

        lines = ax.get_lines()
        assert len(lines) == 3
        assert len(ax.collections) >= 3
        plt.close(fig)

    def test_list_of_2d_groups(self):
        values = [torch.rand(4, 10), torch.rand(4, 10)]

        fig, ax = plot_cka_trend_with_range(values)

        assert len(ax.get_lines()) == 2
        plt.close(fig)

    def test_1d_values_raise(self):
        with pytest.raises(ValueError, match="values must be 2D or 3D"):
            plot_cka_trend_with_range(torch.rand(10))

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one group"):
            plot_cka_trend_with_range([])

    def test_list_of_1d_groups_raises(self):
        with pytest.raises(ValueError, match="Each group must be 2D"):
            plot_cka_trend_with_range([torch.rand(10), torch.rand(10)])

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported values type"):
            plot_cka_trend_with_range({"a": 1})


class TestPlotCkaLayerTrend:
    def test_single_matrix(self):
        matrix = torch.rand(5, 5)
        layers = [f"layer{i}" for i in range(5)]

        fig, ax = plot_cka_layer_trend(matrix, layers=layers)

        assert fig is not None
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        assert xticklabels == layers
        plt.close(fig)

    def test_list_of_matrices(self):
        matrices = [torch.rand(4, 4), torch.rand(4, 4)]

        fig, ax = plot_cka_layer_trend(matrices, legend=True)

        assert len(ax.get_lines()) == 2
        plt.close(fig)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="at least one matrix"):
            plot_cka_layer_trend([])

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported cka_matrices type"):
            plot_cka_layer_trend({"a": 1})

    def test_layers_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="layers length must match"):
            plot_cka_layer_trend(torch.rand(4, 4), layers=["a", "b"])

    def test_layer_name_depth(self):
        matrix = torch.rand(2, 2)
        layers = ["block.conv.weight", "block.relu.weight"]

        fig, ax = plot_cka_layer_trend(
            matrix, layers=layers, layer_name_depth=1
        )

        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        assert xticklabels == ["weight", "weight"]
        plt.close(fig)

    def test_numeric_ticks_without_layers(self):
        fig, ax = plot_cka_layer_trend(torch.rand(3, 3))

        ticks = list(ax.get_xticks())
        assert ticks == [0, 1, 2]
        plt.close(fig)

    def test_show_true(self):
        with unittest.mock.patch.object(plt, "show") as mock_show:
            fig, ax = plot_cka_layer_trend(torch.rand(3, 3), show=True)

        mock_show.assert_called_once()
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

    def test_heatmap_transparent_png(self):
        matrix = torch.rand(4, 4)
        fig, _ = plot_cka_heatmap(matrix)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cka_heatmap.png")
            save_figure(fig, path, transparent=True)

            from PIL import Image

            image = Image.open(path)
            assert image.mode == "RGBA"
            alpha = np.array(image)[:, :, 3]
            assert np.any(alpha < 255)

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


class TestTextColor:
    def test_heatmap_and_colorbar(self):
        fig, ax = plot_cka_heatmap(torch.rand(4, 4), text_color="white", title="H")
        assert ax.title.get_color() == "white"
        assert ax.xaxis.label.get_color() == "white"
        cax = next(a for a in fig.axes if a is not ax)
        assert cax.yaxis.label.get_color() == "white"
        plt.close(fig)

    def test_trend_legend_and_layer_ticks(self):
        fig, ax = plot_cka_trend(
            torch.rand(2, 5),
            labels=["a", "b"],
            legend=True,
            title="T",
            text_color="white",
        )
        assert ax.title.get_color() == "white"
        assert ax.xaxis.label.get_color() == "white"
        for text in ax.get_legend().get_texts():
            assert text.get_color() == "white"
        assert ax.get_legend().get_frame().get_facecolor()[3] > 0.0
        plt.close(fig)

        fig, ax = plot_cka_layer_trend(
            torch.eye(3), layers=["l1", "l2", "l3"], text_color="white"
        )
        for label in ax.get_xticklabels():
            assert label.get_color() == "white"
        plt.close(fig)

    def test_legend_frame_follows_transparent_save(self):
        values = torch.rand(2, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            fig, ax = plot_cka_trend(values, labels=["a", "b"], legend=True)
            seen = {}
            real_savefig = fig.savefig

            def spy(*args, **kwargs):
                seen["alpha"] = ax.get_legend().get_frame().get_facecolor()[3]
                return real_savefig(*args, **kwargs)

            fig.savefig = spy
            save_figure(fig, os.path.join(tmpdir, "t.png"), transparent=True)
            assert seen["alpha"] == 0.0

            fig, ax = plot_cka_trend(values, labels=["a", "b"], legend=True)
            seen = {}
            real_savefig = fig.savefig

            def spy_opaque(*args, **kwargs):
                seen["alpha"] = ax.get_legend().get_frame().get_facecolor()[3]
                return real_savefig(*args, **kwargs)

            fig.savefig = spy_opaque
            save_figure(fig, os.path.join(tmpdir, "o.png"), transparent=False)
            assert seen["alpha"] > 0.0
