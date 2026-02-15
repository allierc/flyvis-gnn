"""Plot functions for flyvis data generation and analysis.

All functions accept plain numpy arrays (not NeuronState) and a
FigureStyle instance. This keeps them independent of the dataclass
layout and reusable from any caller (generator, trainer, analysis).
"""
from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from flyvis_gnn.figure_style import FigureStyle, default_style


# --------------------------------------------------------------------------- #
#  Shared constants
# --------------------------------------------------------------------------- #

INDEX_TO_NAME: dict[int, str] = {
    0: 'am', 1: 'c2', 2: 'c3', 3: 'ct1(lo1)', 4: 'ct1(m10)',
    5: 'l1', 6: 'l2', 7: 'l3', 8: 'l4', 9: 'l5',
    10: 'lawf1', 11: 'lawf2', 12: 'mi1', 13: 'mi10', 14: 'mi11',
    15: 'mi12', 16: 'mi13', 17: 'mi14', 18: 'mi15', 19: 'mi2',
    20: 'mi3', 21: 'mi4', 22: 'mi9', 23: 'r1', 24: 'r2',
    25: 'r3', 26: 'r4', 27: 'r5', 28: 'r6', 29: 'r7', 30: 'r8',
    31: 't1', 32: 't2', 33: 't2a', 34: 't3', 35: 't4a',
    36: 't4b', 37: 't4c', 38: 't4d', 39: 't5a', 40: 't5b',
    41: 't5c', 42: 't5d', 43: 'tm1', 44: 'tm16', 45: 'tm2',
    46: 'tm20', 47: 'tm28', 48: 'tm3', 49: 'tm30', 50: 'tm4',
    51: 'tm5y', 52: 'tm5a', 53: 'tm5b', 54: 'tm5c', 55: 'tm9',
    56: 'tmy10', 57: 'tmy13', 58: 'tmy14', 59: 'tmy15',
    60: 'tmy18', 61: 'tmy3', 62: 'tmy4', 63: 'tmy5a', 64: 'tmy9',
}

ANATOMICAL_ORDER: list[Optional[int]] = [
    None, 23, 24, 25, 26, 27, 28, 29, 30,
    5, 6, 7, 8, 9, 10, 11, 12,
    19, 20, 21, 22,
    13, 14, 15, 16, 17, 18,
    43, 45, 48, 50, 44, 46, 47, 49, 51, 52, 53, 54, 55,
    61, 62, 63, 56, 57, 58, 59, 60, 64,
    1, 2, 4, 3,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    0,
]


# --------------------------------------------------------------------------- #
#  Plot functions
# --------------------------------------------------------------------------- #

def plot_spatial_activity_grid(
    positions: np.ndarray,
    voltages: np.ndarray,
    stimulus: np.ndarray,
    neuron_types: np.ndarray,
    output_path: str,
    calcium: Optional[np.ndarray] = None,
    n_input_neurons: Optional[int] = None,
    index_to_name: Optional[dict] = None,
    anatomical_order: Optional[list] = None,
    style: FigureStyle = default_style,
) -> None:
    """8x9 or 16x9 hex scatter grid of per-neuron-type spatial activity.

    Args:
        positions: (N, 2) spatial positions for hex scatter.
        voltages: (N,) voltage per neuron.
        stimulus: (n_input,) stimulus values for input neurons.
        neuron_types: (N,) integer neuron type per neuron.
        output_path: where to save the figure.
        calcium: (N,) calcium values (if not None, adds bottom 8 rows).
        n_input_neurons: number of input neurons (defaults to len(stimulus)).
        index_to_name: type index -> name mapping. Defaults to INDEX_TO_NAME.
        anatomical_order: panel ordering. Defaults to ANATOMICAL_ORDER.
        style: FigureStyle instance.
    """
    names = index_to_name or INDEX_TO_NAME
    order = anatomical_order or ANATOMICAL_ORDER
    n_inp = n_input_neurons or len(stimulus)
    include_calcium = calcium is not None

    n_cols = 9
    n_rows = 16 if include_calcium else 8
    panel_w, panel_h = 2.0, 1.8
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_w * n_cols, panel_h * n_rows),
        facecolor=style.background,
    )
    plt.subplots_adjust(hspace=1.2)
    axes_flat = axes.flatten()

    # hide trailing panels in voltage section
    n_panels = len(order)
    for i in range(n_panels, n_cols * 8):
        if i < len(axes_flat):
            axes_flat[i].set_visible(False)
    if include_calcium:
        for i in range(n_panels + n_cols * 8, len(axes_flat)):
            axes_flat[i].set_visible(False)

    vmin_v, vmax_v = style.hex_voltage_range
    vmin_s, vmax_s = style.hex_stimulus_range
    vmin_ca, vmax_ca = style.hex_calcium_range

    for panel_idx, type_idx in enumerate(order):
        # --- voltage panel ---
        ax_v = axes_flat[panel_idx]
        _draw_hex_panel(
            ax_v, type_idx, positions, voltages, stimulus,
            neuron_types, n_inp, names,
            cmap=style.cmap, vmin=vmin_v, vmax=vmax_v,
            stim_cmap=style.cmap, stim_vmin=vmin_s, stim_vmax=vmax_s,
            style=style,
        )

        # --- calcium panel (if present) ---
        if include_calcium:
            ax_ca = axes_flat[panel_idx + n_cols * 8]
            if type_idx is None:
                # stimulus panel (same as voltage section)
                ax_ca.scatter(
                    positions[:n_inp, 0], positions[:n_inp, 1],
                    s=style.hex_stimulus_marker_size, c=stimulus,
                    cmap=style.cmap, vmin=vmin_s, vmax=vmax_s,
                    marker=style.hex_marker, alpha=1.0, linewidths=0,
                )
                ax_ca.set_title(style._label('stimuli'), fontsize=style.font_size)
            else:
                mask = neuron_types == type_idx
                count = int(np.sum(mask))
                name = names.get(type_idx, f'type_{type_idx}')
                if count > 0:
                    ax_ca.scatter(
                        positions[:count, 0], positions[:count, 1],
                        s=style.hex_marker_size, c=calcium[mask],
                        cmap=style.cmap_calcium, vmin=vmin_ca, vmax=vmax_ca,
                        marker=style.hex_marker, alpha=1, linewidths=0,
                    )
                ax_ca.set_title(style._label(name), fontsize=style.font_size)
            ax_ca.set_facecolor(style.background)
            ax_ca.set_xticks([])
            ax_ca.set_yticks([])
            ax_ca.set_aspect('equal')
            for spine in ax_ca.spines.values():
                spine.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95 if not include_calcium else 0.92, bottom=0.05)
    style.savefig(fig, output_path)


def plot_kinograph(
    activity: np.ndarray,
    stimulus: np.ndarray,
    output_path: str,
    rank_90_act: int = 0,
    rank_99_act: int = 0,
    rank_90_inp: int = 0,
    rank_99_inp: int = 0,
    zoom_size: int = 200,
    style: FigureStyle = default_style,
) -> None:
    """2x2 kinograph: full activity + zoom, full stimulus + zoom.

    Args:
        activity: (n_neurons, n_frames) transposed voltage array.
        stimulus: (n_input_neurons, n_frames) transposed stimulus array.
        output_path: where to save the figure.
        rank_90_act: effective rank at 90% variance (activity).
        rank_99_act: effective rank at 99% variance (activity).
        rank_90_inp: effective rank at 90% variance (input).
        rank_99_inp: effective rank at 99% variance (input).
        zoom_size: size of zoom window in neurons and frames.
        style: FigureStyle instance.
    """
    n_neurons, n_frames = activity.shape
    n_input, _ = stimulus.shape
    vmax_act = np.abs(activity).max()
    vmax_inp = np.abs(stimulus).max() * 1.2
    zoom_f = min(zoom_size, n_frames)
    zoom_n_act = min(zoom_size, n_neurons)
    zoom_n_inp = min(zoom_size, n_input)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(style.figure_height * 3.5, style.figure_height * 2.5),
        facecolor=style.background,
        gridspec_kw={'width_ratios': [2, 1]},
    )

    imshow_kw = dict(aspect='auto', cmap=style.cmap, origin='lower', interpolation='nearest')

    # top-left: full activity
    ax = axes[0, 0]
    im = ax.imshow(activity, vmin=-vmax_act, vmax=vmax_act, **imshow_kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=style.tick_font_size)
    style.ylabel(ax, 'neurons')
    style.xlabel(ax, 'time (frames)')
    ax.set_xticks([0, n_frames - 1])
    ax.set_xticklabels([0, n_frames], fontsize=style.tick_font_size)
    ax.set_yticks([0, n_neurons - 1])
    ax.set_yticklabels([1, n_neurons], fontsize=style.tick_font_size)
    style.annotate(ax, f'rank(90%)={rank_90_act}  rank(99%)={rank_99_act}', (0.02, 0.97), va='top', ha='left')

    # top-right: zoom activity
    ax = axes[0, 1]
    im = ax.imshow(activity[:zoom_n_act, :zoom_f], vmin=-vmax_act, vmax=vmax_act, **imshow_kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=style.tick_font_size)
    style.ylabel(ax, 'neurons')
    style.xlabel(ax, 'time (frames)')
    ax.set_xticks([0, zoom_f - 1])
    ax.set_xticklabels([0, zoom_f], fontsize=style.tick_font_size)
    ax.set_yticks([0, zoom_n_act - 1])
    ax.set_yticklabels([1, zoom_n_act], fontsize=style.tick_font_size)

    # bottom-left: full stimulus
    ax = axes[1, 0]
    im = ax.imshow(stimulus, vmin=-vmax_inp, vmax=vmax_inp, **imshow_kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=style.tick_font_size)
    style.ylabel(ax, 'input neurons')
    style.xlabel(ax, 'time (frames)')
    ax.set_xticks([0, stimulus.shape[1] - 1])
    ax.set_xticklabels([0, stimulus.shape[1]], fontsize=style.tick_font_size)
    ax.set_yticks([0, n_input - 1])
    ax.set_yticklabels([1, n_input], fontsize=style.tick_font_size)
    style.annotate(ax, f'rank(90%)={rank_90_inp}  rank(99%)={rank_99_inp}', (0.02, 0.97), va='top', ha='left')

    # bottom-right: zoom stimulus
    ax = axes[1, 1]
    im = ax.imshow(stimulus[:zoom_n_inp, :zoom_f], vmin=-vmax_inp, vmax=vmax_inp, **imshow_kw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=style.tick_font_size)
    style.ylabel(ax, 'input neurons')
    style.xlabel(ax, 'time (frames)')
    ax.set_xticks([0, zoom_f - 1])
    ax.set_xticklabels([0, zoom_f], fontsize=style.tick_font_size)
    ax.set_yticks([0, zoom_n_inp - 1])
    ax.set_yticklabels([1, zoom_n_inp], fontsize=style.tick_font_size)

    plt.tight_layout()
    style.savefig(fig, output_path)


def plot_activity_traces(
    activity: np.ndarray,
    output_path: str,
    n_traces: int = 100,
    max_frames: int = 10000,
    n_input_neurons: int = 0,
    style: FigureStyle = default_style,
) -> None:
    """Sampled neuron voltage traces stacked vertically.

    Args:
        activity: (n_neurons, n_frames) transposed voltage array.
        output_path: where to save the figure.
        n_traces: number of neurons to sample.
        max_frames: truncate x-axis at this frame count.
        n_input_neurons: shown as annotation.
        style: FigureStyle instance.
    """
    n_neurons, n_frames = activity.shape
    n_traces = min(n_traces, n_neurons)
    sampled_idx = np.sort(np.random.choice(n_neurons, n_traces, replace=False))
    sampled = activity[sampled_idx]
    offset = sampled + 2 * np.arange(n_traces)[:, None]

    fig, ax = style.figure(aspect=3.0)
    ax.plot(offset.T, linewidth=0.5, alpha=0.7, color=style.foreground)
    style.xlabel(ax, 'time (frames)')
    ax.set_yticks([])
    ax.set_xlim([0, min(n_frames, max_frames)])
    ax.set_ylim([offset[0].min() - 2, offset[-1].max() + 2])
    if n_input_neurons > 0:
        style.annotate(ax, f'{n_input_neurons} neurons', (0.98, 0.98), va='top', ha='right')

    style.savefig(fig, output_path)


def plot_selected_neuron_traces(
    activity: np.ndarray,
    type_list: np.ndarray,
    output_path: str,
    selected_types: Optional[list[int]] = None,
    start_frame: int = 63000,
    end_frame: int = 63500,
    index_to_name: Optional[dict] = None,
    step_v: float = 1.5,
    style: FigureStyle = default_style,
) -> None:
    """Traces for specific neuron types over a time window.

    Args:
        activity: (n_neurons, n_frames) full activity array.
        type_list: (n_neurons,) integer neuron type per neuron.
        output_path: where to save the figure.
        selected_types: list of type indices to plot. Defaults to
            [l1, mi1, mi2, r1, t1, t4a, t5a, tm1, tm4, tm9].
        start_frame: start of time window.
        end_frame: end of time window.
        index_to_name: type index -> name mapping. Defaults to INDEX_TO_NAME.
        step_v: vertical offset between traces.
        style: FigureStyle instance.
    """
    names = index_to_name or INDEX_TO_NAME
    if selected_types is None:
        selected_types = [5, 12, 19, 23, 31, 35, 39, 43, 50, 55]

    # find one neuron per selected type
    neuron_indices = []
    for stype in selected_types:
        indices = np.where(type_list == stype)[0]
        if len(indices) > 0:
            neuron_indices.append(indices[0])

    n_sel = len(neuron_indices)
    if n_sel == 0:
        return

    true_slice = activity[neuron_indices, start_frame:end_frame]

    fig, ax = style.figure(aspect=3.0)
    for i in range(n_sel):
        baseline = np.mean(true_slice[i])
        ax.plot(true_slice[i] - baseline + i * step_v,
                linewidth=style.line_width, c='green', alpha=0.75)

    for i in range(n_sel):
        ax.text(-100, i * step_v, style._label(names.get(selected_types[i], f'type_{selected_types[i]}')),
                fontsize=style.tick_font_size, va='center', color=style.foreground)

    ax.set_ylim([-step_v, n_sel * step_v])
    ax.set_yticks([])
    ax.set_xticks([0, end_frame - start_frame])
    ax.set_xticklabels([start_frame, end_frame], fontsize=style.tick_font_size)
    style.xlabel(ax, 'frame')

    plt.subplots_adjust(left=0.05)
    style.savefig(fig, output_path)


# --------------------------------------------------------------------------- #
#  Private helpers
# --------------------------------------------------------------------------- #

def _draw_hex_panel(
    ax, type_idx, positions, voltages, stimulus, neuron_types,
    n_input_neurons, names, cmap, vmin, vmax,
    stim_cmap, stim_vmin, stim_vmax, style,
):
    """Draw a single hex scatter panel (voltage or stimulus)."""
    if type_idx is None:
        ax.scatter(
            positions[:n_input_neurons, 0], positions[:n_input_neurons, 1],
            s=style.hex_stimulus_marker_size, c=stimulus,
            cmap=stim_cmap, vmin=stim_vmin, vmax=stim_vmax,
            marker=style.hex_marker, alpha=1.0, linewidths=0,
        )
        ax.set_title(style._label('stimuli'), fontsize=style.font_size)
    else:
        mask = neuron_types == type_idx
        count = int(np.sum(mask))
        name = names.get(type_idx, f'type_{type_idx}')
        if count > 0:
            ax.scatter(
                positions[:count, 0], positions[:count, 1],
                s=style.hex_marker_size, c=voltages[mask],
                cmap=cmap, vmin=vmin, vmax=vmax,
                marker=style.hex_marker, alpha=1, linewidths=0,
            )
        ax.set_title(style._label(name), fontsize=style.font_size)

    ax.set_facecolor(style.background)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
