#!/usr/bin/env python3
import argparse
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot


def remove_padded_values(arr, pad_val=-1):
    if pad_val in arr:
        return np.delete(arr, np.where(arr == pad_val)[0])
    return arr


def group_histograms(arrays, role):
    ref = arrays[0]
    vmin, vmax = int(ref.min()), int(ref.max())
    bin_edges = np.arange(vmin - 0.5, vmax + 1.5, 1)

    n = len(arrays)
    cols, rows = 3, ceil(n / 3)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = axes.ravel()

    for idx, (data, ax) in enumerate(zip(arrays, axes)):
        ax.hist(data, bins=bin_edges, edgecolor="black")
        ax.set_xticks(np.arange(vmin, vmax + 1))
        ax.tick_params(axis="x", rotation=90)
        ax.set_title(f"{role} {idx} routing")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig


def load_arrays(path):
    with uproot.open(path) as f:
        tree = f["Events"]
        arrays = tree.arrays(library="np")

    block_data, cls_block_data = [], []
    for key in tree.keys():
        clean = remove_padded_values(arrays[key])
        if key.startswith("block_"):
            block_data.append(clean)
        else:
            cls_block_data.append(clean)

    return block_data, cls_block_data


def main():
    parser = argparse.ArgumentParser(
        description="Show encoder/decoder routing histograms in two windows."
    )
    parser.add_argument("rootfile", help="Path to the .root file")
    args = parser.parse_args()

    root_path = Path(args.rootfile)
    if not root_path.exists():
        parser.error(f"File not found: {root_path}")

    block_data, cls_block_data = load_arrays(root_path)

    fig_enc = group_histograms(block_data, "Encoder")
    fig_dec = group_histograms(cls_block_data, "Decoder")

    plt.show()


if __name__ == "__main__":
    main()