"""Single file module to create a TMAP plot with Faerun and Matplotlib.

Usage:
    >>> from create_tmap import create_tmap
    >>> from rdkit import Chem
    >>> from mhfp.encoder import MHFPEncoder
    >>> smis = ["CCO", "CCN", "CCC", "CCF", "CCCl", "CCBr"]
    >>> enc = MHFPEncoder()
    >>> fps = [enc.encode(smi) for smi in smis]
    >>> bg = [True, True, False, False, False, False]
    >>> groups = ["Background", "Background", "A", "A", "B", "B"]
    >>> save_path = "tmap_dir/tmap_name"
    >>> create_tmap(save_path, smis, fps, bg, groups)
    >>> # The TMAP faerun plot will be saved as "tmap_dir/tmap_name.html"
    >>> # and the TMAP matplotlib plot as "tmap_dir/tmap_name.png".
    >>> # Additional data will be saved as "tmap_dir/tmap_name.js".
    >>> # The points and tree data will be saved as
    >>> # "tmap_dir/tmap_name_points.csv" and "tmap_dir/tmap_name_tree.csv"
    >>> # respectively.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.colors as mpl_colors
import numpy.random as npr
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from matplotlib.figure import Figure

    from chem_tools._typing import StrPath
    from chem_tools.external import _tmap as tm
else:
    try:
        import tmap as tm
    except ImportError:
        from chem_tools.external import _tmap as tm
logger = logging.getLogger(__package__)

_TAB_20 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#AEC7E8",
    "#FFBB78",
    "#98DF8A",
    "#FF9896",
    "#C5B0D5",
    "#C49C94",
    "#F7B6D2",
    "#C7C7C7",
    "#DBDB8D",
    "#9EDAE5",
]
TMAP_COLORMAP = mpl_colors.ListedColormap(_TAB_20, name="tmap")


def create_lsh_forest(
    fps: Sequence[tm.VectorUint | np.ndarray],
) -> tuple[
    tm.LSHForest,
    tm.VectorFloat,
    tm.VectorFloat,
    tm.VectorUint,
    tm.VectorUint,
    tm.GraphProperties,
]:
    """Create an LSHForest object for the TMAPHelper object.

    Parameters:
    fps: A sequence of fingerprint vectors.

    Returns:
        A tuple containing the LSHForest object, x and y coordinates,
        source and target indices, and graph properties.

    Raises:
        ImportError: If TMAP is not installed.
    """
    dims = len(fps[0])
    lf = tm.LSHForest(dims, 128)
    logger.debug("Creating LSHForest...")
    vector_fps = [tm.VectorUint(fp) for fp in fps]
    lf.batch_add(vector_fps)
    logger.debug("Added fingerprints to LSHForest.")
    lf.index()
    logger.debug("LSHForest created.")
    cfg = tm.LayoutConfiguration()
    cfg.k = 100
    cfg.sl_repeats = 2
    cfg.mmm_repeats = 2
    cfg.node_size = 2
    layout = tm.layout_from_lsh_forest(lf, config=cfg)
    return (lf, *layout)


def plot_matplotlib(
    x: Sequence[float] | tm.VectorFloat,
    y: Sequence[float] | tm.VectorFloat,
    s: Sequence[int] | tm.VectorUint,
    t: Sequence[int] | tm.VectorUint,
    bg: Sequence[bool],
    groups: Sequence[str],
) -> tuple[Figure, pd.DataFrame, pd.DataFrame]:
    """Plot the TMAP using Matplotlib.

    `x`, `y`, `s`, and `t` are an LSHForest layout.

    Parameters:
        x: X coordinates of the points.
        y: Y coordinates of the points.
        s: Source indices for the tree edges.
        t: Target indices for the tree edges.
        bg: Background flag for each point.
        groups: Group labels for each point.

    Returns:
        A tuple containing the Matplotlib figure,
        points DataFrame, and tree DataFrame.
    """
    from matplotlib import pyplot as plt

    points = pd.DataFrame({"x": x, "y": y, "bg": bg, "group": groups})
    tree = pd.DataFrame({"from": s, "to": t})

    from_points = points.loc[tree["from"], ["x", "y"]]
    to_points = points.loc[tree["to"], ["x", "y"]]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        [from_points["x"], to_points["x"]],
        [from_points["y"], to_points["y"]],
        linewidth=0.2,
        zorder=1,
        color="black",
    )
    fg_points = points[~points["bg"]]
    for group, group_points in fg_points.groupby("group"):
        ax.scatter(group_points["x"], group_points["y"], s=5, label=group)

    ax.legend(loc="upper right")
    ax.axis("off")

    fig.tight_layout()
    return fig, points, tree


def plot_faerun(
    save_path: StrPath,
    x: Sequence[float] | tm.VectorFloat,
    y: Sequence[float] | tm.VectorFloat,
    s: Sequence[int] | tm.VectorUint,
    t: Sequence[int] | tm.VectorUint,
    smis: Sequence[str],
    bg: Sequence[bool],
    groups: Sequence[str],
) -> None:
    """Plot the TMAP using Faerun.

    `x`, `y`, `s`, and `t` are an LSHForest layout.
    Saves the plot as an HTML file to `save_path` with suffix ".html"
    and additional data with suffix ".js".

    Parameters:
    save_path: Path to save the Faerun plot.
    x: X coordinates of the points.
    y: Y coordinates of the points.
    s: Source indices for the tree edges.
    t: Target indices for the tree edges.
    smis: SMILES strings for each point.
    bg: Background flag for each point.
    groups: Group labels for each point.
    """
    import faerun

    save_path = Path(save_path)
    name = save_path.stem
    f = faerun.Faerun(
        clear_color="#FFFFFF",  # background color: white
        coords=False,
        view="front",
    )
    capital_name = name.capitalize()
    fg_name = capital_name
    bg_name = capital_name + "_background"
    tree_name = capital_name + "_tree"
    fg_x = [x[ix] for ix, is_bg in enumerate(bg) if not is_bg]
    fg_y = [y[ix] for ix, is_bg in enumerate(bg) if not is_bg]
    fg_groups = [group for group, is_bg in zip(groups, bg, strict=True) if not is_bg]
    fg_legend_labels, fg_data = faerun.Faerun.create_categories(fg_groups)
    fg_labels = [
        f"{smi}__{ix}__{group}"
        for ix, (smi, group, is_bg) in enumerate(zip(smis, groups, bg, strict=True))
        if not is_bg
    ]
    logger.debug("Adding background scatter.")
    f.add_scatter(
        bg_name,
        {"x": x, "y": y, "c": [0] * len(x)},
        interactive=False,
        shader="smoothCircle",
        point_scale=0,
        categorical=True,
    )
    if fg_x:
        logger.debug("Adding foreground scatter.")
        f.add_scatter(
            fg_name,
            {"x": fg_x, "y": fg_y, "c": [fg_data], "labels": fg_labels},
            selected_labels=["SMILES", "ID", "Group"],
            shader="smoothCircle",
            point_scale=5,
            has_legend=True,
            interactive=True,
            title_index=0,
            legend_title="",
            colormap=[TMAP_COLORMAP],
            categorical=[True],
            legend_labels=[fg_legend_labels],
            series_title=["Group"],
            max_legend_label=[None],
            min_legend_label=[None],
        )
    f.add_tree(tree_name, {"from": s, "to": t}, point_helper=bg_name)
    f.plot(name, template="smiles", path=str(save_path.parent))


def create_tmap(
    save_path: StrPath,
    smis: Sequence[str],
    fps: Sequence[tm.VectorUint],
    bg: Sequence[bool],
    groups: Sequence[str],
    randomize: bool = False,
) -> None:
    """Plot a TMAP plot with Faerun and Matplotlib.

    Creates an LSHForest object from the input fingerprints.
    Saves the faerun plot as an HTML file to `save_path` with suffix ".html"
    and additional data with suffix ".js".
    The matplotlib plot is saved as a PNG file to `save_path` with suffix ".png".
    Additional data is saved as CSV files to `save_path` with suffix "_points.csv"
    and "_tree.csv".

    Parameters:
    save_path: Path to save the TMAP plot.
    smis: SMILES strings for each point.
    fps: Fingerprint vectors for each point.
    bg: Background flag for each point.
    groups: Group labels for each point.
    randomize: Whether to randomize the order of the points.

    Raises:
        ImportError: If TMAP is not installed.
        ValueError: If the input data has different lengths.
    """
    if importlib.util.find_spec("tmap") is None:
        raise ImportError("TMAP is not installed.")

    if not all(len(fp) == len(fps[0]) for fp in fps):
        raise ValueError("Input data must have the same length.")

    if randomize:
        order = npr.permutation(len(fps))
        fps = [fps[i] for i in order]
        smis = [smis[i] for i in order]
        bg = [bg[i] for i in order]
        groups = [groups[i] for i in order]

    save_path = Path(save_path)
    name = save_path.stem

    _, x, y, s, t, _ = create_lsh_forest(fps)

    fig, points, tree = plot_matplotlib(x, y, s, t, bg, groups)

    fig.savefig(save_path.with_suffix(".png"))
    points.to_csv(save_path.parent / f"{name}_points.csv")
    tree.to_csv(save_path.parent / f"{name}_tree.csv", index=False)

    plot_faerun(save_path, x, y, s, t, smis, bg, groups)


__all__ = ["create_lsh_forest", "create_tmap", "plot_faerun", "plot_matplotlib"]
