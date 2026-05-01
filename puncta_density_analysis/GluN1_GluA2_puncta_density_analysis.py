# ChatGPT (OpenAI) and Claude (Antrophic) were partly used as assistance for code generation which was reviewed and verified by the creator.

# [1]   N. Otsu, “A Threshold Selection Method from Gray-Level Histograms,”
#       IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62–66,
#       Jan. 1979. doi: 10.1109/TSMC.1979.4310076.
#
# [2]   Y. Wang, C. Wang, P. Ranefall, G. J. Broussard, Y. Wang, G. Shi, B. Lyu,
#       C.-T. Wu, Y. Wang, L. Tian, and G. Yu, “SynQuant: An automatic tool to
#       quantify synapses from microscopy images,” Bioinformatics, vol. 36, no. 5,
#       pp. 1599–1606, Mar. 2020. doi: 10.1093/bioinformatics/btz760.

#Import libraries
import re
import struct
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import tifffile
from skimage import filters, morphology, measure
from skimage.draw import polygon as draw_polygon
from scipy import stats as spstats

# =============================================================================
# 0. GLOBAL FONT SETTINGS  
# =============================================================================
plt.rcParams.update({
    "font.size":            14,
    "axes.titlesize":       22,
    "axes.labelsize":       18,
    "xtick.labelsize":      13,
    "ytick.labelsize":      13,
    "legend.fontsize":      10,
    "legend.title_fontsize":12,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
})

# =============================================================================
# Original microscopy channel colors
# =============================================================================
CMAP_MAP2  = LinearSegmentedColormap.from_list("map2_red",    ["black", "#FF0000"])   # red
CMAP_GLUN1 = LinearSegmentedColormap.from_list("glun1_orange",["black", "#FF8C00"])   # orange
CMAP_GLUA2 = LinearSegmentedColormap.from_list("glua2_orange",["black", "#FF8C00"])   # orange

# =============================================================================
# USER SETTINGS
# =============================================================================

INPUT_DIR      = r"Folder with split .tif files of MAP2 with removed somas and GluN1 and GluA2 images"
OUTPUT_DIR     = r"Output folder"
ROI_BASE_DIR   = r"Folder with extracted Puncta per sample"
#
PIXEL_SIZE_UM  = 0.13    # µm / pixel pixelsize - adjust to microscope setup
SAVE_QC_IMAGES = True    # set False to skip per-sample pipeline figures

GLUN1_SUFFIX = "_GluN1.tif"
GLUA2_SUFFIX = "_GluA2.tif"
MAP2_SUFFIX  = "_MAP2_sr.tif"

# =============================================================================
# Analysis Parameter
# =============================================================================
# Set normalization percentile 
BG_PERCENTILE  = 5
SAT_PERCENTILE = 99.5

# MAP2 dendrite segmentation parameter were adjusted to optimize the segmentation results
# Gaussian blur filter
MAP2_GAUSS_SIGMA_GluN1 = 2      
MAP2_GAUSS_SIGMA_GluA2  = 2
# White top-hat filter
MAP2_TOPHAT_RADIUS_GluN1 = 5
MAP2_TOPHAT_RADIUS_GluA2 = 30
# Otsu Threshold multiplier
MAP2_THRESH_MULT_GluN1 = 0.5
MAP2_THRESH_MULT_GluA2 = 0.8
# Remove very small detected dendrites
MAP2_MIN_OBJ_SIZE      = 2
# close small gaps between dendrites 
MAP2_CLOSE_RADIUS      = 2
# Threshold to exclude removed soma regions
MAP2_PAINTED_BLACK_THR = 0.02  

# Dendrite dilation for ROI filtering
DENDRITE_DILATION = 2

# Figure / scale bar
SCALEBAR_UM         = 10
SCALEBAR_COLOR      = "white"
SCALEBAR_COLOR_MASK = "black"

# IQR outlier detection
IQR_MULTIPLIER = 1.5   # standard: 1.5 × IQR

# =============================================================================
# Assign plot colors
# =============================================================================
_set3 = sns.color_palette("Set3", 12)

condition_base_colors = {
    "Control": _set3[4],
    "cTBS":    _set3[5],
    "iTBS":    _set3[6],
}
_pools_n = 3
condition_shade_ramps = {
    cond: sns.dark_palette(base_color, n_colors=_pools_n + 2, reverse=True)[1:_pools_n + 1]
    for cond, base_color in condition_base_colors.items()
}

COLORS = {cond: ramp[0] for cond, ramp in condition_shade_ramps.items()}

# =============================================================================
# Sample name parser
# =============================================================================

_SAMPLE_RE = re.compile(r"^(cTBS|iTBS)_([CS])(\d+)$", re.IGNORECASE)


def parse_sample(name: str):
    # extract the important information from the sample name
    """Return (experiment, status, group) from e.g. 'cTBS_C2'."""
    m = _SAMPLE_RE.match(name)
    if not m:
        return "Unknown", "Unknown", "Unknown"
    experiment = m.group(1)
    status     = "Control" if m.group(2).upper() == "C" else experiment
    group      = f"{experiment}_{status}"
    return experiment, status, group


# =============================================================================
# Image helper functions
# =============================================================================

def load(path):
    # load .tif file
    img = tifffile.imread(str(path)).astype(np.float32)
    if img.ndim == 3:
        img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]
    return img


def normalize(img):
    # normalize each image between the 5th background percentile  and the 99.5th saturation percentile 
    bg  = np.percentile(img, BG_PERCENTILE)
    sat = np.percentile(img, SAT_PERCENTILE)
    if sat - bg < 1e-6:
        return np.zeros_like(img)
    return np.clip((img - bg) / (sat - bg), 0, 1)


def add_scale_bar(ax, image_shape, pixel_size_um,
                  scalebar_um=None, color="white", fontsize=10, margin_frac=0.03):
    # add scale bar to microscopic images
    if scalebar_um is None:
        scalebar_um = SCALEBAR_UM
    h, w       = image_shape
    bar_px     = scalebar_um / pixel_size_um
    margin_x   = w * margin_frac
    margin_y   = h * margin_frac
    bar_height = max(2, h * 0.012)
    x0 = w - margin_x - bar_px
    y0 = h - margin_y - bar_height
    ax.add_patch(Rectangle((x0, y0), bar_px, bar_height,
                            linewidth=0, edgecolor="none", facecolor=color))
    ax.text(x0 + bar_px / 2, y0 - margin_y * 0.4,
            f"{scalebar_um} µm", ha="center", va="bottom",
            color=color, fontsize=fontsize)

# =============================================================================
# MAP2 dendrite segmentation
# =============================================================================

def segment_dendrites(map2_norm, experiment):
    if experiment == "cTBS":
        gauss_sigma   = MAP2_GAUSS_SIGMA_GluN1
        tophat_radius = MAP2_TOPHAT_RADIUS_GluN1
    else:
        gauss_sigma   = MAP2_GAUSS_SIGMA_GluA2
        tophat_radius = MAP2_TOPHAT_RADIUS_GluA2
    # Apply gaussian blur fillter to reduce noise
    blurred       = filters.gaussian(map2_norm, sigma=gauss_sigma)
    painted_black = map2_norm < MAP2_PAINTED_BLACK_THR # extract blacked out soma regions
    blurred[painted_black] = 0.0 # set blured soma regions back to black
    # Apply white top-hat filter to extract bright soma regions
    tophat = morphology.white_tophat(blurred, morphology.disk(tophat_radius))
    tophat[painted_black] = 0.0
    # Apply Otsu threshold [1] only to tissue pixel, not to blacked out soma regions
    tissue_px = tophat[~painted_black & (tophat > 0.001)]
    if tissue_px.size < 100: tissue_px = tophat[tophat > 0]
    if tissue_px.size < 100: tissue_px = tophat.ravel()
    # Reduce Otsu threshold to include faint dendrites 
    if experiment == "cTBS":
        otsu_thresh = filters.threshold_otsu(tissue_px) * MAP2_THRESH_MULT_GluN1
    else:
        otsu_thresh = filters.threshold_otsu(tissue_px) * MAP2_THRESH_MULT_GluA2
    mask = (tophat > otsu_thresh) & ~painted_black
    # Remove very small dendrite sections
    mask = morphology.remove_small_objects(mask, min_size=MAP2_MIN_OBJ_SIZE)
    # Close gaps between nearby dendrites
    mask = morphology.closing(mask, morphology.disk(MAP2_CLOSE_RADIUS))
    # Skeletonize the binary mask to its centerlines
    skel = morphology.skeletonize(mask)
    return mask, skel


# =============================================================================
# ImageJ ROI loading
# =============================================================================
# For each GluN1 and GluA2 image, a set of detected puncta / ROIs was detected with the puncta detection Fiji (ImageJ distribution) plugin SynQuant [2].
def _parse_imagej_roi(data):
    """
    Parse a single ImageJ .roi file (binary format).
    Returns polygon vertices as (rows, cols) arrays — i.e. (ys, xs) in
    image coordinates.
    Supports traced / freehand / polygon ROI types (types 0, 7, 8).
    """
    roi_type = data[6]
    top      = struct.unpack('>h', data[8:10])[0]
    left     = struct.unpack('>h', data[10:12])[0]
    bottom   = struct.unpack('>h', data[12:14])[0]
    right    = struct.unpack('>h', data[14:16])[0]
    n_coords = struct.unpack('>h', data[16:18])[0]

    if n_coords <= 0:
        return None, None

    offset = 64
    needed = offset + n_coords * 4
    if len(data) < needed:
        return None, None

    x_offsets = [struct.unpack('>h', data[offset + i*2 : offset + i*2 + 2])[0]
                 for i in range(n_coords)]
    offset += n_coords * 2
    y_offsets = [struct.unpack('>h', data[offset + i*2 : offset + i*2 + 2])[0]
                 for i in range(n_coords)]

    xs = np.array([x + left for x in x_offsets])   # column coordinates
    ys = np.array([y + top  for y in y_offsets])    # row coordinates

    return ys, xs   # (rows, cols)


def load_rois_from_zip(roi_zip_path):
    """
    Load all .roi files from a zip archive (Fiji ROI Manager export).
    Returns a list of (rows, cols) polygon coordinate pairs.
    """
    rois = []
    with zipfile.ZipFile(str(roi_zip_path), 'r') as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith('.roi'):
                continue
            data = z.read(name)
            rows, cols = _parse_imagej_roi(data)
            if rows is not None and len(rows) >= 3:
                rois.append((rows, cols))
    return rois


def rois_to_labeled_image(rois, image_shape):
    """
    Rasterise a list of polygon ROIs into a labeled image.
    Each ROI gets a unique integer label (1, 2, 3, …).
    """
    labeled = np.zeros(image_shape, dtype=np.int32)
    h, w    = image_shape
    for idx, (rows, cols) in enumerate(rois, start=1):
        rr, cc = draw_polygon(rows, cols, shape=image_shape)
        labeled[rr, cc] = idx
    return labeled


def filter_rois_by_mask(rois, dend_mask_dilated, image_shape):
    """
    Keep only ROIs with their centroid sitting on the dilated dendrite mask.
    """
    h, w = image_shape
    kept = []
    for (rows, cols) in rois:
        cy = float(np.mean(rows))
        cx = float(np.mean(cols))
        ri, ci = int(round(cy)), int(round(cx))
        if 0 <= ri < h and 0 <= ci < w and dend_mask_dilated[ri, ci]:
            kept.append((rows, cols))
    return kept, len(rois), len(kept), len(rois) - len(kept)


# =============================================================================
# IQR Outlier detection
# =============================================================================

def detect_iqr_outliers(df, density_col, group_col="group", multiplier=None):
    # Detect Oulier based on the interquartile range
    if multiplier is None:
        multiplier = IQR_MULTIPLIER # 1.5 x IQR limit
    

    df = df.copy()
    outlier_col = f"{density_col}_outlier"
    df[outlier_col]   = False
    df["_iqr_lower"]  = np.nan
    df["_iqr_upper"]  = np.nan

    for grp, idx in df.groupby(group_col).groups.items():
        vals = df.loc[idx, density_col].dropna()
        if len(vals) < 4:
            continue
        # Define quartiles
        q1  = vals.quantile(0.25)
        q3  = vals.quantile(0.75)
        iqr = q3 - q1
        # Set Outlier boundaries
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        df.loc[idx, "_iqr_lower"] = lower
        df.loc[idx, "_iqr_upper"] = upper
        # Find outliers outside of the iqr threshold
        mask = (df.loc[idx, density_col] < lower) | (df.loc[idx, density_col] > upper)
        df.loc[idx[mask], outlier_col] = True

    return df


def print_outlier_report(df, density_col, channel_label):
    # Print the Outlier report 
    outlier_col = f"{density_col}_outlier"
    if outlier_col not in df.columns:
        print(f"  [INFO] No outlier column found for {channel_label} — skipping report.")
        return df

    n_outliers = df[outlier_col].sum()
    n_total    = df[density_col].notna().sum()

    print(f"\n{'═' * 60}")
    print(f"  IQR OUTLIER REPORT — {channel_label} density  "
          f"(k = {IQR_MULTIPLIER})")
    print(f"{'═' * 60}")

    groups = df.groupby("group")
    for grp_name, grp_df in groups:
        vals = grp_df[density_col].dropna()
        if vals.empty:
            continue
        q1    = vals.quantile(0.25)
        q3    = vals.quantile(0.75)
        iqr   = q3 - q1
        lower = grp_df["_iqr_lower"].iloc[0] if "_iqr_lower" in grp_df.columns else np.nan
        upper = grp_df["_iqr_upper"].iloc[0] if "_iqr_upper" in grp_df.columns else np.nan

        grp_outliers = grp_df[grp_df[outlier_col]]

        print(f"\n  Group: {grp_name}  (n = {len(vals)})")
        print(f"    Q1 = {q1:.5f}   Q3 = {q3:.5f}   IQR = {iqr:.5f}")
        print(f"    Fence:  [{lower:.5f},  {upper:.5f}]")

        if grp_outliers.empty:
            print(f"    → No outliers detected.")
        else:
            print(f"    → {len(grp_outliers)} outlier(s):")
            for _, row in grp_outliers.iterrows():
                val    = row[density_col]
                side   = "LOW" if val < lower else "HIGH"
                print(f"        {row['sample']:20s}  density = {val:.5f}  ({side})")

    print(f"\n  Summary: {n_outliers} outlier(s) out of {n_total} samples")
    if n_outliers > 0:
        print(f"  Flagged samples: "
              f"{', '.join(df.loc[df[outlier_col], 'sample'].tolist())}")
    print(f"{'═' * 60}\n")

    return df


# =============================================================================
# Puncta analysis pipeline plot
# =============================================================================

def save_qc_figure(result, channel_key, out_dir):
    sample        = result["sample"]
    channel_label = "GluN1" if channel_key == "glun1" else "GluA2"
    shape         = result["_map2_raw"].shape
    h, w          = shape

    # Pick the fluorescence colormap for the channel 
    if channel_key == "glun1":
        ch_cmap = CMAP_GLUN1
    else:
        ch_cmap = CMAP_GLUA2

    # colours for overlays
    color_on   = (0.4,  0.76, 0.65)   # teal  — kept puncta
    color_off  = (1.0,  0.0,  0.0)    # red   — rejected puncta (off mask)
    color_mask = (0.3,  0.6,  1.0)    # blue  — dilated mask overlay

    density_key = f"{channel_key}_density_per_um"
    n_pct_key   = f"{channel_key}_n_puncta"
    labeled     = result[f"_labeled_{channel_key}"]
    labeled_all = result[f"_labeled_all_{channel_key}"]
    ch_raw      = result[f"_{channel_key}_raw"]
    dend_mask   = result["_dend_mask"]
    skel        = result["_skeleton"]
    # Dilate dendrites to include puncta located on the side
    dend_dil    = morphology.dilation(dend_mask, morphology.disk(DENDRITE_DILATION))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.08})
    fig.suptitle(
        f"Dendrite segmentation & puncta detection  |  {channel_label}",
        fontsize=26, y=1.0)

    def _panel(ax, img, title, cmap, sb_color):
        ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        add_scale_bar(ax, shape, PIXEL_SIZE_UM, color=sb_color)

    # ── Row 0 ─────────────────────────────────────────────────────────────────
    panelTitleFontsize = 14
    # MAP2 raw → green fluorescence LUT
    _panel(axes[0, 0], result["_map2_raw"], "MAP2",
           CMAP_MAP2,  SCALEBAR_COLOR)
    _panel(axes[0, 1], dend_mask.astype(np.uint8), "Segmented dendrites",
           "Blues", SCALEBAR_COLOR_MASK)
    _panel(axes[0, 2], skel.astype(np.uint8), f"Skeleton {result['dendrite_length_um']:.0f} µm",
           "hot",   SCALEBAR_COLOR)

    # ── Row 1, panel 0: channel raw → fluorescence LUT ───────────────────────
    _panel(axes[1, 0], ch_raw, f"{channel_label} ", ch_cmap, SCALEBAR_COLOR)

    # ── Row 1, panel 1: ALL ROIs coloured by kept (teal) vs rejected (red) ───
    n_total    = result[f"_{channel_key}_n_total"]
    n_kept     = result[n_pct_key]
    n_rejected = result[f"_{channel_key}_n_rejected"]

    ax_ws = axes[1, 1]
    ax_ws.imshow(ch_raw, cmap="gray", interpolation="nearest")

    overlay_rgba = np.zeros((h, w, 4), dtype=float)
    rejected_mask = (labeled_all > 0) & (labeled == 0)
    kept_mask     = (labeled > 0)
    overlay_rgba[rejected_mask] = (*color_off, 0.3)
    overlay_rgba[kept_mask]     = (*color_on,  0.3)
    ax_ws.imshow(overlay_rgba, interpolation="nearest")

    ax_ws.set_title(f"{channel_label} puncta\n{n_kept} On - "
                    f"{n_rejected} Off - {n_total} Total",
                    fontsize=panelTitleFontsize, pad=20)
    ax_ws.set_xticks([]); ax_ws.set_yticks([])
    for sp in ax_ws.spines.values():
        sp.set_visible(False)
    ax_ws.legend(handles=[
        Patch(facecolor=(*color_on,  0.65), edgecolor="none", label="On Dendrites"),
        Patch(facecolor=(*color_off, 0.65), edgecolor="none", label="Off Dendrites"),
        
    ], loc="upper right", fontsize=10, framealpha=0.7,
       facecolor="black", labelcolor="white")
    add_scale_bar(ax_ws, shape, PIXEL_SIZE_UM, color=SCALEBAR_COLOR)

    # ── Row 1, panel 2: kept puncta + dilated mask overlay on channel ─────────
    ax_ov = axes[1, 2]
    ax_ov.imshow(ch_raw, cmap="gray", interpolation="nearest")

    dil_rgba             = np.zeros((h, w, 4), dtype=float)
    dil_rgba[dend_dil]   = (*color_mask, 0.20)
    ax_ov.imshow(dil_rgba, interpolation="nearest")

    punct_rgba = np.zeros((h, w, 4), dtype=float)
    punct_rgba[labeled > 0] = (*color_on, 0.75)
    ax_ov.imshow(punct_rgba, interpolation="nearest")

    ax_ov.set_title(
        f"{channel_label} puncta on dendrites\n"
        f"{n_kept} puncta - "
        f"{result['dendrite_length_um']:.0f} µm - "
        f"{result[density_key]:.3f} p/µm",
        fontsize=panelTitleFontsize, pad=20)
    ax_ov.set_xticks([]); ax_ov.set_yticks([])
    for sp in ax_ov.spines.values():
        sp.set_visible(False)
    ax_ov.legend(handles=[
        Patch(facecolor=(*color_on,   0.75), edgecolor="none", label="Puncta (on mask)"),
        Patch(facecolor=(*color_mask, 0.40), edgecolor="none", label="Dilated dendrite mask"),
    ], loc="upper right", fontsize=10, framealpha=0.7,
       facecolor="black", labelcolor="white")
    add_scale_bar(ax_ov, shape, PIXEL_SIZE_UM, color=SCALEBAR_COLOR)

    fig.savefig(Path(out_dir) / f"{sample}_QC_{channel_label}.png")
    plt.close(fig)
    print(f"  QC figure saved: {sample}_QC_{channel_label}.png")


# =============================================================================
# STATISTICS 
# =============================================================================

def rank_biserial(a, b):
    # Effect size for Mann-Whitney U. Range -1 to +1
    U, _ = spstats.mannwhitneyu(a, b, alternative="two-sided")
    return 1 - (2 * U) / (len(a) * len(b))


def significance_label(p):
    # Significance indicator for plots
    return "****" if p < 0.0001 else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def run_stats(ctrl_vals, stim_vals, experiment, channel_label):
    ctrl_vals = np.asarray(ctrl_vals, dtype=float)
    stim_vals = np.asarray(stim_vals, dtype=float)

    if len(ctrl_vals) < 2 or len(stim_vals) < 2:
        print(f"  [WARN] Not enough data for statistics.")
        return np.nan, np.nan, "n/a"
    # Calculate Mann-Whitney U-test
    U, p_val = spstats.mannwhitneyu(ctrl_vals, stim_vals, alternative="two-sided")
    # Calculate rank biserial effect size
    r        = rank_biserial(ctrl_vals, stim_vals)
    # Interpretation thresholds for rank biserial effect size
    r_interp = ("negligible" if abs(r) < 0.1 else "small"  if abs(r) < 0.3
                else "medium" if abs(r) < 0.5 else "large")
    sig      = significance_label(p_val)

    print(f"\n  {experiment} — {channel_label} statistics")
    print(f"    Mann-Whitney U = {U:.1f},  p = {p_val:.4f}  ({sig})")
    print(f"    Rank-biserial r = {r:.3f}  ({r_interp})")
    print(f"    n_ctrl = {len(ctrl_vals)},  n_stim = {len(stim_vals)}")

    return p_val, r, sig


# =============================================================================
# COMPARISON BAR CHART  (matched to vesicle pool visual parameters)
# =============================================================================

def _significance_bracket(ax, x0, x1, y_bracket, label):
    # Add significance bracket to plot
    step_y = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.10
    ax.plot([x0, x0, x1, x1],
            [y_bracket - step_y * 0.15, y_bracket,
             y_bracket, y_bracket - step_y * 0.15],
            color="black", linewidth=1.0, clip_on=False)
    ax.text((x0 + x1) / 2, y_bracket + step_y * 0.05, label,
            ha="center", va="bottom", fontsize=16, clip_on=False)


def comparison_bar_chart(df_subset, density_col, channel_label,
                         experiment, ctrl_group, stim_group, out_path):
    # Plot bar plot
    ctrl_vals = df_subset[df_subset["group"] == ctrl_group][density_col].values
    stim_vals = df_subset[df_subset["group"] == stim_group][density_col].values

    p_val, d, sig = run_stats(ctrl_vals, stim_vals, experiment, channel_label)

    means = [np.mean(ctrl_vals), np.mean(stim_vals)]
    sems  = [spstats.sem(ctrl_vals), spstats.sem(stim_vals)]
    ns    = [len(ctrl_vals), len(stim_vals)]
    x     = np.arange(2)

    fig, ax = plt.subplots(figsize=(5, 6.5))
    all_vals = np.concatenate([ctrl_vals, stim_vals])
    bar_colors = [COLORS["Control"], COLORS.get(experiment, "#aaaaaa")]

    ax.bar(x, means, width=0.52,
           color=bar_colors, edgecolor="none", linewidth=0,
           zorder=0, alpha=1)
    # Add sample scatter
    rng = np.random.default_rng(seed=42)
    for i, vals in enumerate([ctrl_vals, stim_vals]):
        jit = rng.uniform(-0.2 * 0.52, 0.2 * 0.52, size=len(vals))
        ax.scatter(np.full(len(vals), x[i]) + jit, vals,
                   color="gray", alpha=0.6, s=22, zorder=1, linewidths=0.8)

    ax.bar(x, means, yerr=sems, width=0.52,
           color="none", edgecolor="black", linewidth=0.8,
           capsize=5, error_kw=dict(elinewidth=1.5, ecolor="black", zorder=3),
           zorder=3)

    y_top  = max(m + s for m, s in zip(means, sems))
    y_max = 0.85
    ax.set_ylim(0, y_max * 1.28)

    _significance_bracket(ax, x[0], x[1], y_max * 1.18, sig)

    for i, n in enumerate(ns):
        ax.annotate(f"n={n}", xy=(x[i], 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -32), textcoords="offset points",
                    ha="center", va="top", fontsize=16, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(["Control", experiment], fontsize=18)
    ax.set_ylabel(f"{channel_label} puncta density [puncta/µm]", fontsize=18)
    ax.set_title(f"{channel_label} puncta density per\ndendrite length of Control and {experiment}", fontsize=22, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  Chart saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Setup folder paths
    in_dir   = Path(INPUT_DIR).resolve()
    out_dir  = Path(OUTPUT_DIR).resolve()
    roi_base = Path(ROI_BASE_DIR).resolve()

    if not in_dir.exists():
        print(f"[ERROR] INPUT_DIR does not exist: {in_dir}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Search for related images
    glun1_files = sorted(in_dir.glob(f"*{GLUN1_SUFFIX}"))
    if not glun1_files:
        print(f"[ERROR] No files matching '*{GLUN1_SUFFIX}' in {in_dir}")
        sys.exit(1)

    triplets, missing = [], []
    for gf in glun1_files:
        stem = gf.name.replace(GLUN1_SUFFIX, "")
        af   = gf.with_name(stem + GLUA2_SUFFIX)
        mf   = gf.with_name(stem + MAP2_SUFFIX)
        if af.exists() and mf.exists():
            triplets.append((gf, af, mf))
        else:
            missing.append(gf.name)

    if missing:
        print(f"[WARNING] Incomplete triplets skipped: {', '.join(missing)}")
    print(f"Found {len(triplets)} triplet(s) in {in_dir}\n{'─' * 55}")

    # Process each sample 
    rows = []
    for i, (gf, af, mf) in enumerate(triplets, 1):
        # Extract the sample information
        stem                       = gf.name.replace(GLUN1_SUFFIX, "")
        experiment, status, group  = parse_sample(stem)
        print(f"[{i}/{len(triplets)}] {stem}  ({experiment}) ...", flush=True)
        # Load the puncta ROIs
        roi_zip = roi_base / f"RoiSet_{stem}.zip"
        if not roi_zip.is_file():
            print(f"  [WARN] ROI zip not found: {roi_zip}  — skipping sample")
            continue

        try:
            # Pre-process MAP2 images and segment the dendrites
            map2_raw        = load(mf)
            map2_n          = normalize(map2_raw)
            dend_mask, skel = segment_dendrites(map2_n, experiment)
            dend_length_px  = int(skel.sum()) # Calculate dendrite length
            dend_length_um  = dend_length_px * PIXEL_SIZE_UM
            image_shape     = map2_raw.shape
            # Dilate dendrite mask
            dend_dilated = morphology.dilation(dend_mask, morphology.disk(DENDRITE_DILATION))

            result = dict(
                sample             = stem,
                experiment         = experiment,
                status             = status,
                group              = group,
                dendrite_length_px = dend_length_px,
                dendrite_length_um = round(dend_length_um, 2),
                _map2_raw          = map2_raw,
                _dend_mask         = dend_mask,
                _skeleton          = skel,
            )

            all_rois = load_rois_from_zip(roi_zip)
            # Filter for puncta on dendrites
            kept_rois, n_total, n_kept, n_rejected = filter_rois_by_mask(
                all_rois, dend_dilated, image_shape)

            labeled_all  = rois_to_labeled_image(all_rois,  image_shape)
            labeled_kept = rois_to_labeled_image(kept_rois, image_shape)

            print(f"  ROIs loaded: {n_total} total → {n_kept} kept, "
                  f"{n_rejected} rejected (centroid off dilated mask)")
            # Compute puncta density per dendrite length
            density = n_kept / dend_length_um if dend_length_um > 0 else 0.0

            if experiment == "cTBS":
                ch_key  = "glun1"
                ch_raw  = load(gf)
            else:
                ch_key  = "glua2"
                ch_raw  = load(af)

            channel_label = "GluN1" if ch_key == "glun1" else "GluA2"

            result.update({
                f"{ch_key}_n_puncta":       n_kept,
                f"{ch_key}_density_per_um": round(density, 5),
                f"_{ch_key}_raw":           ch_raw,
                f"_{ch_key}_n":             normalize(ch_raw),
                f"_labeled_{ch_key}":       labeled_kept,
                f"_labeled_all_{ch_key}":   labeled_all,
                f"_{ch_key}_n_total":       n_total,
                f"_{ch_key}_n_rejected":    n_rejected,
            })

            print(f"  {channel_label}: {n_kept} puncta / "
                  f"{dend_length_um:.0f} µm = {density:.4f} p/µm")

            if SAVE_QC_IMAGES:
                save_qc_figure(result, ch_key, out_dir)

            rows.append({k: v for k, v in result.items() if not k.startswith("_")})

        except Exception as e:
            print(f"  FAILED — {e}")
            import traceback; traceback.print_exc()

    if not rows:
        print("[ERROR] No results produced.")
        sys.exit(1)

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "receptor_density_results.csv", index=False)

    show_cols = [c for c in ["sample", "experiment", "status", "dendrite_length_um",
                              "glun1_n_puncta", "glun1_density_per_um",
                              "glua2_n_puncta",  "glua2_density_per_um"]
                 if c in df.columns]
    print(f"\n{'─' * 55}")
    print(df[show_cols].to_string(index=False))
    print(f"\nCSV saved: {out_dir / 'receptor_density_results.csv'}")

    # IQR Outlier Detection 
    df_ctbs = df[df["experiment"] == "cTBS"].copy()
    df_itbs = df[df["experiment"] == "iTBS"].copy()

    if not df_ctbs.empty and "glun1_density_per_um" in df_ctbs.columns:
        df_ctbs = detect_iqr_outliers(df_ctbs, "glun1_density_per_um")
        df_ctbs = print_outlier_report(df_ctbs, "glun1_density_per_um", "GluN1")

    if not df_itbs.empty and "glua2_density_per_um" in df_itbs.columns:
        df_itbs = detect_iqr_outliers(df_itbs, "glua2_density_per_um")
        df_itbs = print_outlier_report(df_itbs, "glua2_density_per_um", "GluA2")

    df_all_flagged = pd.concat([df_ctbs, df_itbs], ignore_index=True)
    save_cols = [c for c in df_all_flagged.columns if not c.startswith("_")]
    df_all_flagged[save_cols].to_csv(
        out_dir / "receptor_density_results_outliers.csv", index=False)
    print(f"Outlier-flagged CSV saved: "
          f"{out_dir / 'receptor_density_results_outliers.csv'}")

    # Comparison bar plot
    if not df_ctbs.empty and "glun1_density_per_um" in df_ctbs.columns:
        comparison_bar_chart(df_ctbs, "glun1_density_per_um", "GluN1", "cTBS",
                             "cTBS_Control", "cTBS_cTBS",
                             out_dir / "cTBS_GluN1_receptor_density.png")

    if not df_itbs.empty and "glua2_density_per_um" in df_itbs.columns:
        comparison_bar_chart(df_itbs, "glua2_density_per_um", "GluA2", "iTBS",
                             "iTBS_Control", "iTBS_iTBS",
                             out_dir / "iTBS_GluA2_receptor_density.png")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
