# The analysis of the vesicle pool sizes is based on the code from:
# VesiclePoolSizes by Jana Wrosch (now Dahlmanns)
# https://github.com/janawrosch/VesiclePoolSizes.git
# Original code has been modified for this project.
#
# This code was used to analyse the vesicle pool size measurements that were performed according to the protocol of Dahlmanns and Dahlmanns [1].
# [1]   M. Dahlmanns and J. K. Dahlmanns, “Synaptic vesicle pool monitoring
#       with synapto-phluorin,” in Synaptic Vesicles: Methods and Protocols, 
#       J. Dahlmanns and M. Dahlmanns, Eds. New York, NY: Springer US, 2022,
#       pp. 181–192. doi: 10.1007/978-1-0716-1916-2_14.
#
# [2]   H. Jia, N. L. Rochefort, X. Chen, and A. Konnerth, “In vivo two-photon
#       imaging of sensory-evoked dendritic calcium signals in cortical neurons,”
#       Nature Protocols, vol. 6, no. 1, pp. 28–35, Jan. 2011. doi: 10.1038/nprot.2010.169.
#       
#
# [3]   I. F. Sbalzarini and P. Koumoutsakos, “Feature point tracking and trajectory analysis for video imaging in cell biology,” Journal of Structural
#       Biology, vol. 151, no. 2, pp. 182–195, Aug. 2005. doi: 10.1016/j.jsb.005.06.002.
#
#
# [4]   P. Filzmoser and K. Hron, “Outlier Detection for Compositional Data Using
#       Robust Methods,” Mathematical Geosciences, vol. 40, pp. 233–248, Apr.
#       2008. doi: 10.1007/s11004-007-9141-5.
#
#
# ChatGPT (OpenAI) and Claude (Antrophic) were partly used as assistance for code generation which was reviewed and verified by the creator.

#import libraries
import os
import tifffile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from peakdetect_sbalzarini import peakdetect_sbalzarini
import seaborn as sns
from scipy.ndimage import uniform_filter, uniform_filter1d
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from relative_fluorescence import relative_fluorescence
from scipy import stats
from sklearn.covariance import MinCovDet
from itertools import combinations
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.multivariate.manova import MANOVA


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
})

# =============================================================================
# 1. PARAMETERS
# =============================================================================
# --- Run mode ---
RUN_SUMMARY_ONLY = True   # True = skip processing, load CSV and jump to plots
# List of samples according to their condition
conditions = {
    "Control" : {"prefix": "iTBS",     "samples": ["C9", "C11", "C13", "C14", "C15", "C16","C17", "C18", "C19", "C20", "C23", "C24", "C25", "C27", "C28", "C29", "C30","C31"]},
    "cTBS"    : {"prefix": "cTBS", "samples": ["S1", "S2", "S5", "S6", "S8", "S10", "S13", "S14"]},
    "iTBS"    : {"prefix": "iTBS", "samples": ["S1", "S2", "S5", "S7", "S8", "S10", "S11", "S12", "S14", "S15", "S16"]},
}

batch_mode            = True    # Use batchprocessing to process samples parallel
plot_final_trace      = False   # toggle if trace plot is shown once it is done
plot_diagnostic_plots = False   # toggle plots are shown for diagnostics
# base folder with folders for each samples full recording in form of .tif files
base_folder= r"base_folder"

fig_output_folder = os.path.join(base_folder, "figures") # output folder
os.makedirs(fig_output_folder, exist_ok=True)
# Parameter for sbalzarini peak detection
w   = 10    # radius around peak candidates
pth = 1     # upper percentile of the intensity histogram 

# Minimum good ROIs required 
min_good_rois = 2

# Pool measurement window width
mean_window = 20

# Rolling-baseline half-window for sigma(DF) image (frames)
ROLLING_BASELINE_HALFWIN = 50

# Spatial downsample factor for ROI detection to reduce processing power
ROI_DETECT_DOWNSAMPLE = 2

# Number of parallel worker processes (None = use all CPU cores)
N_WORKERS = 4

# Framerate 
RECORDING_HZ = 4

# Plot start frame - cut of algorithm induced artifact at the beginning
PLOT_START_FRAME = 72

# =============================================================================
# Color palette for plots
# =============================================================================
pools = ["RRP", "RECP", "RESP"]
pool_xlabels = ["Ready Releasable Pool", "Recycling Pool", "Reserve Pool"]

_set3 = sns.color_palette("Set3", 12)
condition_base_colors = {
    "Control": _set3[4],
    "cTBS":    _set3[5],
    "iTBS":    _set3[6],
}

condition_shade_ramps = {
    cond: sns.dark_palette(base_color, n_colors=len(pools) + 2, reverse=True)[1:len(pools) + 1]
    for cond, base_color in condition_base_colors.items()
}

rrp_color, rec_color, total_color = condition_shade_ramps["Control"]
# Filename for results
csv_file = base_folder + "\\vesicle_pool_results.csv"


# =============================================================================
# HELPERS
# =============================================================================

def compute_delta_f_std_fast(img_stack, halfwin=ROLLING_BASELINE_HALFWIN):
    # compute the standard deviation of of each pixel over time
    baseline = uniform_filter1d(img_stack, size=2 * halfwin + 1, axis=0, mode="nearest") # Baseline fluorescence from moving average 
    diff   = img_stack - baseline # Difference between intensity value and baseline
    result = diff.std(axis=0) # Standard deviation of the intensity over time
    del diff, baseline
    return result


def _compute_pools(trace_1d, mean_window,
                   baseline_start, rrpstim_start, vorrecstim_start,
                   recstim_start, vortotalstim_start,
                   total_plateau_start, total_plateau_stop):
    # Pool size calculation translated from https://github.com/janawrosch/VesiclePoolSizes.git to pyhton
    sl = slice
    # Calculation of the mean of each window before and after the stimulations
    bl_mean           = trace_1d[sl(baseline_start,     baseline_start     + mean_window)].mean()   
    rrp_mean          = trace_1d[sl(rrpstim_start,       rrpstim_start      + mean_window)].mean()
    vorrecstim_mean   = trace_1d[sl(vorrecstim_start,    vorrecstim_start   + mean_window)].mean()
    rec_mean          = trace_1d[sl(recstim_start,       recstim_start      + mean_window)].mean()
    vortotalstim_mean = trace_1d[sl(vortotalstim_start,  vortotalstim_start + mean_window)].mean()
    total_mean        = trace_1d[sl(total_plateau_start, total_plateau_stop)].mean()
    # Calculation of the fluorescence increase
    delta_rrp   = rrp_mean   - bl_mean
    delta_rec   = rec_mean   - vorrecstim_mean
    delta_total = total_mean - vortotalstim_mean
    # Normalization to the total vesicle pool size
    denom    = delta_rrp + delta_rec + delta_total
    safe_div = lambda n: (n / denom * 100) if denom != 0 else np.nan
    rrp_frac, rec_frac, total_frac = safe_div(delta_rrp), safe_div(delta_rec), safe_div(delta_total)

    safe_sc = lambda v: ((v - bl_mean) / denom * 100) if denom != 0 else np.nan

    return dict(
        bl_mean=bl_mean, rrp_mean=rrp_mean, rec_mean=rec_mean, total_mean=total_mean,
        vorrecstim_mean=vorrecstim_mean, vortotalstim_mean=vortotalstim_mean,
        delta_rrp=delta_rrp, delta_rec=delta_rec, delta_total=delta_total,
        rrp_frac=rrp_frac, rec_frac=rec_frac, total_frac=total_frac,
        mean_scaled_rrp=safe_sc(rrp_mean),
        mean_scaled_rec=safe_sc(rec_mean),
        mean_scaled_total=safe_sc(total_mean),
        mean_scaled_vorrecstim=safe_sc(vorrecstim_mean),
        mean_scaled_vortotalstim=safe_sc(vortotalstim_mean),
    )



# =============================================================================
# FIGURE HELPERS
# =============================================================================

def _get_condition_colors(condition):
    # helper function for color assignment
    ramp = condition_shade_ramps.get(condition, condition_shade_ramps["Control"])
    return ramp[0], ramp[1], ramp[2]


def _make_pool_legend_handles(rrp_c, rec_c, tc):
    # helper function for plot legend
    return [
        Patch(facecolor=rrp_c, alpha=0.18, edgecolor="none", label="RRP"),
        Patch(facecolor=rrp_c, alpha=0.20, edgecolor="none", label="pre RRP"),
        Patch(facecolor=rrp_c, alpha=0.45, edgecolor="none", label="post RRP"),
        Patch(facecolor=rec_c, alpha=0.18, edgecolor="none", label="RECP"),
        Patch(facecolor=rec_c, alpha=0.20, edgecolor="none", label="pre RECP"),
        Patch(facecolor=rec_c, alpha=0.45, edgecolor="none", label="post RECP"),
        Patch(facecolor=tc,    alpha=0.18, edgecolor="none", label="RESP"),
        Patch(facecolor=tc,    alpha=0.20, edgecolor="none", label="pre RESP"),
        Patch(facecolor=tc,    alpha=0.45, edgecolor="none", label="post RESP"),
    ]


def _add_pool_decorations_scaled(ax, pool_vals, windows, n_frames, condition=None):
    # Add color decorations for better visualization of the pool sizes in the scaled relative fluorescence trace plot
    rrp_c, rec_c, tc = _get_condition_colors(condition)

    y_bl       = 0.0
    y_rrp      = pool_vals["mean_scaled_rrp"]
    y_vorrec   = pool_vals["mean_scaled_vorrecstim"]
    y_rec      = pool_vals["mean_scaled_rec"]
    y_vortotal = pool_vals["mean_scaled_vortotalstim"]
    y_total    = pool_vals["mean_scaled_total"]

    ax.axhspan(y_bl,       y_rrp,   color=rrp_c, alpha=0.18, zorder=0)
    ax.axhspan(y_vorrec,   y_rec,   color=rec_c, alpha=0.18, zorder=0)
    ax.axhspan(y_vortotal, y_total, color=tc,    alpha=0.18, zorder=0)

    window_y_limits = {
        "baseline"     : (y_bl,      y_rrp),
        "rrp_stim"     : (y_bl,      y_rrp),
        "rrp_plateau"  : (y_bl,      y_rrp),
        "vorrecstim"   : (y_vorrec,  y_rec),
        "rec_stim"     : (y_vorrec,  y_rec),
        "rec_plateau"  : (y_vorrec,  y_rec),
        "vortotalstim" : (y_vortotal, y_total),
        "total_plateau": (y_vortotal, y_total),
    }
    window_colors = {
        "baseline": rrp_c, "rrp_stim": rrp_c, "rrp_plateau": rrp_c,
        "vorrecstim": rec_c, "rec_stim": rec_c, "rec_plateau": rec_c,
        "vortotalstim": tc, "total_plateau": tc,
    }
    window_alphas = {
        "baseline": 0.20, "rrp_stim": 0.20, "rrp_plateau": 0.45,
        "vorrecstim": 0.20, "rec_stim": 0.20, "rec_plateau": 0.45,
        "vortotalstim": 0.20, "total_plateau": 0.45,
    }

    ylim    = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    for name, (s, e, _c, _a) in windows.items():
        y_lo, y_hi = window_y_limits[name]
        y_lo_vis = max(y_lo, ylim[0])
        y_hi_vis = min(y_hi, ylim[1])
        if y_hi_vis <= y_lo_vis:
            continue
        ymin_f = (y_lo_vis - ylim[0]) / y_range
        ymax_f = (y_hi_vis - ylim[0]) / y_range
        ax.axvspan(s / RECORDING_HZ, e / RECORDING_HZ,
                   ymin=ymin_f, ymax=ymax_f,
                   color=window_colors[name], alpha=window_alphas[name], zorder=1)


def _add_pool_decorations_raw(ax, pv, windows, n_frames, condition=None):
    # Add color decorations for better visualization of the pool sizes in the raw relative fluorescence trace plot
    rrp_c, rec_c, tc = _get_condition_colors(condition)

    y_bl       = pv["bl_mean"]
    y_rrp      = pv["rrp_mean"]
    y_vorrec   = pv["vorrecstim_mean"]
    y_rec      = pv["rec_mean"]
    y_vortotal = pv["vortotalstim_mean"]
    y_total    = pv["total_mean"]

    ax.axhspan(y_bl,       y_rrp,   color=rrp_c, alpha=0.18, zorder=0)
    ax.axhspan(y_vorrec,   y_rec,   color=rec_c, alpha=0.18, zorder=0)
    ax.axhspan(y_vortotal, y_total, color=tc,    alpha=0.18, zorder=0)

    window_y_limits = {
        "baseline"     : (y_bl,      y_rrp),
        "rrp_stim"     : (y_bl,      y_rrp),
        "rrp_plateau"  : (y_bl,      y_rrp),
        "vorrecstim"   : (y_vorrec,  y_rec),
        "rec_stim"     : (y_vorrec,  y_rec),
        "rec_plateau"  : (y_vorrec,  y_rec),
        "vortotalstim" : (y_vortotal, y_total),
        "total_plateau": (y_vortotal, y_total),
    }
    window_colors = {
        "baseline": rrp_c, "rrp_stim": rrp_c, "rrp_plateau": rrp_c,
        "vorrecstim": rec_c, "rec_stim": rec_c, "rec_plateau": rec_c,
        "vortotalstim": tc, "total_plateau": tc,
    }
    window_alphas = {
        "baseline": 0.20, "rrp_stim": 0.20, "rrp_plateau": 0.45,
        "vorrecstim": 0.20, "rec_stim": 0.20, "rec_plateau": 0.45,
        "vortotalstim": 0.20, "total_plateau": 0.45,
    }

    ylim    = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    for name, (s, e, _c, _a) in windows.items():
        y_lo, y_hi = window_y_limits[name]
        y_lo_vis = max(y_lo, ylim[0])
        y_hi_vis = min(y_hi, ylim[1])
        if y_hi_vis <= y_lo_vis:
            continue
        ymin_f = (y_lo_vis - ylim[0]) / y_range
        ymax_f = (y_hi_vis - ylim[0]) / y_range
        ax.axvspan(s / RECORDING_HZ, e / RECORDING_HZ,
                   ymin=ymin_f, ymax=ymax_f,
                   color=window_colors[name], alpha=window_alphas[name], zorder=1)


def _save_scaled_trace_plot(mean_scaled, n_frames, pool_vals, windows,
                             sample_name, folder_name, fig_output_folder,
                             condition=None, show=False):
    # Plot the scaled relative fluorescence trace and save to png 
    time_axis = np.arange(n_frames) / RECORDING_HZ

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.plot(time_axis, mean_scaled, color="black", linewidth=2)

    visible  = mean_scaled[PLOT_START_FRAME:]
    y_margin = 1
    ax.set_ylim(min(visible.min(), 0) - y_margin,
            visible.max() + y_margin)

    ax.set_xlim(PLOT_START_FRAME / RECORDING_HZ, n_frames / RECORDING_HZ)

    _add_pool_decorations_scaled(ax, pool_vals, windows, n_frames, condition=condition)

    rrp_c, rec_c, tc = _get_condition_colors(condition)
    ax.legend(handles=_make_pool_legend_handles(rrp_c, rec_c, tc),
              fontsize=14, loc="upper left", framealpha=0.8)

    ax.set_xlabel("Time [s]", fontsize=18)
    ax.set_ylabel("Cumulative relative fluorescence (%)", fontsize=18)
    ax.set_title(f"Scaled cumulative fluorescence trace", fontsize=25, pad=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize = 18)
    plt.tight_layout()
    path = os.path.join(fig_output_folder, f"{folder_name}_scaled_trace.png")
    fig.savefig(path)
    print(f"  Saved: {path}")
    if show: plt.show()
    plt.close(fig)


# =============================================================================
# CORE PROCESSING FUNCTION
# =============================================================================

def process_sample(sample_name, folder_prefix, condition_name=None,
                   show_diagnostic_plots=False, show_final_trace=False):

    print(f"\n{'='*60}")
    print(f"  Processing {sample_name}")
    print(f"{'='*60}")
    # load sample
    folder_name = sample_name if not folder_prefix else f"{folder_prefix}_{sample_name}"
    folder      = os.path.join(base_folder, folder_name)

    files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".tif")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    img_stack = np.array(
        [tifffile.imread(os.path.join(folder, f)) for f in files],
        dtype=np.float32
    )
    n_frames = img_stack.shape[0]
    print(f"  Loaded {n_frames} frames  ({img_stack.nbytes / 1e6:.0f} MB, float32)")

    # =========================================================================
    # ROI DETECTION
    # =========================================================================
    print("  Computing sigma(DF) image ...")
    sigma_img  = compute_delta_f_std_fast(img_stack, halfwin=ROLLING_BASELINE_HALFWIN) # compute the standard deviation of fluorescence for each pixel over time
    sigma_norm = (sigma_img - sigma_img.min()) / (sigma_img.max() - sigma_img.min() + 1e-12) # Min-Max Normalization 
    # Reduce image size to reduce processing power for peak detection
    ds = ROI_DETECT_DOWNSAMPLE
    if ds > 1:
        H, W       = sigma_norm.shape
        H_ds, W_ds = H // ds, W // ds
        sigma_ds   = sigma_norm[:H_ds*ds, :W_ds*ds].reshape(H_ds, ds, W_ds, ds).mean(axis=(1, 3))
        w_detect   = max(1, w // ds)
    else:
        sigma_ds, w_detect = sigma_norm, w
    # Use feature point detection to extract ROIs [3]
    regions_ds = peakdetect_sbalzarini(sigma_ds, w=w_detect, pth=pth, show_img=False) 
    # Transform ROIs back to original image
    if ds > 1:
        for r in regions_ds:
            cx, cy = r["Centroid"]
            r["Centroid"] = (cx * ds, cy * ds)
            if isinstance(r["PixelIdxList"], tuple) and len(r["PixelIdxList"]) == 2:
                rows_ds, cols_ds = r["PixelIdxList"]
                r["PixelIdxList"] = (rows_ds * ds, cols_ds * ds)
            else:
                H_ds_s, W_ds_s = sigma_ds.shape
                rows_ds, cols_ds = np.unravel_index(r["PixelIdxList"],
                                                    (H_ds_s, W_ds_s), order='F')
                r["PixelIdxList"] = (rows_ds * ds, cols_ds * ds)

    regions = regions_ds
    n_rois  = len(regions)
    print(f"  Detected {n_rois} ROIs  (downsample={ds}x)")
    # Centroid coordinates of all detected ROIs
    all_C = [r["Centroid"][0] for r in regions]
    all_R = [r["Centroid"][1] for r in regions]

    # Plot fluctuation map of the standard deviation image with the detected ROIs
    fig_sigma, ax_sigma = plt.subplots(figsize=(7, 6))
    ax_sigma.imshow(uniform_filter(sigma_norm, size=5), cmap="viridis")
    ax_sigma.scatter(all_C, all_R, alpha=0.35, c="orange", s=10,
                     label=f"{n_rois} detected ROIs")
    ax_sigma.set_title(f"Standard deviation of fluorescence intensity", fontsize=18)
    ax_sigma.set_xticks([])
    ax_sigma.set_yticks([])
    ax_sigma.legend(fontsize=12, framealpha=0.8, loc='upper left')
    ax_sigma.axis("image")
    PIXEL_SIZE_UM = 0.064
    SCALEBAR_UM   = 10
    scalebar_px   = SCALEBAR_UM / PIXEL_SIZE_UM
    img_h, img_w  = sigma_norm.shape
    margin_x      = img_w  * 0.05
    margin_y      = img_h  * 0.05
    bar_thickness = max(2, img_h * 0.012)
    bar_x_end   = img_w  - margin_x
    bar_x_start = bar_x_end - scalebar_px
    bar_y       = img_h  - margin_y
    ax_sigma.add_patch(plt.Rectangle(
        (bar_x_start, bar_y - bar_thickness / 2),
        scalebar_px, bar_thickness,
        color="white", zorder=10
    ))
    ax_sigma.text(
        (bar_x_start + bar_x_end) / 2, bar_y - bar_thickness * 1.8,
        f"{SCALEBAR_UM} µm",
        color="white", fontsize=11, ha="center", va="bottom",
        fontweight="bold", zorder=11
    )
    plt.tight_layout()
    # Save fluctuation map of the standard deviation image with the detected ROIs
    path_sigma = os.path.join(fig_output_folder, f"{folder_name}_roi_sigma.png")
    fig_sigma.savefig(path_sigma, dpi=300)
    print(f"  Saved: {path_sigma}")
    if show_diagnostic_plots: plt.show()
    plt.close(fig_sigma)

    # -------------------------------------------------------------------------
    # Extract per-ROI fluorescence traces
    # -------------------------------------------------------------------------
    h, w_img  = img_stack.shape[1], img_stack.shape[2]
    meanstack = np.zeros((n_rois, n_frames), dtype=np.float32)

    for i, region in enumerate(regions):
        pixels = region["PixelIdxList"]
        if isinstance(pixels, tuple) and len(pixels) == 2:
            rows, cols = pixels
        else:
            rows, cols = np.unravel_index(pixels, (h, w_img), order='F')
        # Filter ROIs that exceed the image borders
        rows = np.clip(np.asarray(rows), 0, h - 1)
        cols = np.clip(np.asarray(cols), 0, w_img - 1)
        meanstack[i] = img_stack[:, rows, cols].mean(axis=1)
    del img_stack

    # Calculate the relative fluorescence trace for all ROIs [2]
    dff_raw = relative_fluorescence(meanstack)
    # Use the np.maximum.accumulate() function to only extract increases in fluorescence and ignore decreases 
    dff     = np.array([np.maximum.accumulate(dff_raw[i]) for i in range(n_rois)], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Set windows before and after the man stimulation increase
    # -------------------------------------------------------------------------
    baseline_start      = 80
    baseline_stop       = baseline_start     + mean_window
    rrpstim_start       = 115
    rrpstim_stop        = rrpstim_start      + mean_window
    vorrecstim_start    = 200
    vorrecstim_stop     = vorrecstim_start   + mean_window
    recstim_start       = 275
    recstim_stop        = recstim_start      + mean_window
    vortotalstim_start  = 370
    vortotalstim_stop   = vortotalstim_start + mean_window

    baseline_plateau_start     = baseline_start
    rrp_plateau_start          = rrpstim_start
    vorrecstim_plateau_start   = vorrecstim_start
    rec_plateau_start          = recstim_start
    vortotalstim_plateau_start = vortotalstim_start
    # Automatically detect the peak of the NH4Cl induced fluorescence increase
    total_search_start  = 400
    dff_raw_mean        = dff.mean(axis=0)
    peak_offset         = np.argmax(dff_raw_mean[total_search_start:])
    peak_frame          = total_search_start + peak_offset
    quarter_win         = mean_window // 4
    # Set window slightly shifted to the right to exclude the strong increasing section before the peak
    total_plateau_start = max(total_search_start, peak_frame - quarter_win)
    if total_plateau_start + mean_window > 520: # Handle peak at the end of the recording
        total_plateau_start = 520
    total_plateau_stop  = min(total_plateau_start + mean_window, n_frames - 1)
    totalstim_start     = total_plateau_start
    totalstim_stop      = total_plateau_stop

    print(f"  Total pool peak at frame {peak_frame}  "
          f"(raw mean dF/F = {dff_raw_mean[peak_frame]:.4f})")
    print(f"  Total plateau window: {total_plateau_start} - {total_plateau_stop}")
    # Assign windows to color and position of the plots
    plot_windows = {
        "baseline"     : (baseline_plateau_start,     baseline_plateau_start + mean_window,     "gray",      0.15),
        "rrp_stim"     : (rrpstim_start,              rrpstim_stop,                             rrp_color,   0.15),
        "rrp_plateau"  : (rrp_plateau_start,          rrp_plateau_start + mean_window,          rrp_color,   0.35),
        "vorrecstim"   : (vorrecstim_plateau_start,   vorrecstim_plateau_start + mean_window,   "gray",      0.15),
        "rec_stim"     : (recstim_start,              recstim_stop,                             rec_color,   0.15),
        "rec_plateau"  : (rec_plateau_start,          rec_plateau_start + mean_window,          rec_color,   0.35),
        "vortotalstim" : (vortotalstim_plateau_start, vortotalstim_plateau_start + mean_window, "gray",      0.15),
        "total_plateau": (total_plateau_start,        total_plateau_stop,                       total_color, 0.35),
    }
    # Plot the detection of the NH4Cl-induce fluorescence peak 
    if show_diagnostic_plots:
        fig_diag, ax_diag = plt.subplots(figsize=(10, 4))
        ax_diag.plot(dff_raw_mean, color="black", linewidth=1.2)
        ax_diag.axvspan(total_search_start, n_frames - 1, alpha=0.08, color=total_color)
        ax_diag.axvline(x=peak_frame, color=total_color, linewidth=1.5, linestyle="--",
                        label=f"Detected peak (frame {peak_frame})")
        ax_diag.axvspan(total_plateau_start, total_plateau_stop, alpha=0.35, color=total_color)
        ax_diag.set_xlabel("Frame"); ax_diag.set_ylabel("dF/F")
        ax_diag.set_title(f"{sample_name}  -  Total plateau auto-detection")
        ax_diag.legend(fontsize=11, loc="upper left")
        plt.tight_layout(); plt.show()

    # =========================================================================
    # Filter out ROIs that only had fluorescence decrease due to systematic errors
    # =========================================================================

    bl_vals           = dff[:, baseline_plateau_start:     baseline_plateau_start + mean_window].mean(axis=1)
    rrp_vals          = dff[:, rrp_plateau_start:          rrp_plateau_start      + mean_window].mean(axis=1)
    vorrecstim_vals   = dff[:, vorrecstim_plateau_start:   vorrecstim_plateau_start + mean_window].mean(axis=1)
    rec_vals          = dff[:, rec_plateau_start:          rec_plateau_start      + mean_window].mean(axis=1)
    vortotalstim_vals = dff[:, vortotalstim_plateau_start: vortotalstim_plateau_start + mean_window].mean(axis=1)
    total_vals        = dff[:, total_plateau_start:        total_plateau_stop].mean(axis=1)

    delta_rrp_v   = rrp_vals   - bl_vals
    delta_rec_v   = rec_vals   - vorrecstim_vals
    delta_total_v = total_vals - vortotalstim_vals
    # If all three pools are negative or zero the ROI is filtered out 
    bad_pool      = (delta_rrp_v <= 0.0) & (delta_rec_v <= 0.0) & (delta_total_v <= 0.0)
    good_roi_mask = ~bad_pool
    selected_rois = np.where(good_roi_mask)[0]

    print(f"  Pool filter  : {good_roi_mask.sum()} / {n_rois} pass")
    # samples with less than 2 good ROIs are excluded 
    if len(selected_rois) <= min_good_rois:
        print(f"  WARNING: Only {len(selected_rois)} good ROI(s) for {sample_name} "
              f"(threshold: >{min_good_rois}) - skipping.")
        return None
    # Plot individual ROI traces
    if show_diagnostic_plots:
        n_cols = 5
        n_rows = int(np.ceil(n_rois / n_cols))
        fig_grid, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows), sharex=True)
        axes = axes.flatten()
        for i in range(n_rois):
            ax_roi      = axes[i]
            trace_color = "black" if good_roi_mask[i] else "red"
            ax_roi.plot(dff[i], color=trace_color, linewidth=0.8)
            for _, (s, e, c, a) in plot_windows.items():
                ax_roi.axvspan(s, e, alpha=a, color=c)
            status = "+" if good_roi_mask[i] else "x"
            ax_roi.set_title(f"ROI {i+1} {status}", fontsize=7, color=trace_color, pad=2)
            ax_roi.relim(); ax_roi.autoscale_view()
        for j in range(n_rois, len(axes)):
            fig_grid.delaxes(axes[j])
        fig_grid.suptitle(f"{sample_name}  |  All ROIs  |  "
                          f"{good_roi_mask.sum()} good / {n_rois} total", fontsize=14)
        plt.tight_layout(); plt.show()

    # =========================================================================
    # Pool size calculation with the mean trace of all ROIs
    # =========================================================================
    mean_trace = dff[selected_rois].mean(axis=0)

    pool_args = (mean_window,
                 baseline_start, rrpstim_start, vorrecstim_start,
                 recstim_start, vortotalstim_start,
                 total_plateau_start, total_plateau_stop)

    pv = _compute_pools(mean_trace, *pool_args)

    denom_val   = pv["delta_rrp"] + pv["delta_rec"] + pv["delta_total"]
    # Set Baseline to 0 and scale the trace to the total vesicle pool size
    mean_scaled = ((mean_trace - pv["bl_mean"]) / denom_val * 100
                   if denom_val != 0 else np.full_like(mean_trace, np.nan))

    # =========================================================================
    # FIGURES
    # =========================================================================
    time_axis = np.arange(n_frames) / RECORDING_HZ # Transform framerate to seconds
    rrp_c, rec_c, tc = _get_condition_colors(condition_name) # set colors for the vesicle pool sizes

    # Plot raw mean trace for each sample
    fig_t, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    ax1.plot(time_axis, mean_trace, color="black", linewidth=2)

    visible_trace = mean_trace[PLOT_START_FRAME:] # Cut of the first frames because of an algorithm introduced artifact of the relative_fluorescence() function in combination with the np.maximum.accumulate() function 
    y_pad = (visible_trace.max() - visible_trace.min()) * 0.08
    ax1.set_ylim(visible_trace.min() - y_pad, visible_trace.max() + y_pad)
    ax1.set_xlim(PLOT_START_FRAME / RECORDING_HZ, n_frames / RECORDING_HZ)

    _add_pool_decorations_raw(ax1, pv, plot_windows, n_frames, condition=condition_name)

    ax1.legend(handles=_make_pool_legend_handles(rrp_c, rec_c, tc),
               fontsize=10, loc="upper left", framealpha=0.8)

    ax1.set_ylabel("Relative fluorescence ΔF/F", fontsize=17)
    ax1.set_xlabel("Time [s]", fontsize=17)
    ax1.set_title(f"{condition_name}_{sample_name}  -  Mean trace  ({len(selected_rois)} good ROIs)", fontsize=20, pad= 20)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(fig_output_folder, f"{folder_name}_ROI_mean_trace.png")
    fig_t.savefig(path); print(f"  Saved: {path}")
    if show_diagnostic_plots: plt.show()
    plt.close(fig_t)

    # Plot scaled mean trace for each sample
    _save_scaled_trace_plot(mean_scaled, n_frames, pv, plot_windows,
                             sample_name, folder_name, fig_output_folder,
                             condition=condition_name, show=show_final_trace)

    print(f"\n  --- Pool results for {sample_name} ---")
    print(f"  DRRP   : {pv['delta_rrp']:.4f}  |  scaled: {pv['mean_scaled_rrp']:.2f}%  |  fraction: {pv['rrp_frac']:.2f}%")
    print(f"  DRec   : {pv['delta_rec']:.4f}  |  scaled: {pv['mean_scaled_rec']:.2f}%  |  fraction: {pv['rec_frac']:.2f}%")
    print(f"  DTotal : {pv['delta_total']:.4f}  |  scaled: {pv['mean_scaled_total']:.2f}%")

    return {
        "Sample"                  : sample_name,
        "Total ROIs detected"     : n_rois,
        "Good ROIs used"          : int(good_roi_mask.sum()),
        "Total plateau peak frame": int(peak_frame),
        "RRP delta (dF/F)"        : pv["delta_rrp"],
        "Rec delta (dF/F)"        : pv["delta_rec"],
        "Total delta (dF/F)"      : pv["delta_total"],
        "RRP scaled (%)"          : pv["mean_scaled_rrp"],
        "Rec scaled (%)"          : pv["mean_scaled_rec"],
        "Total scaled (%)"        : pv["mean_scaled_total"],
        "RRP fraction (%)"        : pv["rrp_frac"],
        "Rec fraction (%)"        : pv["rec_frac"],
        "Total fraction (%)"      : pv["total_frac"],
    }


# =============================================================================
# WORKER
# =============================================================================
def _worker(args):
    # Process several samples parallel
    sample, prefix, condition = args
    try:
        out = process_sample(sample, prefix, condition_name=condition,
                             show_diagnostic_plots=False,
                             show_final_trace=False)
        if out is None:
            return None
        out["Condition"] = condition
        return condition, out
    except Exception as exc:
        print(f"  ERROR processing {sample} ({condition}): {exc}")
        return None


# =============================================================================
# 4. RUN
# =============================================================================
all_condition_results = {c: [] for c in conditions}
# Run only summary plots on previously saved vesicle pool size results
if RUN_SUMMARY_ONLY:
    print(f"\nRUN_SUMMARY_ONLY=True — loading results from {csv_file}")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    loaded_df = pd.read_csv(csv_file)
    for _, row in loaded_df.iterrows():
        cond = row.get("Condition")
        if cond in all_condition_results:
            all_condition_results[cond].append(row.to_dict())
    print(f"  Loaded {len(loaded_df)} rows for conditions: "
          f"{loaded_df['Condition'].unique().tolist()}")
# Run the full analysis with parallel workers
elif batch_mode:
    jobs = [
        (sample, cond_data["prefix"], cond_name)
        for cond_name, cond_data in conditions.items()
        for sample in cond_data["samples"]
    ]
    n_workers = N_WORKERS or multiprocessing.cpu_count()
    print(f"\nRunning {len(jobs)} samples on {n_workers} worker(s) ...\n")

    if __name__ == "__main__":
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker, job): job for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                cond, r = result
                all_condition_results[cond].append(r)
# Run the full analysis for each sample subsequently
else:
    first_condition = list(conditions.keys())[0]
    first_sample    = conditions[first_condition]["samples"][0]
    prefix          = conditions[first_condition]["prefix"]

    out = process_sample(first_sample, prefix, condition_name=first_condition,
                         show_diagnostic_plots=plot_diagnostic_plots,
                         show_final_trace=plot_final_trace)
    if out is not None:
        out["Condition"] = first_condition
        all_condition_results[first_condition] = [out]

all_results = [r for cond in all_condition_results.values() for r in cond]


# =============================================================================
# 5. Save results to summary csv
# =============================================================================
def save_results_to_csv(all_results, csv_path):
    if not all_results:
        return
    results_df = pd.DataFrame(all_results)
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    if not existing_df.empty:
        for col in results_df.columns:
            if col not in existing_df.columns:
                existing_df[col] = np.nan
        existing_df = existing_df.reindex(columns=results_df.columns)

    if existing_df.empty:
        updated_df = results_df
    else:
        key_cols = ["Sample", "Condition"]
        for _, row in results_df.iterrows():
            mask = np.ones(len(existing_df), dtype=bool)
            for k in key_cols:
                if k in existing_df.columns:
                    mask &= (existing_df[k] == row.get(k))
            if mask.any():
                existing_df.loc[mask, :] = row.values
            else:
                existing_df = pd.concat([existing_df, pd.DataFrame([row])], ignore_index=True)
        updated_df = existing_df

    updated_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(updated_df[["Sample", "Condition",
                       "RRP scaled (%)", "Rec scaled (%)",
                       "Total scaled (%)", "Good ROIs used"]].to_string(index=False))


if not RUN_SUMMARY_ONLY:
    save_results_to_csv(all_results, csv_file)


# =============================================================================
# 6. OUTLIER DETECTION  —  ILR pivot + robust Mahalanobis distance
# =============================================================================

def ilr_pivot(X):
    # Apply isometric log-ratio transformation
    log_X = np.log(X)
    D = X.shape[1]
    Z = np.zeros((X.shape[0], D - 1))
    for j in range(1, D):
        scale = np.sqrt(j / (j + 1))
        Z[:, j - 1] = scale * (log_X[:, :j].mean(axis=1) - log_X[:, j])
    return Z


def detect_compositional_outliers(summary_df, frac_columns):
    # Outlier detection of compositional data using robust Mahalanobis distance [4]
    comp = summary_df[frac_columns].values / 100.0
    n, D = comp.shape # D = degrees of fredom

    # Replace exact zeros with half the observed non-zero minimum, then re-close
    nonzero_min = comp[comp > 0].min() if (comp > 0).any() else 1e-10
    comp = np.where(comp <= 0, nonzero_min / 2, comp)
    comp = comp / comp.sum(axis=1, keepdims=True) # normalize again for each row to sum to 1
    #isometric log-ratio transformation
    Z = ilr_pivot(comp)
    # take subset of Data with smalled covariance determinant
    h = max(0.5, (n + (D - 1) + 1) / (2 * n))
    # Fit minimum covariance determinant to the robust subset
    mcd = MinCovDet(support_fraction=h, random_state=42).fit(Z)

    diffs = Z - mcd.location_ # Robust center of each ilr coordinate
    cov_inv = np.linalg.inv(mcd.covariance_) # invert the robust covariance matrix
    maha_sq = np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs) # Calculate the squared distance for each sample
    maha_d  = np.sqrt(maha_sq) # Calculate distance 

    df_chi2  = D - 1 # Degrees of Freedom of chi_
    cutoff_d = np.sqrt(stats.chi2.ppf(0.975, df=df_chi2)) # Set cutoff to the 97.5th percentile of the chi-squared distribution
    p_value  = 1 - stats.chi2.cdf(maha_sq, df=df_chi2)

    is_outlier = maha_d > cutoff_d # detect outlier greater than the cutoff

    diag = pd.DataFrame({
        "Sample":          summary_df["Sample"].values,
        "Condition":       summary_df["Condition"].values,
        "ILR_1":           Z[:, 0].round(4),
        "ILR_2":           Z[:, 1].round(4),
        "Robust_Mahal_D":  maha_d.round(4),
        "p_value":         p_value.round(6),
        "Outlier":         is_outlier,
    })
    return is_outlier, diag, cutoff_d, h


# =============================================================================
# 7. SUMMARY PLOTS — ILR outlier removal → MANOVA (Pillai) → conditional
#    ANOVA + Tukey → visualisation
# =============================================================================


# ── CLR helpers (for visualisation axes) ─────────────────────────────────────
def _clr_transform(comp_array):
    # Calculate the centered log-ratio transformation of the compositional data
    X = np.array(comp_array, dtype=float)
    nonzero_min = X[X > 0].min() if (X > 0).any() else 1e-10 # replace zeros with small values to avoid log(0)
    delta = nonzero_min / 2
    X = np.where(X <= 0, delta, X)
    X = X / X.sum(axis=1, keepdims=True)
    log_X = np.log(X)
    return log_X - log_X.mean(axis=1, keepdims=True)


def _inv_clr(clr_array):
    # inverse centered log-ratio transformation
    e = np.exp(np.asarray(clr_array, dtype=float))
    return e / e.sum(axis=-1, keepdims=True)


# ── Effect sizes ─────────────────────────────────────────────────────────────
def _cohens_d_hedges_g(x, y):
    # Calculated Cohens d and Hedges' g effectsize
    x, y   = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    # Calulate only if n > 1
    if nx < 2 or ny < 2:
        return np.nan, np.nan
    # Calculate pooled standard deviation
    pooled_sd = np.sqrt(
        ((nx - 1) * x.std(ddof=1)**2 + (ny - 1) * y.std(ddof=1)**2) / (nx + ny - 2)
    )
    if pooled_sd == 0:
        return np.nan, np.nan
    d = (x.mean() - y.mean()) / pooled_sd # Cohens d
    return d, d * (1 - 3 / (4 * (nx + ny) - 9)) # Hedges' g


def significance_label(p):
    #define significance labels for plots
    return "****" if p < 0.0001 else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


# =============================================================================
# MAIN SUMMARY FUNCTION
# =============================================================================

def run_summary_plots(all_results, all_condition_results):
    # Plot all summary plots and perform statistical anaylsis
    if not all_results:
        return

    summary_df_raw  = pd.DataFrame(all_results)
    condition_names = [c for c in all_condition_results if all_condition_results[c]]
    n_conditions    = len(condition_names)

    frac_columns = ["RRP fraction (%)", "Rec fraction (%)", "Total fraction (%)"]
    ilr_columns  = ["ILR_1", "ILR_2"]
    clr_columns  = ["CLR_RRP", "CLR_RECP", "CLR_RESP"]

    # =====================================================================
    # Step 0 — Compositional outlier detection (ILR + MCD)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 0: COMPOSITIONAL OUTLIER DETECTION  (ILR pivot + MCD)")
    print("=" * 70)
    # Detect compositional outlier with Mahalanobis distance of isometric log-ratio transformed data
    outlier_mask, outlier_diag, cutoff_d, h_frac = detect_compositional_outliers(
        summary_df_raw, frac_columns
    )

    n_outliers = outlier_mask.sum()
    n_total    = len(summary_df_raw)

    print(f"\n  Samples analysed       : {n_total}")
    print(f"  MCD support fraction h : {h_frac:.4f}")
    print(f"  Chi2 cutoff (0.975, df=2): D = {cutoff_d:.4f}")
    print(f"  Outliers detected      : {n_outliers}")
    # Report outliers
    if n_outliers > 0:
        print(f"\n  {'Sample':<12s} {'Condition':<10s} {'Mahal D':>9s} {'p-value':>10s}")
        print("  " + "-" * 45)
        for _, row in outlier_diag[outlier_diag["Outlier"]].iterrows():
            print(f"  {row['Sample']:<12s} {row['Condition']:<10s} "
                  f"{row['Robust_Mahal_D']:9.4f} {row['p_value']:10.6f}")
    else:
        print("\n  No outliers detected — all samples retained.")

    # Remove outliers
    summary_df = summary_df_raw[~outlier_mask].reset_index(drop=True)
    print(f"\n  Samples after removal  : {len(summary_df)}")

    # Update per-condition sample counts
    for cond in condition_names:
        n_before = len(summary_df_raw[summary_df_raw["Condition"] == cond])
        n_after  = len(summary_df[summary_df["Condition"] == cond])
        removed  = n_before - n_after
        suffix   = f"  (removed {removed})" if removed > 0 else ""
        print(f"    {cond:<10s}: {n_before} → {n_after}{suffix}")

    # Save outlier diagnostics
    outlier_diag.to_csv(os.path.join(fig_output_folder, "outlier_diagnostics.csv"), index=False)
    print(f"\n  Outlier diagnostics saved to: {os.path.join(fig_output_folder, 'outlier_diagnostics.csv')}")

    # =====================================================================
    # Step 1 — ILR transform (for MANOVA / ANOVA)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 1: ILR TRANSFORM  (for statistical tests)")
    print("=" * 70)

    comp_clean = summary_df[frac_columns].values / 100.0
    # replace zeros with small values to avoid log(0)
    nz_min = comp_clean[comp_clean > 0].min() if (comp_clean > 0).any() else 1e-10
    comp_clean = np.where(comp_clean <= 0, nz_min / 2, comp_clean)
    comp_clean = comp_clean / comp_clean.sum(axis=1, keepdims=True) # normalize again for each row to sum to 1
    ilr_vals   = ilr_pivot(comp_clean) # perform isometric log-ratio transform

    for i, col in enumerate(ilr_columns):
        summary_df[col] = ilr_vals[:, i]

    # Also add CLR columns for visualisation
    clr_vals = _clr_transform(summary_df[frac_columns].values)
    for i, col in enumerate(clr_columns):
        summary_df[col] = clr_vals[:, i]

    print("  ILR columns added:", ilr_columns)
    print("  CLR columns added:", clr_columns, "(for visualisation)")

    # =====================================================================
    # Step 2 — MANOVA  (Pillai's trace on ILR coordinates)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STEP 2: MANOVA  (ILR_1 + ILR_2 ~ Condition)  — Pillai's trace")
    print("=" * 70)

    manova_df = summary_df[["Condition"] + ilr_columns].dropna()
    manova_significant = False
    # Initialise to NaN so downstream code never fails
    pillai_p = pillai_F = pillai_V = np.nan
    pillai_partial_eta2 = pillai_partial_eps2 = pillai_omega2 = np.nan
    try:
        # Perform MANOVA on isometric log-ratio transformed data
        manova_model  = MANOVA.from_formula("ILR_1 + ILR_2 ~ Condition", data=manova_df)
        manova_result = manova_model.mv_test()
        print(manova_result.summary())
        # Extract Pillai's trace values from MANOVA results
        pillai_table = manova_result.results["Condition"]["stat"]
        pillai_p     = pillai_table.loc["Pillai's trace", "Pr > F"]
        pillai_F     = pillai_table.loc["Pillai's trace", "F Value"]
        pillai_V     = pillai_table.loc["Pillai's trace", "Value"]
        num_df       = pillai_table.loc["Pillai's trace", "Num DF"]
        den_df       = pillai_table.loc["Pillai's trace", "Den DF"]

        p_vars   = len(ilr_columns)        
        k_minus1 = n_conditions - 1
        s        = min(p_vars, k_minus1)
        # Calculate partial eta^2 effect size
        pillai_partial_eta2 = pillai_V / s
        # Other effect size measurement were also computed but not used for the results
        pillai_partial_eps2 = max(0.0, (pillai_V - s * p_vars / den_df) / s)
        pillai_omega2 = max(0.0,
            (pillai_F * num_df - num_df) / (pillai_F * num_df + den_df + num_df))
        #eta^2 interpretation
        eta2_label = ("small" if pillai_partial_eta2 < 0.06
                    else "medium" if pillai_partial_eta2 < 0.14
                    else "large")
        print(f"\n  Pillai's trace V = {pillai_V:.4f},  F = {pillai_F:.4f},  "
              f"p = {pillai_p:.6f},  partial η² = {pillai_partial_eta2:.4f}  ({eta2_label})")

        manova_significant = pillai_p < 0.05
    except Exception as exc:
        print(f"  MANOVA failed: {exc}")
        pillai_p = np.nan

    # =====================================================================
    # Step 3 — Conditional ANOVA + Tukey HSD  (only if MANOVA significant)
    # =====================================================================
    tukey_results = {}
    anova_results = {}
    if manova_significant:
        print("\n" + "=" * 70)
        print("  STEP 3: ANOVA + TUKEY HSD  per ILR component  (MANOVA significant)")
        print("=" * 70)

        for pool_label, ilr_col in zip(pools[:2], ilr_columns):
            groups = [summary_df.loc[summary_df["Condition"] == c, ilr_col].dropna().values
                      for c in condition_names]
            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                F_stat, p_anova = f_oneway(*groups)     # Perform oneway ANOVA
                print(f"\n  ANOVA {pool_label} ({ilr_col}): F = {F_stat:.4f}, p = {p_anova:.6f}")
                anova_results[pool_label] = {"F": F_stat, "p": p_anova}
            else:
                print(f"\n  ANOVA {pool_label}: insufficient data")
                anova_results[pool_label] = {"F": np.nan, "p": np.nan}

            all_vals   = np.concatenate(groups)
            all_labels = np.concatenate([np.full(len(g), c) for g, c in zip(groups, condition_names)])
            # Perform pairwise Tukey's HSD test
            tukey      = pairwise_tukeyhsd(all_vals, all_labels, alpha=0.05)
            tukey_df   = pd.DataFrame(data=tukey._results_table.data[1:],
                                      columns=tukey._results_table.data[0])
            print(tukey.summary())

            pair_pvals = {}
            for _, row in tukey_df.iterrows():
                pair_pvals[(row["group1"], row["group2"])] = float(row["p-adj"])
                pair_pvals[(row["group2"], row["group1"])] = float(row["p-adj"])
            tukey_results[pool_label] = pair_pvals
    else:
        print("\n" + "=" * 70)
        print("  STEP 3: SKIPPED  — MANOVA not significant (Pillai p ≥ 0.05)")
        print("         No per-component ANOVAs or Tukey tests performed.")
        print("=" * 70)
    # Calculate effect sizes on ILR-transformed data
    effect_rows = []
    for pool_label, ilr_col in zip(pools[:2], ilr_columns):
        for cond_a, cond_b in combinations(condition_names, 2):
            xa = summary_df.loc[summary_df["Condition"] == cond_a, ilr_col].dropna().values
            xb = summary_df.loc[summary_df["Condition"] == cond_b, ilr_col].dropna().values
            d, g = _cohens_d_hedges_g(xa, xb)
            effect_rows.append({
                "Pool": pool_label, "Group1": cond_a, "Group2": cond_b,
                "Cohen_d": round(d, 4) if not np.isnan(d) else np.nan,
                "Hedges_g": round(g, 4) if not np.isnan(g) else np.nan,
            })

    effect_df = pd.DataFrame(effect_rows)
    print("\n" + "=" * 70)
    print("  EFFECT SIZES  (Cohen's d  /  Hedges' g)  — ILR scale")
    print("=" * 70)
    for _, row in effect_df.iterrows():
        print(f"  {row['Pool']:6s}  {row['Group1']:10s} vs {row['Group2']:10s}"
                f"  |  d = {row['Cohen_d']:+.3f}  |  g = {row['Hedges_g']:+.3f}")
    effect_df.to_csv(os.path.join(fig_output_folder, "effect_sizes_ilr.csv"), index=False)
    print(f"\n  Effect sizes saved to: {os.path.join(fig_output_folder, 'effect_sizes_ilr.csv')}")

    

    # =====================================================================
    # Step 4 — Visualisation
    # =====================================================================

    # ── Summary stats for the original fraction columns ──────────────────
    stat_dict = {cond: {pool: (summary_df.loc[summary_df["Condition"] == cond, col].mean(),
                               summary_df.loc[summary_df["Condition"] == cond, col].sem())
                        for pool, col in zip(pools, frac_columns)}
                 for cond in condition_names}

    # -----------------------------------------------------------------
    # Plot A — Stacked bar (overview)
    # -----------------------------------------------------------------
    x = np.arange(n_conditions)
    fig_a, ax_a = plt.subplots(figsize=(2.5 * n_conditions + 1, 6))
    bottoms = np.zeros(n_conditions)

    for pool_idx, pool in enumerate(pools):
        col   = frac_columns[pool_idx]
        means = np.array([stat_dict[c][pool][0] for c in condition_names])
        bar_colors_per_cond = [condition_shade_ramps[c][pool_idx] for c in condition_names]
        for ci, cond in enumerate(condition_names):
            ax_a.bar(x[ci], means[ci], bottom=bottoms[ci], width=0.5,
                     color=bar_colors_per_cond[ci], edgecolor="black", linewidth=0.8,
                     label=f"{cond} - {pool}" if ci == 0 else "_nolegend_")
        bottoms += means

    legend_handles_a = [
        Patch(facecolor=condition_shade_ramps["Control"][i], edgecolor="black",
              linewidth=0.7, label=pool)
        for i, pool in enumerate(pools)
    ]
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(condition_names, fontsize=18)
    ax_a.set_ylabel("Vesicle pool size [%]", fontsize=20)
    ax_a.set_ylim(0, 115)
    ax_a.set_title("Vesicle pool sizes", fontsize=28)
    ax_a.legend(handles=legend_handles_a, title="Pool", fontsize=18, title_fontsize=13,
                loc="upper left", framealpha=0.7)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    
    for j, cond in enumerate(condition_names):
        n = len(summary_df[summary_df["Condition"] == cond])
        ax_a.annotate(f"n={n}", xy=(j, 0), xycoords=("data", "axes fraction"),
                      xytext=(0, -32), textcoords="offset points",
                      ha="center", va="top", fontsize=12, color="gray")
    plt.tight_layout()
    fname = os.path.join(fig_output_folder, "summary_stacked_bar.png")
    fig_a.savefig(fname); print(f"  Saved: {fname}")
    plt.show(); plt.close(fig_a)

    # -----------------------------------------------------------------
    # Plot B — Grouped bar of fractions 
    # -----------------------------------------------------------------
    group_width = 0.8
    bar_width   = group_width / n_conditions
    x_pool      = np.arange(len(pools))

    fig_b, ax_b = plt.subplots(figsize=(4 + 2.5 * n_conditions, 8))

    all_bar_tops = []
    for cond in condition_names:
        cdf = summary_df[summary_df["Condition"] == cond]
        for col in frac_columns:
            all_bar_tops.append(cdf[col].mean() + cdf[col].sem())
    data_ymax = max(all_bar_tops) if all_bar_tops else 100

    for ci, cond in enumerate(condition_names):
        cdf    = summary_df[summary_df["Condition"] == cond]
        n      = len(cdf)
        means  = [cdf[col].mean() for col in frac_columns]
        sems   = [cdf[col].sem()  for col in frac_columns]
        print(f"\n{cond} (n={n}):")
        for pool, m, s in zip(pools, means, sems):
            print(f"  {pool}: {m:.2f}% ± {s:.2f}% SEM")
        offset = (ci - (n_conditions - 1) / 2) * bar_width

        # Layer 1: bar FILL only (zorder=0, alpha=1)
        for j, (x_p, mv) in enumerate(zip(x_pool + offset, means)):
            ax_b.bar(x_p, mv, width=bar_width * 0.9,
                     color=condition_shade_ramps[cond][0], edgecolor="none",
                     linewidth=0,
                     label=f"{cond} (n={n})" if j == 0 else "_nolegend_",
                     zorder=0, alpha=1.0)

        # Layer 2: scatter points (zorder=1)
        rng = np.random.default_rng(seed=42)
        for j, col in enumerate(frac_columns):
            jit = rng.uniform(-bar_width * 0.2, bar_width * 0.2, size=n)
            ax_b.scatter(np.full(n, x_pool[j] + offset) + jit, cdf[col].values,
                         color="gray", alpha=0.8, s=22, zorder=1, linewidths=0.8)

        # Layer 3: bar EDGES + error bars (zorder=3)
        for j, (x_p, mv, sv) in enumerate(zip(x_pool + offset, means, sems)):
            ax_b.bar(x_p, mv, yerr=sv, width=bar_width * 0.9,
                     color="none", edgecolor="black", linewidth=0.8,
                     capsize=5, error_kw=dict(elinewidth=1.5, ecolor="black", zorder=3),
                     label="_nolegend_", zorder=3)

    # Significance brackets — only if MANOVA was significant
    if manova_significant and tukey_results:
        bracket_base_y     = data_ymax * 1.08
        step_y             = data_ymax * 0.10
        bracket_y_per_pool = {j: bracket_base_y for j in range(len(pools))}

        for j, pool_label in enumerate(pools):
            if pool_label not in tukey_results:
                continue
            for ci, ck in combinations(range(n_conditions), 2):
                cond_a, cond_b = condition_names[ci], condition_names[ck]
                p = tukey_results[pool_label].get((cond_a, cond_b), np.nan)
                if np.isnan(p):
                    continue
                oa = (ci - (n_conditions - 1) / 2) * bar_width
                ob = (ck - (n_conditions - 1) / 2) * bar_width
                x_a, x_b  = x_pool[j] + oa, x_pool[j] + ob
                y_bracket  = bracket_y_per_pool[j]
                bracket_y_per_pool[j] += step_y
                ax_b.plot([x_a, x_a, x_b, x_b],
                          [y_bracket - step_y * 0.15, y_bracket,
                           y_bracket, y_bracket - step_y * 0.15],
                          color="black", linewidth=1.0, clip_on=False)
                ax_b.text((x_a + x_b) / 2, y_bracket + step_y * 0.05,
                          significance_label(p),
                          ha="center", va="bottom", fontsize=12, clip_on=False)

        final_ymax = max(100, max(bracket_y_per_pool.values()) + step_y * 1.5)
    else:
        final_ymax = max(100, data_ymax * 1.15)

    ax_b.set_ylim(0, final_ymax)
    ax_b.set_xticks(x_pool)
    ax_b.set_xticklabels(pool_xlabels, fontsize=18)
    ax_b.set_ylabel("Vesicle pool size (%)", fontsize=22)
    ax_b.tick_params(axis='y',labelsize=18)

    title_suffix = "(Tukey HSD on ILR)" if manova_significant else "(MANOVA n.s.)"
    ax_b.set_title(f"Vesicle pool sizes", fontsize=28)
    ax_b.legend(fontsize=18, framealpha=0.8, loc="upper left")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    plt.tight_layout()
    fname = os.path.join(fig_output_folder, "summary_grouped_bar_tukey.png")
    fig_b.savefig(fname); print(f"  Saved: {fname}")
    plt.show(); plt.close(fig_b)

    # -----------------------------------------------------------------
    # Plot C — CLR boxplots (brackets only if MANOVA sig.)
    #
    # Split-zorder:  box fill (zorder=0, alpha=1) → scatter (zorder=1)
    #                → box lines (zorder=3)
    # -----------------------------------------------------------------
    fig_c, axes_c = plt.subplots(1, 3, figsize=(5 * 3, 6), sharey=False)

    for ax_idx, (pool_label, clr_col) in enumerate(zip(pools, clr_columns)):
        ax_c = axes_c[ax_idx]

        box_data = [summary_df.loc[summary_df["Condition"] == c, clr_col].dropna().values
                    for c in condition_names]

        # Layer 1: box FILL only — invisible lines (zorder=0, alpha=1)
        bp_fill = ax_c.boxplot(
            box_data, positions=np.arange(n_conditions), widths=0.5,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="none", linewidth=0),
            whiskerprops=dict(color="none", linewidth=0),
            capprops=dict(color="none", linewidth=0),
            zorder=0,
        )
        for patch, cond in zip(bp_fill["boxes"], condition_names):
            patch.set_facecolor(condition_shade_ramps[cond][0])
            patch.set_alpha(1.0)
            patch.set_edgecolor("none")
            patch.set_zorder(0)

        # Layer 2: scatter (zorder=1)
        rng = np.random.default_rng(seed=42)
        for ci, (cond, vals) in enumerate(zip(condition_names, box_data)):
            jit = rng.uniform(-0.12, 0.12, size=len(vals))
            ax_c.scatter(np.full(len(vals), ci) + jit, vals,
                         color="gray", alpha=0.8, s=24, zorder=1, linewidths=0.6,
                         edgecolors="black")

        # Layer 3: box LINES only — transparent fill (zorder=3)
        bp_lines = ax_c.boxplot(
            box_data, positions=np.arange(n_conditions), widths=0.5,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.5, zorder=3),
            whiskerprops=dict(color="black", linewidth=0.8, zorder=3),
            capprops=dict(color="black", linewidth=0.8, zorder=3),
            zorder=3,
        )
        for patch in bp_lines["boxes"]:
            patch.set_facecolor("none")
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
            patch.set_zorder(3)

        # Tukey brackets — only if MANOVA significant and pool has Tukey results
        if manova_significant and pool_label in tukey_results:
            all_maxes = [v.max() for v in box_data if len(v)]
            all_mins  = [v.min() for v in box_data if len(v)]
            bracket_base = max(all_maxes) * 1.08 if all_maxes else 1.0
            step         = (max(all_maxes) - min(all_mins)) * 0.12 if all_maxes else 0.1
            bracket_y    = bracket_base

            for ci, ck in combinations(range(n_conditions), 2):
                cond_a, cond_b = condition_names[ci], condition_names[ck]
                p = tukey_results[pool_label].get((cond_a, cond_b), np.nan)
                if np.isnan(p):
                    continue
                ax_c.plot([ci, ci, ck, ck],
                          [bracket_y - step * 0.15, bracket_y,
                           bracket_y, bracket_y - step * 0.15],
                          color="black", linewidth=1.0, clip_on=False)
                ax_c.text((ci + ck) / 2, bracket_y + step * 0.05,
                          significance_label(p),
                          ha="center", va="bottom", fontsize=11, clip_on=False)
                bracket_y += step

        ax_c.set_xticks(range(n_conditions))
        ax_c.set_xticklabels(condition_names, fontsize=13)
        ax_c.set_ylabel("CLR value", fontsize=14)
        ax_c.set_title(f"{pool_xlabels[ax_idx]}", fontsize=16)
        ax_c.spines["top"].set_visible(False)
        ax_c.spines["right"].set_visible(False)

        # ANOVA annotation — only if it was run
        if pool_label in anova_results:
            F_val = anova_results[pool_label]["F"]
            p_val = anova_results[pool_label]["p"]
            if not np.isnan(F_val):
                ax_c.text(0.02, 0.97,
                          f"F = {F_val:.2f}, p = {p_val:.4f}",
                          transform=ax_c.transAxes, fontsize=10,
                          va="top", ha="left",
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    manova_note = (f"Pillai V = {pillai_V:.4f}, p = {pillai_p:.4f}, "
                    f"partial η² = {pillai_partial_eta2:.4f}"
                   if not np.isnan(pillai_p) else "MANOVA not computed")
    fig_c.suptitle(f"CLR-transformed vesicle pool fractions\n({manova_note})",
                   fontsize=20, y=1.04)
    plt.tight_layout()
    fname = os.path.join(fig_output_folder, "summary_clr_boxplots.png")
    fig_c.savefig(fname); print(f"  Saved: {fname}")
    plt.show(); plt.close(fig_c)

    # -----------------------------------------------------------------
    # Save statistics table
    # -----------------------------------------------------------------
    stats_rows = []
    for pool_label, ilr_col in zip(pools[:2], ilr_columns):
        row = {
            "Pool": pool_label,
            "ILR_column": ilr_col,
        }
        if pool_label in anova_results:
            row["ANOVA_F"] = anova_results[pool_label]["F"]
            row["ANOVA_p"] = anova_results[pool_label]["p"]
        else:
            row["ANOVA_F"] = np.nan
            row["ANOVA_p"] = np.nan

        for cond in condition_names:
            vals = summary_df.loc[summary_df["Condition"] == cond, ilr_col].dropna()
            row[f"{cond}_mean_ILR"] = vals.mean()
            row[f"{cond}_std_ILR"]  = vals.std()
            row[f"{cond}_n"]        = len(vals)

        if pool_label in tukey_results:
            for ci, ck in combinations(range(n_conditions), 2):
                ca, cb = condition_names[ci], condition_names[ck]
                p = tukey_results[pool_label].get((ca, cb), np.nan)
                row[f"Tukey_{ca}_vs_{cb}_p"] = p

        stats_rows.append(row)
    stats_rows.insert(0, {
        "Pool": "MANOVA (all)",
        "ILR_column": "ILR_1 + ILR_2",
        "ANOVA_F": pillai_F,
        "ANOVA_p": pillai_p,
        "Pillai_V": pillai_V,
        "Partial_eta2": pillai_partial_eta2,
        "Partial_eps2": pillai_partial_eps2,
        "Partial_omega2": pillai_omega2,
        })
    stats_table = pd.DataFrame(stats_rows)
    stats_path  = os.path.join(fig_output_folder, "ilr_statistics_summary.csv")
    stats_table.to_csv(stats_path, index=False)
    print(f"\n  Full ILR statistics saved to: {stats_path}")


run_summary_plots(all_results, all_condition_results)
