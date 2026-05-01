# ChatGPT (OpenAI) and Claude (Antrophic) were partly used as assistance for code generation which was reviewed and verified by the creator.

# import libraries
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from peakdetect_sbalzarini import peakdetect_sbalzarini
from relative_fluorescence import relative_fluorescence
import re

# -------------------------
# 0. Helper functions
# -------------------------
def extract_frame_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1])

def extract_sample_name(filename):
    match = re.search(r'sample\d+', filename)
    return match.group(0) if match else "unknown"

def frames_to_seconds(n_frames):
    return np.arange(n_frames) / FRAME_RATE

# -------------------------
# 1. Parameters
# -------------------------
# store all current level measurments of one sample in the same folder
# list all sample folders
sample_folders = [
    r"sample_folder"
]
# set output folder
results_dir = r"results_folder"
os.makedirs(results_dir, exist_ok=True)
# list all current levels that were recorded
stim_series = ["1mA", "10mA", "25mA", "35mA", "45mA", "55mA", "675mA", "80mA"]
FRAME_RATE = 1  # Set framerate
w = 20          # Set radius for peak detection algorithm
pth = 1         # Set intensity threshold for peak detection algorithm
# Transform filename currrent levels to current values
stim_to_current = {
    "1mA": 1, "10mA": 10, "25mA": 25, "35mA": 35,
    "45mA": 45, "55mA": 55, "675mA": 67.5, "80mA": 80
}
PIXEL_SIZE_UM = 0.83 # pixelsize of the images adjust to microscope setup (0.64 for 10x - 0.83 for 20x)
SCALEBAR_UM = 50  # 50 µm bar 
# -------------------------
# 2. Global plot parameter
# -------------------------
mpl.rcParams.update({
    "font.size":             14,
    "axes.titlesize":        16,
    "axes.labelsize":        15,
    "xtick.labelsize":       13,
    "ytick.labelsize":       13,
    "legend.fontsize":       12,
    "legend.title_fontsize": 13,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.linewidth":        0.8,
    "xtick.major.width":     0.8,
    "ytick.major.width":     0.8,
    "lines.linewidth":       1.5,
    "figure.dpi":            150,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
})
# Color palette to distinguish the different current levels
PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]
stim_colors = {s: PALETTE[i] for i, s in enumerate(stim_series)}
ERRORBAR_COLOR = "#1a3f6f"

# -------------------------
# 3. Process each sample folder
# -------------------------
all_normalized_traces = {}
all_norm_peak_values  = {}
all_raw_traces        = {}
all_peak_values       = {}

n_total_samples = 0

for folder_idx, folder in enumerate(sample_folders):
    print(f"\n{'='*50}")
    print(f"Processing folder {folder_idx+1}/{len(sample_folders)}: {folder}")
    print(f"{'='*50}")
    # read sample folder
    all_files = [f for f in os.listdir(folder) if f.endswith(".tif")]
    samples = sorted(set(extract_sample_name(f) for f in all_files))
    print("Detected samples:", samples)
    # process each sample
    for sample in samples:
        print(f"\n--- Processing {sample} (folder {folder_idx+1}) ---")
        n_total_samples += 1

        # Build difference between baseline and peak fluorescence to generate an image with only reactive regions for ROI detection 
        # 45mA measurement was chosen because it was the first recorded 
        files_45mA = sorted(
            [f for f in os.listdir(folder) if "45mA" in f and sample in f],
            key=extract_frame_number
        )
        if not files_45mA:
            print(f"No 45mA files found for {sample}, skipping...")
            n_total_samples -= 1
            continue

        stack_45mA = np.array([tifffile.imread(os.path.join(folder, f)) for f in files_45mA])

        # Baseline: average of first 3 frames
        baseline_img = stack_45mA[:3].mean(axis=0)

        # Peak: find brightest frame, then average 3 frames around it
        frame_means = stack_45mA.mean(axis=(1, 2))
        peak_frame = np.argmax(frame_means)
        start = max(0, peak_frame - 1)
        end = min(len(stack_45mA), peak_frame + 2)
        peak_img = stack_45mA[start:end].mean(axis=0)

        # Difference image
        diff_img = peak_img - baseline_img
        diff_img = np.clip(diff_img, 0, None)
        # ROI detection with feature point detection algorithm
        regions = peakdetect_sbalzarini(diff_img, w, pth)
        n_rois = len(regions)
        print(f"Detected {n_rois} ROIs for {sample} (difference image, peak frame={peak_frame})")

        if n_rois == 0:
            print(f"No ROIs detected for {sample}, skipping...")
            n_total_samples -= 1
            continue
        # Save difference image with ROIs and scale bar
        scalebar_px = SCALEBAR_UM / PIXEL_SIZE_UM

        fig_roi, ax_roi = plt.subplots(figsize=(5, 4))
        ax_roi.imshow(diff_img, cmap="viridis")

        # Plot ROI centroids
        for i, region in enumerate(regions):
            cx, cy = region["Centroid"]
            ax_roi.plot(cx, cy, 'o', color="orange", markersize=3, alpha=0.6,
                        label=f'{n_rois} detected ROIs' if i == 0 else None)

        ax_roi.set_title(f"Difference image with detected ROIs")
        ax_roi.legend(fontsize=8, framealpha=0.8, loc='upper left')
        ax_roi.axis("image")
        ax_roi.spines["left"].set_visible(False)
        ax_roi.spines["bottom"].set_visible(False)
        ax_roi.set_xticks([])
        ax_roi.set_yticks([])

        # Scale bar
        img_h, img_w = diff_img.shape
        margin_x = img_w * 0.05
        margin_y = img_h * 0.05
        bar_thickness = max(2, img_h * 0.012)
        bar_x_end = img_w - margin_x
        bar_x_start = bar_x_end - scalebar_px
        bar_y = img_h - margin_y

        ax_roi.add_patch(plt.Rectangle(
            (bar_x_start, bar_y - bar_thickness / 2),
            scalebar_px, bar_thickness,
            color="white", zorder=10
        ))
        ax_roi.text(
            (bar_x_start + bar_x_end) / 2, bar_y - bar_thickness * 1.8,
            f"{SCALEBAR_UM} µm",
            color="white", fontsize=8, ha="center", va="bottom",
            fontweight="bold", zorder=11
        )

        fig_roi.tight_layout()
        for ext in [".pdf", ".png"]:
            fig_roi.savefig(os.path.join(results_dir, f"folder{folder_idx+1}_{sample}_diff_rois{ext}"), dpi=300)
        print(f"  Saved ROI map with scale bar")
        plt.close(fig_roi)

        # Extract fluorescence traces for all stimulation measurements
        traces_by_stim = {}
        for stim in stim_series:
            stim_files = sorted(
                [f for f in os.listdir(folder) if stim in f and sample in f],
                key=extract_frame_number
            )
            if not stim_files:
                print(f"  No files found for {sample} {stim}")
                continue

            print(f"  Processing {len(stim_files)} files for {sample} {stim}...")
            img_stack = np.array([tifffile.imread(os.path.join(folder, f)) for f in stim_files])
            n_frames = img_stack.shape[0]

            # Vectorized ROI extraction
            meanstack = np.zeros((n_rois, n_frames))
            for i, region in enumerate(regions):
                row_idx, col_idx = region["PixelIdxList"]  
                # Filter pixels outside image bounds
                valid = (row_idx < img_stack.shape[1]) & (col_idx < img_stack.shape[2])
                rows, cols = row_idx[valid], col_idx[valid]
                if len(rows) == 0:
                    meanstack[i, :] = 0
                    continue
                meanstack[i, :] = img_stack[:, rows, cols].mean(axis=1)
            # use the relative fluorescence algorithm 
            dF_F = relative_fluorescence(meanstack)
            traces_by_stim[stim] = dF_F

        # Per-sample plot of each fluorescence trace
        fig, ax = plt.subplots(figsize=(7, 4))
        for stim, dF_F in traces_by_stim.items():
            mean_trace = dF_F.mean(axis=0)
            t = frames_to_seconds(len(mean_trace))
            ax.plot(t, mean_trace, color=stim_colors[stim], label=f"{stim_to_current[stim]} mA")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r"Relative Fluorescence$\Delta F/F$")
        ax.set_title(f"Relative fluorescence response to different current levels")
        ax.legend(title="Current", loc="upper right", frameon=True,
                  ncol=2 if len(traces_by_stim) > 5 else 1, framealpha=0.8)
        fig.tight_layout()
        for ext in [".pdf", ".png"]:
            fig.savefig(os.path.join(results_dir, f"folder{folder_idx+1}_{sample}_mean_traces{ext}"))
        plt.close(fig)

        # Normalization to the 45mA trace amplitude
        mean_traces = {stim: dF_F.mean(axis=0) for stim, dF_F in traces_by_stim.items()}

        if "45mA" in mean_traces:
            ref_min = mean_traces["45mA"].min()
            ref_max = mean_traces["45mA"].max()
        else:
            # fallback to global min/max if 45mA is missing
            ref_min = min(tr.min() for tr in mean_traces.values())
            ref_max = max(tr.max() for tr in mean_traces.values())

        for stim, mean_trace in mean_traces.items():
            norm = ((mean_trace - ref_min) / (ref_max - ref_min)
                    if ref_max > ref_min else np.zeros_like(mean_trace))
            all_normalized_traces.setdefault(stim, []).append(norm)
            all_norm_peak_values.setdefault(stim, []).append(norm.max())

        # Collect raw traces
        for stim, dF_F in traces_by_stim.items():
            mean_trace = dF_F.mean(axis=0)
            all_raw_traces.setdefault(stim, []).append(mean_trace)
            all_peak_values.setdefault(stim, []).append(mean_trace.max())

print(f"\n\nTotal samples processed: {n_total_samples}")

# -------------------------
# 4. Grand mean plots
# -------------------------

# Plot A: Mean normalized traces of all samples 
if all_normalized_traces:
    fig, ax = plt.subplots(figsize=(5, 4))
    for stim in stim_series:
        if stim not in all_normalized_traces:
            continue
        traces = all_normalized_traces[stim]
        min_len = min(len(t) for t in traces)
        stack = np.array([t[:min_len] for t in traces])
        grand_mean = stack.mean(axis=0)
        grand_sem = stack.std(axis=0) / np.sqrt(stack.shape[0])
        t = frames_to_seconds(len(grand_mean))
        ax.plot(t, grand_mean, color=stim_colors[stim], label=f"{stim_to_current[stim]} mA")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Normalized $\Delta F/F$")
    ax.set_title(f"Mean normalized relative fluorescence traces (n={n_total_samples})")
    ax.legend(title="Current", loc="upper right", frameon=True,
              ncol=2 if len(all_normalized_traces) > 5 else 1,framealpha=0.8)
    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(results_dir, f"grand_mean_normalized_traces{ext}"))
    print(f"Saved: grand_mean_normalized_traces")
    plt.close(fig)

# Plot B: Mean normalized peak vs current for all samples
if all_norm_peak_values:
    raw_currents, raw_peaks, raw_peak_sem, raw_peak_all = [], [], [], []
    for stim in stim_series:
        if stim not in all_norm_peak_values:
            continue
        vals = np.array(all_norm_peak_values[stim])
        raw_currents.append(stim_to_current[stim])
        raw_peaks.append(vals.mean())
        raw_peak_sem.append(vals.std() / np.sqrt(len(vals)))
        raw_peak_all.append(vals)
    order = np.argsort(raw_currents)
    raw_currents  = np.array(raw_currents)[order]
    raw_peaks     = np.array(raw_peaks)[order]
    raw_peak_sem  = np.array(raw_peak_sem)[order]
    raw_peak_all  = [raw_peak_all[i] for i in order]

    fig, ax = plt.subplots(figsize=(5, 4))
    
    ax.errorbar(raw_currents, raw_peaks, fmt='-o', capsize=4, color=ERRORBAR_COLOR,
                markerfacecolor="white", markeredgewidth=1.5, linewidth=1.5)
    
    ax.set_xlabel("Stimulation current [mA]")
    ax.set_ylabel(r"Peak $\Delta F/F$")
    ax.set_title(f"Peak normalized relative fluorescence\nfor different current level")
    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(results_dir, f"norm_peak_vs_current{ext}"))
    print(f"Saved: norm_peak_vs_current")
    plt.close(fig)

# Plot C: Mean raw traces for all samples 
if all_raw_traces:
    fig, ax = plt.subplots(figsize=(7, 4))
    for stim in stim_series:
        if stim not in all_raw_traces:
            continue
        traces = all_raw_traces[stim]
        min_len = min(len(t) for t in traces)
        stack = np.array([t[:min_len] for t in traces])
        grand_mean = stack.mean(axis=0)
        grand_sem = stack.std(axis=0) / np.sqrt(stack.shape[0])
        t = frames_to_seconds(len(grand_mean))
        ax.plot(t, grand_mean, color=stim_colors[stim], label=f"{stim_to_current[stim]} mA")
        '''
        ax.fill_between(t, grand_mean - grand_sem, grand_mean + grand_sem,
                        color=stim_colors[stim], alpha=0.2, linewidth=0)
        '''
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\Delta F/F$")
    ax.set_title(f"Mean relative fluorescence traces (n={n_total_samples})")
    ax.legend(title="Current", loc="upper right", frameon=True,
              ncol=2 if len(all_raw_traces) > 5 else 1, framealpha=0.8)
    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(results_dir, f"grand_mean_raw_traces{ext}"))
    print(f"Saved: grand_mean_raw_traces")
    plt.close(fig)

# Plot D: Mean raw peak vs current for all samples
if all_peak_values:
    raw_currents, raw_peaks, raw_peak_sem = [], [], []
    for stim in stim_series:
        if stim not in all_peak_values:
            continue
        vals = np.array(all_peak_values[stim])
        raw_currents.append(stim_to_current[stim])
        raw_peaks.append(vals.mean())
        raw_peak_sem.append(vals.std() / np.sqrt(len(vals)))
    order = np.argsort(raw_currents)
    raw_currents = np.array(raw_currents)[order]
    raw_peaks = np.array(raw_peaks)[order]
    raw_peak_sem = np.array(raw_peak_sem)[order]

    fig, ax = plt.subplots(figsize=(5, 4))
    
    ax.errorbar(raw_currents, raw_peaks,
                fmt='-o', capsize=4, color=ERRORBAR_COLOR,
                markerfacecolor="white", markeredgewidth=1.5, linewidth=1.5)
    
    ax.set_xlabel("Stimulation current [mA]")
    ax.set_ylabel(r"Peak $\Delta F/F$")
    ax.set_title(f"Peak relative fluorescence\nfor different current level")
    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(results_dir, f"grand_mean_raw_peak_vs_current{ext}"))
    print(f"Saved: grand_mean_raw_peak_vs_current")
    plt.close(fig)

print(f"\nAll plots saved to: {results_dir}")
