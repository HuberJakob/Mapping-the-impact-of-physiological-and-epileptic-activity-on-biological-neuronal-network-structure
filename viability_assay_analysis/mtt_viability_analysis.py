# ChatGPT (OpenAI) and Claude (Antrophic) were partly used as assistance for code generation which was reviewed and verified by the creator.

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from itertools import combinations
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# =============================================================================
# 0. Global plot parameter
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

# =========================
# Settings
# =========================
REMOVE_OUTLIERS = True  # Toggle automatic outlier removal
SECOND_BATCH = True     # Toggle if the samples were collected in two batches

# =========================
# File paths
# =========================
base = Path(r"base_folder_first_batch")
figures_path = base / "Figures"
figures_path.mkdir(exist_ok=True)

file_550 = base / "20260212_viability_550.csv"  # MTT measurement at 550 nm
file_680 = base / "20260212_viability_680.csv"  # Reference measurement at 680 nm

if SECOND_BATCH:
    base2 = Path(r"base_folder_second_batch")
    file_550_2 = base2 / "20260219_viability_550.csv"   # MTT measurement at 550 nm
    file_680_2 = base2 / "20260219_viability_680.csv"   # Reference measurement at 680 nm

# =============================================================================
# Assign colors to the conditions
# =============================================================================
_set3 = sns.color_palette("Set3", 12)

condition_base_colors = {
    "control": _set3[4],
    "cTBS":    _set3[5],
    "iTBS":    _set3[6],
}

_pools_n = 3
condition_shade_ramps = {
    cond: sns.dark_palette(base_color, n_colors=_pools_n + 2, reverse=True)[1:_pools_n+1]
    for cond, base_color in condition_base_colors.items()
}

CONDITION_COLORS = {cond: ramp[0] for cond, ramp in condition_shade_ramps.items()}

# =========================
# Define Groups according to the 96-well plate layout
# =========================
mtt_groups = {
    "control": ["E1", "E2", "E3", "E4", "E5", "E6"], 
    "cTBS":    ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","G1","G2","G3","G4"], 
    "blank":   ["H1", "H2"] # Reference wells with only MTT reagent
}

if SECOND_BATCH:
    mtt_groups_2 = {
        "control": ["E1", "E2", "E3", "E4"],
        "iTBS":    ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","G1","G2","G3","G4"],
        "blank":   ["H1", "H2"] # Reference wells with only MTT reagent
    }

# =========================
# Read Clariostar plate
# =========================
def read_clariostar_plate(file):
    # Read the generated .csv from the measurement and assign the values to the according wells  
    with open(file, encoding="latin1") as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(";1;2;3;4;5;6;7;8;9;10;11;12"):
            start = i
            break
    data_lines = lines[start+1:start+9]
    rows = []
    index = []
    for line in data_lines:
        parts = line.strip().split(";")
        index.append(parts[0])
        values = [float(x.replace(",", ".")) for x in parts[1:13]]
        rows.append(values)
    df = pd.DataFrame(rows, index=index, columns=[str(i) for i in range(1, 13)])
    return df

# =========================
# Helper functions
# =========================
def get_wells(df, wells):
    # Get the absorbance value from each well
    values = []
    for w in wells:
        row = w[0]
        col = w[1:]
        values.append(df.loc[row, col])
    return np.array(values)

def detect_iqr_outliers(values, wells):
    # Inter-quartile outlier detection
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr  # lower outlier threshold
    upper = q3 + 1.5 * iqr  # upper outlier threshold
    clean_vals, clean_wells, out_vals, out_wells = [], [], [], []
    # Separate outliers from non-outliers 
    for v, w in zip(values, wells):
        if lower <= v <= upper:
            clean_vals.append(v)
            clean_wells.append(w)
        else:
            out_vals.append(v)
            out_wells.append(w)
    return np.array(clean_vals), clean_wells, np.array(out_vals), out_wells

def cohen_d_hedges_g(x, y):
    # Calculate effect size with Cohens d and Hedges g
    x = np.array(x)
    y = np.array(y)
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1)*np.std(x, ddof=1)**2 + (ny - 1)*np.std(y, ddof=1)**2) / (nx + ny - 2))
    d = (np.mean(x) - np.mean(y)) / pooled_std
    correction = 1 - (3 / (4*(nx + ny) - 9))
    g = d * correction
    return d, g

# =========================
# Read plates and correct them with reference subtraction
# =========================
plate_550 = read_clariostar_plate(file_550)
plate_680 = read_clariostar_plate(file_680)
corr_550 = plate_550 - plate_680    # Correct the measurement by subtracting background absorbance from the plate

blank_mean = np.mean(get_wells(corr_550, mtt_groups["blank"]))
corr_550 = corr_550 - blank_mean    # Remove MTT reagent induced absorbance

if SECOND_BATCH:
    plate_550_2 = read_clariostar_plate(file_550_2)
    plate_680_2 = read_clariostar_plate(file_680_2)
    corr_550_2 = plate_550_2 - plate_680_2  # Correct the measurement by subtracting background absorbance from the plate
    blank_mean_2 = np.mean(get_wells(corr_550_2, mtt_groups_2["blank"]))
    corr_550_2 = corr_550_2 - blank_mean_2  # Remove MTT reagent induced absorbance

# =========================
# IQR OUTLIER DETECTION (per batch separately)
# =========================
mtt_clean = {}
mtt_clean_wells = {}
outlier_records = []

for group, wells in mtt_groups.items():
    if group == "blank": # Exclude blank wells
        continue
    # Get absorbance values
    vals = get_wells(corr_550, wells)
    # Detect ouliers
    clean_vals, clean_wells, out_vals, out_wells = detect_iqr_outliers(vals, wells)
    print(f"\nGroup: {group}")
    print(f"Original wells: {wells}")
    print(f"IQR outliers: {out_wells}")
    # Remove outliers
    mtt_clean[group] = clean_vals if REMOVE_OUTLIERS else vals
    mtt_clean_wells[group] = clean_wells if REMOVE_OUTLIERS else wells
    for w, v in zip(out_wells, out_vals):
        outlier_records.append({"Assay": "MTT", "Batch": "Batch 1", "Group": group, "Well": w, "Value": v})

if outlier_records:
    outlier_df = pd.DataFrame(outlier_records)
    print("\nDetected IQR outlier wells:")
    print(outlier_df.to_string(index=False))
    outlier_df.to_csv(base / "MTT_Outlier_wells_IQR.csv", index=False, sep=";")
else:
    print("\nNo IQR outliers detected.")

if SECOND_BATCH:
    mtt_clean_2 = {}
    mtt_clean_wells_2 = {}
    outlier_records_2 = []
    for group, wells in mtt_groups_2.items(): # Exclude blank wells
        if group == "blank":
            continue
        # Get absorbance values
        vals_2 = get_wells(corr_550_2, wells)
        # Detect ouliers
        clean_vals_2, clean_wells_2, out_vals_2, out_wells_2 = detect_iqr_outliers(vals_2, wells)

        print(f"\nGroup: {group}")
        print(f"Original wells: {wells}")
        print(f"IQR outliers: {out_wells_2}")
        # Remove outliers
        mtt_clean_2[group] = clean_vals_2 if REMOVE_OUTLIERS else vals_2
        mtt_clean_wells_2[group] = clean_wells_2 if REMOVE_OUTLIERS else wells
        for w, v in zip(out_wells_2, out_vals_2):
            outlier_records_2.append({"Assay": "MTT", "Batch": "Batch 2", "Group": group, "Well": w, "Value": v})
    if outlier_records_2:
        outlier_df_2 = pd.DataFrame(outlier_records_2)
        print("\nDetected IQR outlier wells (second batch):")
        print(outlier_df_2.to_string(index=False))
        outlier_df_2.to_csv(base2 / "MTT_Outlier_wells_IQR_batch2.csv", index=False, sep=";")
    else:
        print("\nNo IQR outliers detected in second batch.")

# =========================
# MTT Viability
# =========================
control_mean_mtt = np.mean(mtt_clean["control"])
mtt_viability = {group: vals / control_mean_mtt * 100 for group, vals in mtt_clean.items()} # Normalize to the mean of the control group

if SECOND_BATCH:
    control_mean_mtt_2 = np.mean(mtt_clean_2["control"])
    mtt_viability_2 = {group: vals / control_mean_mtt_2 * 100 for group, vals in mtt_clean_2.items()} # Normalize to the mean of the control group

# =========================
# Create dataframe for data handling
# =========================
def make_df_from_clean_wells(clean_wells_dict, clean_vals_dict, original_groups_dict, viability_dict, batch_label):
    rows = []
    for group, wells in clean_wells_dict.items():
        for i, w in enumerate(wells):
            val = viability_dict[group][i]
            rows.append({"Group": group, "Value": val, "Batch": batch_label})
    return pd.DataFrame(rows)

mtt_df_b1 = make_df_from_clean_wells(
    mtt_clean_wells, mtt_clean,
    {k: v for k, v in mtt_groups.items() if k != "blank"},
    mtt_viability,
    "Batch 1 (cTBS)"
)

if SECOND_BATCH:
    mtt_df_b2 = make_df_from_clean_wells(
        mtt_clean_wells_2, mtt_clean_2,
        {k: v for k, v in mtt_groups_2.items() if k != "blank"},
        mtt_viability_2,
        "Batch 2 (iTBS)"
    )
    mtt_df_all = pd.concat([mtt_df_b1, mtt_df_b2], ignore_index=True)

    mtt_viab_all = {
        "control": np.concatenate([mtt_viability["control"], mtt_viability_2["control"]]),
        "cTBS":    mtt_viability.get("cTBS", np.array([])),
        "iTBS":    mtt_viability_2.get("iTBS", np.array([]))
    }

# =========================
# ANOVA + Tukey HSD
# =========================
# Only perform ANOVA for more than 2 conditions
if SECOND_BATCH:

    def make_anova_df(data_dict, assay_name):
        rows = []
        for group, values in data_dict.items():
            for v in values:
                rows.append({"Assay": assay_name, "Group": group, "Value": v})
        return pd.DataFrame(rows)
    # Perform ANOVA
    mtt_anova_df = make_anova_df(mtt_viab_all, "MTT")
    model = ols("Value ~ Group", data=mtt_anova_df).fit()
    mtt_anova_results = sm.stats.anova_lm(model, typ=2)
    # Perform Tukey HSD
    mtt_tukey = pairwise_tukeyhsd(endog=mtt_anova_df["Value"], groups=mtt_anova_df["Group"], alpha=0.05)

    print(mtt_anova_results)
    print(mtt_tukey.summary())

# =========================
# EFFECT SIZE (Cohen's d + Hedges' g)
# =========================
pairs = list(combinations(mtt_viab_all.keys(), 2))
effect_sizes = []
# Calculate effect sizes between pairs
for a, b in pairs:
    d, g = cohen_d_hedges_g(mtt_viab_all[a], mtt_viab_all[b])
    effect_sizes.append({"Group1": a, "Group2": b, "Cohen_d": d, "Hedges_g": g})
print("MTT effect sizes:")
print(pd.DataFrame(effect_sizes))

# =============================================================================
# SIGNIFICANCE HELPERS  (matched to vesicle pool script)
# =============================================================================
def significance_label(p):
    # Significance indicator
    return "****" if p < 0.0001 else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def tukey_to_dict(tukey_result):
    tukey_dict = {}
    for row in tukey_result._results_table.data[1:]:
        grp1, grp2, meandiff, p_adj, lower, upper, reject = row
        sig = significance_label(p_adj)
        tukey_dict[(grp1, grp2)] = {"signif": sig, "p": p_adj}
    return tukey_dict

def add_significance_bracket(ax, x1, x2, y_bottom, h, text):
    # Add significance bracket to plots
    ax.plot([x1, x1, x2, x2],
            [y_bottom - h * 0.15, y_bottom, y_bottom, y_bottom - h * 0.15],
            lw=1.0, c='black', clip_on=False)
    ax.text((x1 + x2) * 0.5, y_bottom + h * 0.05, text,
            ha='center', va='bottom', fontsize=12, clip_on=False)

def get_boxplot_rendered_top(ax):
    # Find significance bracket location in boxplot
    tops = []
    for line in ax.lines:
        ydata = line.get_ydata()
        if len(ydata) > 0:
            tops.append(max(ydata))
    for collection in ax.collections:
        offsets = collection.get_offsets()
        if len(offsets) > 0:
            tops.append(max(offsets[:, 1]))
    return max(tops)

def get_barplot_rendered_top(ax):
    # Find significance bracket location in barplot
    tops = []
    for patch in ax.patches:
        tops.append(patch.get_y() + patch.get_height())
    for line in ax.lines:
        ydata = line.get_ydata()
        if len(ydata) > 0:
            tops.append(max(ydata))
    return max(tops)

def place_brackets(ax, groups, tukey_dict, data_anchor_top, step_y):
    # Plot significance brackets
    group_list = list(groups)
    def span(pair):
        return abs(group_list.index(pair[0]) - group_list.index(pair[1]))
    sorted_pairs = sorted(tukey_dict.items(), key=lambda item: span(item[0]))
    bracket_y = data_anchor_top * 1.08
    for i, ((grp1, grp2), info) in enumerate(sorted_pairs):
        x1 = group_list.index(grp1)
        x2 = group_list.index(grp2)
        add_significance_bracket(ax, x1, x2, bracket_y, step_y, info["signif"])
        bracket_y += step_y
    final_ymax = max(100, bracket_y + step_y * 1.5)
    ax.set_ylim(bottom=ax.get_ylim()[0], top=final_ymax)

# =============================================================================
# Plots
# =============================================================================
def plot_mtt(df, tukey_dict):
    group_order = ["control", "cTBS", "iTBS"]
    palette = [CONDITION_COLORS.get(g, "#aaaaaa") for g in group_order]

    batch_palette = {
        "Batch 1 (cTBS)": "#606060",
        "Batch 2 (iTBS)": "#101010",
    }

    # BOXPLOT
    fig, ax = plt.subplots(figsize=(6, 5))

    # Draw boxplot with transparent lines
    bp = sns.boxplot(x="Group", y="Value", hue="Group", data=df,
                     order=group_order,
                     palette=palette, legend=False, ax=ax,
                     showfliers=False,
                     saturation=1,
                     medianprops=dict(color="none", linewidth=0),
                     whiskerprops=dict(color="none", linewidth=0),
                     capprops=dict(color="none", linewidth=0),
                     boxprops=dict(edgecolor="none", alpha=1.0),
                     zorder=0)
    for patch in ax.patches:
        patch.set_zorder(0)
        

    # Scatter points on top of fill 
    rng = np.random.default_rng(seed=42)
    for batch, color in batch_palette.items():
        batch_data = df[df["Batch"] == batch]
        for gi, g in enumerate(group_order):
            gdata = batch_data[batch_data["Group"] == g]
            if len(gdata) == 0:
                continue
            jit = rng.uniform(-0.12, 0.12, size=len(gdata))
            ax.scatter(np.full(len(gdata), gi) + jit, gdata["Value"].values,
                       color="gray", alpha=0.6, s=22, zorder=1,
                       linewidths=0.6)

    # Re-draw boxplot lines on top of scatterplot
    sns.boxplot(x="Group", y="Value", hue="Group", data=df,
                order=group_order, legend=False, ax=ax,
                showfliers=False, showmeans=True, meanline=True,
                patch_artist=True,
                boxprops=dict(facecolor="none", edgecolor="black", linewidth=0.8, zorder=3),
                medianprops=dict(color="black", linewidth=1.5, linestyle="-", zorder=3),
                meanprops=dict(color="black", linewidth=1.5, dashes=(6, 3), zorder=3),
                whiskerprops=dict(color="black", linewidth=0.8, zorder=3),
                capprops=dict(color="black", linewidth=0.8, zorder=3),
                zorder=3)
    plt.draw()
    actual_order = [t.get_text() for t in ax.get_xticklabels()]
    data_top = get_boxplot_rendered_top(ax)
    y_range = data_top - ax.get_ylim()[0]
    step_y = y_range * 0.10
    place_brackets(ax, actual_order, tukey_dict, data_top, step_y)

    handles = [mlines.Line2D([0], [0], marker='o', color='w',
                             markerfacecolor=color, markersize=8, label=batch)
               for batch, color in batch_palette.items()]
    # Add sample size labels
    for j, g in enumerate(group_order):
        n = len(df[df["Group"] == g])
        ax.annotate(f"n={n}", xy=(j, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -32), textcoords="offset points",
                    ha="center", va="top", fontsize=12, color="gray")
    ax.set_xlabel("")
    ax.set_ylabel("Viability (%)", fontsize=18)
    ax.set_title("MTT Viability", fontsize=22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(figures_path / "MTT_boxplot_tukey.png", dpi=300)
    plt.close()

    # ---- BARPLOT ----
    fig, ax = plt.subplots(figsize=(6, 5))

    means = [df[df["Group"] == g]["Value"].mean() for g in group_order]
    sems  = [df[df["Group"] == g]["Value"].sem()  for g in group_order]
    x = np.arange(len(group_order))

    # Draw barplot with transparent lines
    ax.bar(x, means, width=0.52,
           color=palette, edgecolor="none", linewidth=0,
           zorder=0, alpha=1)

    # Scatterplot on top of fill
    rng = np.random.default_rng(seed=42)
    for gi, g in enumerate(group_order):
        gdata = df[df["Group"] == g]
        n = len(gdata)
        if n == 0:
            continue
        jit = rng.uniform(-0.2 * 0.52, 0.2 * 0.52, size=n)
        ax.scatter(np.full(n, gi) + jit, gdata["Value"].values,
                   color="gray", alpha=0.6, s=22, zorder=1, linewidths=0.8)

    # 3) Redraw barplot lines only
    ax.bar(x, means, yerr=sems, width=0.52,
           color="none", edgecolor="black", linewidth=0.8,
           capsize=5, error_kw=dict(elinewidth=1.5, ecolor="black", zorder=3),
           zorder=3)

    plt.draw()
    actual_order = [t.get_text() for t in ax.get_xticklabels()]
    data_top = max(m + s for m, s in zip(means, sems))
    y_range = data_top - ax.get_ylim()[0]
    step_y = y_range * 0.10
    place_brackets(ax, group_order, tukey_dict, data_top, step_y)

    ax.set_xticks(x)
    ax.set_xticklabels(group_order, fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Viability (%)", fontsize=18)
    ax.set_title("MTT Viability ± SEM", fontsize=22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # place sample sizes
    for j, g in enumerate(group_order):
        n = len(df[df["Group"] == g])
        ax.annotate(f"n={n}", xy=(j, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -32), textcoords="offset points",
                    ha="center", va="top", fontsize=12, color="gray")

    plt.tight_layout()
    plt.savefig(figures_path / "MTT_barplot_tukey.png", dpi=300)
    plt.close()

# =========================
# Plot plots
# =========================
if SECOND_BATCH:
    mtt_tukey_dict = tukey_to_dict(mtt_tukey)
    plot_mtt(mtt_df_all, mtt_tukey_dict)

# =========================
# Save csv
# =========================
mtt_df_b1.to_csv(base / "MTT_viability_summary.csv", index=False, sep=";")
print(f"MTT plots saved in {figures_path}")

for group, values in mtt_viab_all.items():
    print(f"{group}: {np.mean(values):.2f} ± {np.std(values, ddof=1):.2f} %")
for row in mtt_tukey._results_table.data[1:]:
    grp1, grp2, meandiff, p_adj, lower, upper, reject = row
    print(f"{grp1} vs {grp2}: p = {p_adj:.8f}")

# =========================
# Print IQR outlier MTT viability
# =========================
print("\n--- MTT Viability of IQR-detected outlier wells ---")

if outlier_records:
    for rec in outlier_records:
        group = rec["Group"]
        raw_val = rec["Value"]
        viab = raw_val / control_mean_mtt * 100
        print(f"Batch 1 | Group: {group} | Well: {rec['Well']} | Viability: {viab:.2f}%")
else:
    print("No IQR outliers in Batch 1.")

if SECOND_BATCH:
    if outlier_records_2:
        for rec in outlier_records_2:
            group = rec["Group"]
            raw_val = rec["Value"]
            viab = raw_val / control_mean_mtt_2 * 100
            print(f"Batch 2 | Group: {group} | Well: {rec['Well']} | Viability: {viab:.2f}%")
    else:
        print("No IQR outliers in Batch 2.")

# =========================
# Print all MTT viability values
# =========================
print("\n--- All MTT Viability values (pooled, including outliers) ---")

for group, wells in mtt_groups.items():
    if group == "blank":
        continue
    vals = get_wells(corr_550, wells)
    outlier_wells_b1 = [rec["Well"] for rec in outlier_records if rec["Group"] == group]
    for w, raw_val in zip(wells, vals):
        viab = raw_val / control_mean_mtt * 100
        flag = " <-- IQR outlier" if w in outlier_wells_b1 else ""
        print(f"Batch 1 | Group: {group} | Well: {w} | Viability: {viab:.2f}%{flag}")

if SECOND_BATCH:
    for group, wells in mtt_groups_2.items():
        if group == "blank":
            continue
        vals = get_wells(corr_550_2, wells)
        outlier_wells_b2 = [rec["Well"] for rec in outlier_records_2 if rec["Group"] == group]
        for w, raw_val in zip(wells, vals):
            viab = raw_val / control_mean_mtt_2 * 100
            flag = " <-- IQR outlier" if w in outlier_wells_b2 else ""
            print(f"Batch 2 | Group: {group} | Well: {w} | Viability: {viab:.2f}%{flag}")
