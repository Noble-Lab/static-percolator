"""
Search and run Percolator on the single cell data.
"""
import os
import sys
import glob
import logging
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import seaborn as sns
from tqdm import tqdm
from targetdecoy import qvalues

import wispy.theme
import wispy.percolator

# Setup -----------------------------------------------------------------------
MZML = "/scratch/wfondrie/2019specht-high/mzml"
FASTA = "../../../data/fasta/human_swissprot_2019-09.fasta"

np.random.seed(2815)

# Functions -------------------------------------------------------------------
def set_figure_style(fg_color, accent):
    "Set the figure style"
    matplotlib.font_manager.findSystemFonts()
    style = {"axes.edgecolor": fg_color,
             "axes.labelcolor": fg_color,
             "text.color": fg_color,
             "xtick.color": fg_color,
             "ytick.color": fg_color,
             "font.family": "sans-serif",
             "font.sans-serif": ["Roboto"]}

    palette = sns.color_palette([accent, fg_color])

    plt.rcParams["legend.frameon"] = False
    sns.set_context("poster", rc={"lines.linewidth": 4})
    sns.set_palette(palette)
    sns.set_style("ticks", style)


def read_perc(psm_file):
    """Read a Percolator tab-delimited file"""
    psms_df = wispy.percolator.read(psm_file)
    psms_df["name"] = os.path.basename(psm_file.replace(".percolator.*$", ""))

    return psms_df.apply(pd.to_numeric, errors="ignore")


def make_index(fasta_file, name, missed_cleavages=2):
    """
    Create a tide-index for the fasta file.

    This attempts to replicate the original MaxQuant search settings.
    """
    if not os.path.isdir(name):
        cmd = ["crux", "tide-index",
               "--mods-spec", "C+0,K+229.162932,3M+15.99492,3N+0.9840155848",
               "--nterm-peptide-mods-spec", "X+229.162932",
               "--nterm-protein-mods-spec", "1X+42.01056",
               "--enzyme", "trypsin/p",
               "--missed-cleavages", str(missed_cleavages),
               "--output-dir", "index-out",
               fasta_file, name]

        subprocess.run(cmd)

    return os.path.join(name)


def tide_search(mzml_files, name, index):
    """Search a collection of mzML files."""
    if isinstance(mzml_files, str):
        mzml_files = [mzml_files]

    out_files = (f"tide-out/{name}.tide-search.target.txt",
                 f"tide-out/{name}.tide-search.decoy.txt")

    all_exist = all([os.path.isfile(f) for f in out_files])

    if not all_exist:
        cmd = ["crux", "tide-search",
               # Added 10/11/19
               "--remove-precursor-peak", "T",
               "--auto-precursor-window", "warn",
               "--isotope-error", "1",
               "--precursor-window", "50",
               "--precursor-window-type", "ppm",
               "--fragment-tolerance", "0.02",
               "--mz-bin-width", "1.0005079",
               "--compute-sp", "T",
               "--exact-p-value", "T",
               "--score-function", "both",
               #---------------------
               "--fileroot", name,
               "--output-dir", "tide-out"]

        subprocess.run(cmd + mzml_files + [index])

    return out_files


def tide2pin(target, decoy, name):
    """Convert tide results to a pin file."""
    out_file = f"pin-out/{name}.make-pin.pin"

    if not os.path.isfile(out_file):
        cmd = ["crux", "make-pin",
               "--max-charge-feature", "5",
               "--output-dir", "pin-out",
               "--fileroot", name,
               target, decoy]

        subprocess.run(cmd)

    return out_file


def percolate(pin_file, name, weights=None):
    """Percolate the pin file."""
    suffixes = [".target.psms.txt",
                ".decoy.psms.txt",
                ".target.peptides.txt",
                ".decoy.peptides.txt"]

    out_files = [os.path.join("perc-out", name + ".percolator" + s)
                 for s in suffixes]

    all_exist = all([os.path.isfile(f) for f in out_files])
    if not all_exist:
        cmd = ["crux", "percolator",
               "--output-weights", "T",
               "--fileroot", name,
               "--tdc", "T",
               "--output-dir", "perc-out"]

        if weights is not None:
            cmd += ["--init-weights", weights, "--static", "T"]

        print(" ".join(cmd + [pin_file]))
        subprocess.run(cmd + [pin_file])

    return out_files


def plot_point_diff(df, ax=None, width=0.5, size=15):
    """Plot the point difference plot suggested by Jacob"""
    if ax is None:
        ax = plt.gca()

    for i, group_df in enumerate(df.groupby("model")):
        if not i:
            yend = group_df[1].sequence
            yend.index = group_df[1].sample_idx.values
        else:
            ystart = group_df[1].sequence
            ystart.index = group_df[1].sample_idx.values

        ax.scatter(group_df[1].sample_idx, group_df[1].sequence,
                   label=group_df[0], zorder=3-i, s=size)

    boxes = []
    rect_df = pd.DataFrame({"ystart": ystart, "yend": yend})
    rect_df["diff"] = rect_df.ystart - rect_df.yend
    for i, row in rect_df.iterrows():
        boxes.append(Rectangle((float(i) - width/2, row.yend),
                               width, row["diff"]))

    pc = PatchCollection(boxes, facecolors="grey", zorder=1)
    ax.add_collection(pc)
    return ax


def match_num(num_df, res_df):
    """Threshold at the same number of peptides."""
    out_df = []
    res_df = res_df.sort_values(by="percolator score", ascending=False)
    for _, mod_group in res_df.groupby("model"):
        for samp, vals in mod_group.groupby("samp"):
            samp_df = vals.copy().reset_index()
            samp_df["discard"] = samp_df.index >= num_df.loc[samp, :].static
            out_df.append(samp_df)

    return pd.concat(out_df).reset_index(drop=True)


def count_reps(df, col="percolator q-value", thold=0.01):
    """Count the minimum number of replicates a peptide appears"""
    wide_df = pd.pivot_table(df,
                             index=["sequence", "model"],
                             columns="samp",
                             values=col)

    num_reps = (wide_df <= thold).sum(axis=1)
    num_reps.name = "reps"
    num_reps = num_reps.reset_index()
    num_reps = num_reps.groupby(["model", "reps"]).size()
    num_reps = num_reps.iloc[::-1].groupby(["model"]).cumsum()
    num_reps = num_reps.to_frame() \
                       .reset_index() \
                       .pivot(index="reps", columns="model", values=0) \
                       .reset_index()

    return num_reps[num_reps.reps > 0]


def plot_figures(res_df, psm_df, thold, res=300):
    """Make some plots"""
    if not os.path.isdir("figures"):
        os.mkdir("figures")

    # Plot the number of PSMs below thold for each:
    below_thold = res_df[res_df["percolator q-value"] <= thold]
    num_df = below_thold.groupby(["samp","model"]) \
                        .sequence \
                        .count() \
                        .sort_values(ascending=False) \
                        .reset_index()

    pep_num = num_df.pivot(index="samp", columns="model", values="sequence")

    psm_num = psm_df[psm_df["percolator q-value"] <= thold] \
        .groupby(["samp", "model"]) \
        .psm_id.count() \
        .reset_index() \
        .pivot(index="samp", columns="model", values="psm_id")

    psm_num["gain"] = 100*(psm_num.static - psm_num.dynamic) / psm_num.dynamic
    psm_num["level"] = "PSMs"
    pep_num["gain"] = 100*(pep_num.static - pep_num.dynamic) / pep_num.dynamic
    pep_num["level"] = "Peptides"
    nums = pd.concat([psm_num, pep_num])

    # Plot the number of peptides shared between files
    match_df = match_num(pep_num, res_df)
    match_df.set_index(["sequence", "model"])
    res_df.set_index(["sequence", "model"])

    res_reps = count_reps(res_df, thold=thold)
    match_reps = count_reps(match_df, col="discard", thold=0.5)

    models = np.unique(res_df.model).tolist()
    res_diff = res_reps[models[1]].values - res_reps[models[0]].values
    match_diff = res_reps[models[1]].values - match_reps[models[0]].values

    fig, axs = plt.subplots(2, 1, figsize=(3.33, 5))
    ax = axs[0]
    sns.violinplot(nums.level, nums.gain, inner=None, linewidth=0, ax=ax)
    sns.swarmplot(nums.level, nums.gain, size=4, color="black", ax=ax)
    ax.axhline(0, color="black", linewidth=1, linestyle="dashed")
    ax.set_xlabel("")
    ax.set_ylabel("Gain by Static Model (%)\n ")
    ax.text(-0.1, 1.15, "A", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="right")

    ax = axs[1]
    ax.axhline(0, color="black", linewidth=1, linestyle="dashed")
    ax.plot(res_reps.reps, res_diff)
    ax.set_xlabel("Minimum Number of\nExperiments Detected")
    ax.set_ylabel("Peptides Gained by\nStatic Model")
    ax.text(-0.1, 1.15, "B", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="right")
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig("figures/rep_numbers.png", dpi=res)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(7, 2.75))
    ax = axs[0]
    ax.plot(res_reps.reps, res_reps.static, label="Static Model", zorder=3)
    ax.plot(res_reps.reps, res_reps.dynamic, label="Dynamic Model", zorder=1)
    ax.plot(match_reps.reps, match_reps.dynamic, label="Dynamic Model w/\nStatic Thresholds",zorder=2)
    ax.legend()
    ax.set_ylabel("Number of Peptides")
    ax.set_yscale("log")
    ax.set_xlabel("Minimum Number of Experiments Detected")
    ax.text(-0.1, 1.15, "A", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="right")

    ax = axs[1]
    ax.axhline(0, color="black", linewidth=1, linestyle="dashed")
    ax.plot(match_reps.reps, match_diff)
    ax.set_ylabel("Peptides Gained by\nStatic Model")
    ax.set_xlabel("Minimum Number of Experiments Detected")
    ax.text(-0.1, 1.15, "B", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="right")

    plt.tight_layout()
    plt.savefig("figures/scope_supp.png", dpi=res)
    plt.close()

    return nums, pd.DataFrame({"reps": res_reps.reps, "diff": res_diff})


# MAIN ------------------------------------------------------------------------
def main():
    """The main function"""
    logging.basicConfig(level=logging.INFO)
    #set_figure_style(FG_COLOR, ACCENT)
    palette = wispy.theme.paper()
    plt.rcParams["legend.frameon"] = False


    # Make index
    tide_idx = make_index(FASTA, "human.index")

    # Analyze QC files
    qc_name = "qc"
    qc_files = glob.glob(os.path.join(MZML, "*_QC_*.mzML.gz"))
    qc_tide = tide_search(qc_files, qc_name, tide_idx)
    qc_pin = tide2pin(*qc_tide, name=qc_name)
    qc_out = percolate(qc_pin, qc_name)
    weights = os.path.join("perc-out", "qc.percolator.weights.txt")

    # Analyze the SC files
    x_files = glob.glob(os.path.join(MZML, "*_X_*.mzML.gz"))

    # Two file are very small with no PSMs with q<0.01...
    small_files = ["190222S_LCA9_X_FP94BD.mzML.gz",
                   "190228S_LCA9_X_FP94BE.mzML.gz"]
    x_files = [f for f in x_files if os.path.basename(f) not in small_files]

    normal = []
    static = []
    for x_file in x_files:
        x_name = os.path.basename(x_file.replace(".mzML.gz", ""))
        x_tide = tide_search(x_file, x_name, tide_idx)
        x_pin = tide2pin(*x_tide, name=x_name)
        normal.append(percolate(x_pin, x_name + ".normal"))
        static.append(percolate(x_pin, x_name + ".static", weights=weights))

    # Aggregate Results
    normal_psms = pd.concat([read_perc(f[0]) for f in normal])
    normal_pep = pd.concat([read_perc(f[2]) for f in normal])
    normal_psms["model"] = "dynamic"
    normal_pep["model"] = "dynamic"

    static_psms = pd.concat([read_perc(f[0]) for f in static])
    static_pep = pd.concat([read_perc(f[2]) for f in static])
    static_psms["model"] = "static"
    static_pep["model"] = "static"

    psms = pd.concat([static_psms, normal_psms]).reset_index(drop=True)
    psms["samp"] = psms.name.str.replace(r"\..+$", "")
    psms["psm_id"] = psms.samp + "_" + psms.scan.astype(str)

    peptides = pd.concat([static_pep, normal_pep]).reset_index(drop=True)
    peptides["samp"] = peptides.name.str.replace(r"\..+$", "")

    # Plot results
    ret = plot_figures(peptides, psms, 0.01)

    return ret


if __name__ == "__main__":
    gains, diffs = main()
    psms = gains[gains.level == "PSMs"]
    print(f"Max PSM gain: {psms.gain.max()}")
    print(f"Min PSM gain: {psms.gain.min()}")
    pos = (psms.gain > 0).sum()
    print(f"{pos} runs gained PSMs out of {len(psms)} ({pos/len(psms):.2f})")

    peps = gains[gains.level == "Peptides"]
    print(f"Max peptide gain: {peps.gain.max()}")
    print(f"Min peptide gain: {peps.gain.min()}")
    pos = (peps.gain > 0).sum()
    print(f"{pos} runs gained peptides out of {len(peps)} ({pos/len(peps):.2f})")
