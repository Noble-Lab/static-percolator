"""
Search and percolator some gel bands
"""
import os
import glob
import logging
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import wispy.theme
import wispy.percolator
from targetdecoy import qvalues

# Setup -----------------------------------------------------------------------
MZML = "/net/noble/vol1/data/crux-datasets/2019basilicata-denovo/mzML"
FASTA = "/net/noble/vol1/data/crux-datasets/2019basilicata-denovo/fasta"
WEIGHTS = "../191014_scope-ms/perc-out/qc.percolator.weights.txt"

np.random.seed(78272)

# Functions -------------------------------------------------------------------
def make_index(fasta_file, name, missed_cleavages=2):
    """Create a tide-index"""
    if not os.path.isdir(name):
        cmd = ["crux", "tide-index",
               "--mods-spec", ("3M+15.99492, 3NQ+0.9840155848, 3K+42.010565, "
                               "3K+42.046950, 3K+56.026215, 3K+70.024,"
                               "3KR+14.015650"),
               "--nterm-protein-mods-spec", "1X+42.01056",
               "--enzyme", "trypsin/p",
               "--max-mods", "2",
               "--missed-cleavages", str(missed_cleavages),
               "--output-dir", "index-out",
               fasta_file, name]

        subprocess.run(cmd, check=True)

    return os.path.join(name)


def tide_search(mzml_files, name, index):
    """Search a collection of mzML files"""
    if isinstance(mzml_files, str):
        mzml_files = [mzml_files]

    out_files = (f"tide-out/{name}.tide-search.target.txt",
                 f"tide-out/{name}.tide-search.decoy.txt")

    all_exist = all([os.path.isfile(f) for f in out_files])

    if not all_exist:
        cmd = ["crux", "tide-search",
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
               "--fileroot", name,
               "--output-dir", "tide-out"]

        subprocess.run(cmd + mzml_files + [index], check=True)

    return out_files


def tide2pin (target, decoy, name):
    """Convert tide results to pin file."""
    out_file = f"pin-out/{name}.make-pin.pin"

    if not os.path.isfile(out_file):
        cmd = ["crux", "make-pin",
               "--max-charge-feature", "5",
               "--output-dir", "pin-out",
               "--fileroot", name,
               target, decoy]

        subprocess.run(cmd)

    return out_file


def percolate(pin_file, name, weights=None, static=True):
    """Percolate the pin file"""
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

        if weights is not None and static:
            cmd += ["--init-weights", weights,
                    "--override", "T",
                    "--static", "T"]
        elif weights is not None and not static:
            cmd += ["--init-weights", weights,
                    "--override", "T",
                    "--static", "F"]

        print(" ".join(cmd + [pin_file]))
        subprocess.run(cmd + [pin_file], check=True)

    return out_files


def assign_confidence(target_file, name):
    """Run TDC on each tide result"""
    out_file = f"conf-out/{name}.assign-confidence.target.txt"
    if not os.path.isfile(out_file):
        cmd = ["crux", "assign-confidence",
               "--output-dir", "conf-out",
               "--fileroot", name,
               target_file]

        subprocess.run(cmd, check=True)

    return out_file


def read_perc(psm_file):
    """Read a Percolator tab-delimited file"""
    psms_df = wispy.percolator.read(psm_file)
    psms_df["name"] = os.path.basename(psm_file.replace(".percolator.*$", ""))

    return psms_df.apply(pd.to_numeric, errors="ignore")


def plot_figures(res_df, gain_df, ann_points, thold, res=300):
    """Plot all the figures!"""
    if not os.path.isdir("figures"):
        os.mkdir("figures")

    fig, axs = plt.subplots(1, 3, figsize=(7, 2.25))
    ax = axs[0]
    ax.axhline(0, color="black", linestyle="dashed", linewidth=1, zorder=0)
    ax.scatter(gain_df.conf, gain_df.gain1 * 100, label="1% FDR", s=10,
               edgecolor="white", linewidth=0.25)
    #ax.scatter(gain_df.conf, gain_df.gain2 * 100, label="5% FDR", s=10)

    point_locs = ((2100, 23), (2500, 3))
    for txt_loc, point, lab in zip(point_locs, ann_points, ("B", "C")):
        loc1 = (point.conf.values[0], point.gain1.values[0]*100)
        loc2 = (point.conf.values[0], point.gain2.values[0]*100)
        arrows = {"arrowstyle": "-", "edgecolor": "black", "linewidth": 1,
                  "shrinkA": 1, "shrinkB": 3, "relpos": (0, 0.5)}
        ax.annotate(lab, loc1, xytext=txt_loc, ha="left", va="center",
                    size="small", weight="semibold", arrowprops=arrows)
     #   ax.annotate("", loc1, xytext=txt_loc, arrowprops=arrows)
     #   ax.annotate("", loc2, xytext=txt_loc, arrowprops=arrows)

    ax.text(-0.1, 1.15, "A", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="right")
    #ax.legend(frameon=False, handletextpad=0)
    ax.set_xlabel("Confident PSMs (Tide 1% FDR)")
    ax.set_ylabel("Peptides Gained by\nStatic Model (%)")

    res_df["model_label"] = res_df.model.str.title()
    for samp, axlab, ax in zip(ann_points, ("B", "C"), axs[1:]):
        dat = res_df[res_df.samp == samp.samp.values[0]]
        ax.axvline(0.01, color="black", linewidth=1, linestyle="dashed")
        ax.text(-0.1, 1.15, axlab, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="right")
        dat = dat.sort_values("model_label", ascending=False)
        z = 2
        for lab, mod in dat.groupby("model_label", sort=False):
            qvalues.plot(mod["percolator q-value"], ax=ax, label=lab, zorder=z)
            z -= 1

        ax.set_ylabel("Accepted Peptides")
        ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig("figures/gain.png", dpi=res)

    return thold

# MAIN ------------------------------------------------------------------------
def main():
    """The main function"""
    logging.basicConfig(level=logging.INFO)
    palette = wispy.theme.paper()

    # Make index
    tide_idx = make_index(FASTA, "human.index")

    # Search and Percolate training files
    train_name = "train"
    train_files = (glob.glob(os.path.join(MZML, "*I-gb*.mzML.gz")) +
		           glob.glob(os.path.join(MZML, "*H-gb*.mzML.gz")))

    train_tide = tide_search(train_files, train_name, tide_idx)
    train_pin = tide2pin(*train_tide, train_name)
    train_out = percolate(train_pin, train_name)
    weights = os.path.join("perc-out", "train.percolator.weights.txt")


    # Search files and percolate
    samps = []
    normal = []
    static = []
    tide = []

    test_files = [os.path.join(MZML, f) for f in os.listdir(MZML)
                  if f.endswith(".mzML.gz")]
    test_files = [f for f in test_files if f not in train_files]

    for mz_file in test_files:
        name = os.path.basename(mz_file.replace(".mzML.gz", ""))
        tide_res = tide_search(mz_file, name, tide_idx)
        pin = tide2pin(*tide_res, name)
        normal.append(percolate(pin, name + ".normal"))
        #static.append(percolate(pin, name + ".other", weights=WEIGHTS))
        static.append(percolate(pin, name + ".static", weights=weights))
        tide_conf = read_perc(assign_confidence(tide_res[0], name))
        tide.append((tide_conf["tdc q-value"] <= 0.01).sum())
        samps.append(name)

    # Aggregate results
    normal_pep = pd.concat([read_perc(f[2]) for f in normal])
    normal_pep["model"] = "dynamic"
    static_pep = pd.concat([read_perc(f[2]) for f in static])
    static_pep["model"] = "static"

    peptides = pd.concat([static_pep, normal_pep]).reset_index(drop=True)
    peptides["samp"] = peptides.name.str.replace(r"\..+$", "")

    gain_df = []
    for samp, conf in zip(samps, tide):
        peps = peptides[peptides.samp == samp]
        df = pd.DataFrame({"samp": [samp], "conf": conf})

        for n, thold in enumerate((0.01, 0.05)):
            passing = peps[peps["percolator q-value"] <= thold]
            num_static = (passing.model == "static").sum()
            num_default = (passing.model == "dynamic").sum()
            gain = (num_static - num_default) / num_default
            df[f"gain{n+1}"] = gain

        gain_df.append(df)


    gain_df = pd.concat(gain_df)
    worst = gain_df[gain_df.gain1 == gain_df.gain1.min()]
    best = gain_df[gain_df.gain1 == gain_df.gain1.max()]

    ret = plot_figures(peptides, gain_df, (best, worst), 0.01)

    return gain_df, peptides, best, worst

if __name__ == "__main__":
    x = main()
    a = x[1][(x[1].samp == "MSB27866A") & (x[1]["percolator q-value"] <= 0.1)]
    print(x[0])
    pos = (x[0].gain1 > 0).sum()
    print(f"Number with more peptides at 1% FDR: {pos}")
    print(f"Total number of runs: {len(x[0])}")
    print(f"Maximum gain {x[0].gain1.max()}")
    print(f"Minimum gain {x[0].gain1.min()}")
