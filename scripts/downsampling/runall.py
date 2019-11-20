"""
This module tests the effects of sample size on Percolator results.
"""
import os
import re
import glob
import gzip
import random
import logging
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import wispy.percolator
import wispy.theme

# Setup -----------------------------------------------------------------------
random.seed(42)
np.random.seed(651)

SOURCE_FILE = "../../../../2018_wfondrie_msxl/data/pin/kim.pin.gz"
NUM_LINES = 23330312
TRAIN_SAMPLE = int(1e6)
TEST_SAMPLE = 100000
PERC_SEEDS = [541, 7, 19843, 862, 33]
HICONF = 0.01
HICONF_FRAC = 0.4
HICONF_TOTAL = 100000

# Functions -------------------------------------------------------------------
def count_lines(pin_file):
    """Count lines in the gzipped pin file"""
    logging.info("Counting Lines in PIN file...")
    with gzip.open(pin_file) as kimfile:
        num_lines = sum(1 for _ in tqdm(kimfile, ascii=True))

    return num_lines


def make_splits(pin_file, n_train, n_test, num_lines):
    """Sample n PSMs for each."""
    df_out = ["df-out/train.pkl", "df-out/test.pkl"]

    if all([os.path.isfile(f) for f in df_out]):
        logging.info("Loading PIN DataFrames...")
        train_df = pd.read_pickle(df_out[0])
        test_df = pd.read_pickle(df_out[1])
        return train_df, test_df

    # Else...
    logging.info("Sampling PIN file...")
    n_total = n_train + n_test
    if n_total >= num_lines:
        raise ValueError("Total number of lines exceeded by splits")

    lines = np.array(np.arange(num_lines-1))
    lines = np.random.choice(lines, size=n_total, replace=False)
    lines = [set(lines[:n_train]), set(lines[n_train:])]

    train_df = []
    test_df = []
    append_train = train_df.append
    append_test = test_df.append
    with gzip.open(pin_file, "rb") as pin:
        header = pin.readline().decode("utf-8").replace("\n", "").split("\t")
        for idx, line in tqdm(enumerate(pin), ascii=True):
            if idx in lines[0]:
                append_train(read_line(line, len(header)-1))
            elif idx in lines[1]:
                append_test(read_line(line, len(header)-1))

    train_df = pd.DataFrame(columns=header, data=train_df)
    test_df = pd.DataFrame(columns=header, data=test_df)
    train_df.to_pickle(df_out[0])
    test_df.to_pickle(df_out[1])

    return train_df, test_df


def read_line(line, nbreaks):
    """Read a pin file line"""
    return line.decode("utf-8").replace("\n", "").split("\t", nbreaks)


def write_pin(pin_df, pin_file):
    """Write a dataframe to pin file format"""
    if os.path.isfile(pin_file):
        logging.info("%s already exists. Skipping...", pin_file)
        return pin_file

    return wispy.percolator.write_pin(pin_df, pin_file)


def prepare_dirs():
    """Prepare the working directory"""
    dirs = ("figures", "pin-out", "perc-out", "df-out")
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)


def percolate(pin_file, name, seed=1, weights=None):
    """Percolate the pin file"""
    suffixes = [".target.psms.txt", ".decoy.psms.txt",
                ".target.peptides.txt", ".decoy.peptides.txt",
                ".weights.txt"]

    out_files = [os.path.join("perc-out", name + ".percolator" + s)
                 for s in suffixes]

    all_exist = all([os.path.isfile(f) for f in out_files])
    if not all_exist:
        cmd = ["crux", "percolator",
               "--output-weights", "T",
               "--fileroot", name,
               "--tdc", "T",
               "--output-dir", "perc-out",
               "--percolator-seed", str(seed)]

        if weights is not None:
            cmd += ["--init-weights", weights, "--static", "T"]

        subprocess.run(cmd + [pin_file])

    return out_files


def read_perc(psm_file):
    """Read a Percolator tab-delimited file"""
    psms_df = wispy.percolator.read(psm_file)
    file_base = os.path.basename(psm_file.split(".")[0])
    psms_df["name"] = file_base
    psms_df["major"] = "_major_" in file_base
    psms_df["samp"] = int(re.search("samp(.+?)_", file_base).group(1))
    psms_df["seed"] = re.search("seed(.)", file_base).group(1)
    return psms_df


def widen(pin_df, values="percolator q-value"):
    """Widen a concatenated pin dataframe"""
    groups = ["scan", "sequence", "samp", "seed"]
    wide = pin_df.groupby(groups)[values] \
                 .sum() \
                 .unstack("seed") \
                 .reset_index()

    return wide


def count_psms(wide_df, thold=0.01):
    """Count the union and intersection of PSMs for each samp"""
    count_df = []
    wide_df = wide_df.drop(columns=["scan", "sequence"])
    print(wide_df.columns)
    for samp, group in wide_df.groupby("samp"):
        group = group.drop(columns=["samp"])
        num_seeds = group.shape[1]
        num_pass = (group <= thold).sum(axis=1)
        df = pd.DataFrame({"Intersection": [(num_pass == num_seeds).sum()],
                           "Union": [(num_pass > 0).sum()]})
        df["samp"] = samp
        count_df.append(df)

    return pd.concat(count_df)


def count_passing(df, thold=0.01):
    """Count the number of passing psms"""
    df["passing"] = df["percolator q-value"] <= thold
    ret_df = df.groupby(["samp", "seed"]).passing.sum()
    return ret_df.reset_index()


def make_slide_figures(all_df, conf_df, res=100):
    """Make cool plots for slides"""
    wispy.theme.talk()
    # Plot seed vs seed q-value curves ----------------------------------------
    wide_all = widen(all_df[all_df.major])
    wide_conf = widen(conf_df[conf_df.major])

    fig, axs = plt.subplots(1, 4, figsize=(19, 5), sharey=True)
    plot_btw_qvalues(wide_all, axs.flatten(), size=50, lw=4)
    plt.savefig("figures/all_qvalue_talk.png", dpi=res)

    fig, axs = plt.subplots(1, 4, figsize=(19, 5), sharey=True)
    plot_btw_qvalues(wide_conf, axs.flatten(), size=50, lw=4)
    plt.savefig("figures/conf_qvalue_talk.png", dpi=res)

    # Plot the number of passing PSMs -----------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(19, 8))
    passing_all = count_passing(all_df)
    plot_psms(passing_all, axs[0])
    axs[0].set_xlabel("Total Training PSMs")
    passing_conf = count_passing(conf_df)
    plot_psms(passing_conf, axs[1])
    axs[1].set_xlabel("Confident Training PSMs")
    plt.tight_layout()
    plt.savefig("figures/passing-psms_talk.png", dpi=res)

    return passing_all, passing_conf


def make_figures(all_df, conf_df, res=300):
    """Make cool plots"""
    # Plot seed vs seed q-value curves ----------------------------------------
    qlabels = [["A", "B", "C", "D"], ["E", "F", "G", "H"]]

    wide_all = widen(all_df[all_df.major])
    wide_conf = widen(conf_df[conf_df.major])

    fig, axs = plt.subplots(2, 4, figsize=(7, 4), sharey=True)
    plot_qvalues(wide_all, axs[0, :].flatten(), qlabels[0])
    plot_qvalues(wide_conf, axs[1, :].flatten(), qlabels[1])
    plt.savefig("figures/seeds_qvalue_pairs.png", dpi=res)

    fig, axs = plt.subplots(2, 4, figsize=(7, 4), sharey=True)
    plot_btw_qvalues(wide_all, axs[0, :].flatten(), qlabels[0])
    plot_btw_qvalues(wide_conf, axs[1, :].flatten(), qlabels[1])
    plt.savefig("figures/sample_qvalue_pairs.png", dpi=res)

    # Plot the number of passing PSMs -----------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(3.33, 4.5))
    #fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))
    passing_all = count_passing(all_df)
    plot_psms(passing_all, axs[0])
    axs[0].set_xlabel("Total Training PSMs")
    axs[0].text(-0.1, 1.15, "A", transform=axs[0].transAxes,
                fontsize=12, fontweight="bold", va='top', ha='right')

    passing_conf = count_passing(conf_df)
    plot_psms(passing_conf, axs[1])
    axs[1].set_xlabel("Confident Training PSMs")
    axs[1].text(-0.1, 1.15, "B", transform=axs[1].transAxes,
                fontsize=12, fontweight="bold", va='top', ha='right')

    plt.tight_layout()
    plt.savefig("figures/passing-psms.png", dpi=res)

    return passing_all, passing_conf


def plot_psms(df, ax, rev=True):
    """Plot the number of PSMs accepted"""
    ax.set(xscale="log")
    p = sns.lineplot(x=df.samp, y=df.passing, ci=95, marker="o", ax=ax)
    ax.set_ylabel("Accepted Test PSMs")
    plt.tight_layout()

    if rev:
        ax.invert_xaxis()


def plot_qvalues(df, axs, labels):
    """Plot q-values from two seeds against each other"""
    labels = list(labels)
    labels.reverse()
    axs = np.flip(axs)
    for i, group, ax in zip(range(len(axs)), df.groupby("samp"), axs):
        samp = group[0]
        samp_df = group[1]
        ax.axhline(0.01, color="black", linewidth=1, linestyle="dashed")
        ax.axvline(0.01, color="black", linewidth=1, linestyle="dashed")
        ax.scatter(samp_df["1"].values, samp_df["2"].values, alpha=0.01, s=10, marker=".")
        ax.set_xlabel("q-value (Seed 1)")

        if i == len(axs) - 1:
            ax.set_ylabel("q-value (Seed 2)")

        ax.set_xlim(-0.01, 0.11)
        ax.set_ylim(-0.01, 0.11)
        ax.set_aspect("equal")
        ax.text(-0.1, 1.15, labels[i], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va='top', ha='right')

    plt.tight_layout()
    return axs


def plot_btw_qvalues(df, axs, labels=None, size=10, lw=1):
    """Plot q-values between samples against each other."""
    if labels is not None:
        labels = list(labels)
        labels.reverse()

    axs = np.flip(axs)
    max_num = df.samp.max()
    y = df["3"][df.samp == max_num].values

    for i, group, ax in zip(range(len(axs)), df.groupby("samp"), axs):
        samp = group[0]
        samp_df = group[1]

        ax.axhline(0.01, color="black", linewidth=lw, linestyle="dashed")
        ax.axvline(0.01, color="black", linewidth=lw, linestyle="dashed")
        ax.scatter(samp_df["4"].values, y, alpha=0.05, s=size, marker=".")
        ax.set_xlabel(format_label(samp))

        if i == len(axs) - 1:
            ax.set_ylabel(format_label(max_num))

        ax.set_xlim(-0.01, 0.11)
        ax.set_ylim(-0.01, 0.11)
        ax.set_aspect("equal")

        if labels is not None:
            ax.text(-0.1, 1.15, labels[i], transform=ax.transAxes,
                    fontsize=12, fontweight="bold", va="top", ha="right")

    plt.tight_layout()
    return axs


def format_label(num):
    """Convert to a compact scientific notation"""
    lab = f"{num:.0e}"
    lab = lab.replace("e+0", "e")
    return f"q-value ({lab} PSMs)"


# MAIN ------------------------------------------------------------------------
def main():
    """The main function"""
    logging.basicConfig(level=logging.INFO)
    palette = wispy.theme.paper()
    prepare_dirs()

    out_dfs = ["df-out/overall_res.pkl",
               "df-out/conf_res.pkl"]

    if all([os.path.isfile(f) for f in out_dfs]):
        logging.info("Loading saved results...")
        return [pd.read_pickle(f) for f in out_dfs]

    if NUM_LINES is None:
        num_lines = count_lines(SOURCE_FILE)
    else:
        num_lines = NUM_LINES

    # Sample pin
    train_df, test_df = make_splits(SOURCE_FILE, TRAIN_SAMPLE, TEST_SAMPLE,
                                    num_lines)

    # Save train and test PIN files.
    train_pin = write_pin(train_df, "pin-out/train.pin")
    test_pin = write_pin(test_df, "pin-out/test.pin")

    # Downsample overall PSMs -------------------------------------------------
    overall_res = []
    total_major = [int(1*10**i) for i in range(2, 6)]
    total_minor = [int(1*10**i) for i in (3.5, 2.5, 6)]
    total_type = ["major"]*len(total_major) + ["minor"]*len(total_minor)

    for num, mtype in zip(total_major + total_minor, total_type):
        out_root = f"_samp{num}_{mtype}"
        pin_out = f"pin-out/train{out_root}.pin"

        if num != int(1e6):
            if not os.path.isfile(pin_out):
                small_df = train_df.sample(n=num)
                small_pin = write_pin(small_df, pin_out)
            else:
                small_pin = pin_out
        else:
            small_pin = train_pin

        for i, s in enumerate(PERC_SEEDS):
            res_files = percolate(small_pin, f"train{out_root}_seed{i+1}",
                                  seed=s)
            res = percolate(test_pin, f"test{out_root}_seed{i+1}",
                            weights=res_files[4])
            overall_res.append(read_perc(res[0]))

    overall_res = pd.concat(overall_res)

    # Downsample High Quality PSMs --------------------------------------------
    perc_res = read_perc("perc-out/train_samp1000000_minor_seed1.percolator."
                         "target.psms.txt")
    expmass = (perc_res["spectrum neutral mass"] + 1.007276).round(4).astype(str)
    perc_res = perc_res.set_index(perc_res["scan"].astype(str) + "_" + expmass)
    perc_res = perc_res[perc_res["percolator q-value"] <= HICONF]

    train_expmass = train_df["ExpMass"].astype(float).round(4).astype(str)
    train_df = train_df.set_index(train_df["ScanNr"].astype(str) + "_"
                                  + train_expmass)

    # The confident and unconfident psms
    conf_psms = train_df.loc[perc_res.index, :]
    other_psms = train_df.drop(index=perc_res.index)

    conf_res = []
    conf_major = [int((HICONF_TOTAL * HICONF_FRAC) / 10**i) for i in range(4)]
    conf_minor = [int((HICONF_TOTAL * HICONF_FRAC) / 10**i) for i
                  in (1.5, 2.5, 3.5, 4)]

    conf_type = ["major"]*len(conf_major) + ["minor"]*len(conf_minor)

    for num, mtype in zip(conf_major + conf_minor, conf_type):
        out_root = f"_conf_samp{num}_{mtype}"
        pin_out = f"pin-out/train{out_root}.pin"
        num_other = HICONF_TOTAL - num
        if not os.path.isfile(pin_out):
            conf = conf_psms.sample(n=int(num))
            other = other_psms.sample(n=int(num_other))
            small_df = pd.concat([conf, other])
            small_pin = write_pin(small_df, pin_out)
        else:
            small_pin = pin_out

        for i, s in enumerate(PERC_SEEDS):
            logging.info("Samp %i, seed %i: %i confident PSMs, %i total PSMs",
                         num, i, num, HICONF_TOTAL)
            res_files = percolate(small_pin, f"train{out_root}_seed{i+1}",
                                  seed=s)
            res = percolate(test_pin, f"test{out_root}_seed{i+1}",
                            weights=res_files[4])
            conf_res.append(read_perc(res[0]))

    conf_res = pd.concat(conf_res)

    overall_res.to_pickle("df-out/overall_res.pkl")
    conf_res.to_pickle("df-out/conf_res.pkl")
    return overall_res, conf_res


if __name__ == "__main__":
    overall, conf = main()
    logging.info("Making Figures...")
    x = make_figures(overall, conf)
    make_slide_figures(overall, conf)
    print(x[1])
