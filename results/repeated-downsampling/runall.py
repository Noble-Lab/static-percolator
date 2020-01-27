"""
This module tests the effects of sample size on Percolator results.
"""
import os
import re
import gzip
import pickle
import logging
import subprocess
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import wispy.percolator
import wispy.theme

# Setup -----------------------------------------------------------------------
np.random.seed(42)

NUM_LINES = 23330312
SOURCE_FILE = os.path.join("..", "..", "data", "pin", "kim.pin.gz")
TEST_SAMPLE = 100000
HICONF = 0.01
HICONF_FRAC = 0.4
HICONF_TOTAL = 100000
TIMES = 10

# Seeds for the individual samples so I can cache results.
SEED1, SEED2, SEED3 = np.random.randint(1000, size=3)

# Functions -------------------------------------------------------------------
def prepare_dirs():
    """Prepare the working directory"""
    dirs = ("figures", "pin-out", "perc-out", "df-out", "pkl-out")
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)


def count_lines(pin_file):
    """Count lines in the gzipped pin file"""
    logging.info("Counting Lines in PIN file...")
    with gzip.open(pin_file) as kimfile:
        num_lines = sum(1 for _ in tqdm(kimfile, ascii=True))

    return num_lines


def split_psms(total_size, test_size):
    """Split PSMs into a train and test set."""
    logging.info("Splitting out Test set...")
    np.random.seed(SEED1)
    all_rows = np.arange(total_size)
    np.random.shuffle(all_rows)
    test = all_rows[:test_size]
    train = all_rows[test_size:]

    return test, train


def rows_to_pin(rows, pin_file, out_file):
    """Write the specified rows of 'pin_file' to a new PIN file."""
    if os.path.isfile(out_file):
        return out_file

    rows = set(rows)
    with gzip.open(pin_file, "r") as pin:
        new_pin = [pin.readline()]
        new_pin += [r for i, r in tqdm(enumerate(pin)) if i in rows]

    with open(out_file, "wb+") as pin_out:
        pin_out.writelines(new_pin)

    return out_file


def sample_psms(num, times, psms, pin_file, suffix):
    """Sample PSMs"""
    out_files = [os.path.join("pin-out", f"train_samp{num}_rep{i}{suffix}.pin")
                 for i in range(times)]

    all_exist = all([os.path.isfile(f) for f in out_files])
    if all_exist:
        logging.info("All files for num=%i, times=%i are present. Skipping...",
                     num, times)
        return out_files

    psms = np.random.permutation(psms)[:num*times] # + 1
    assignments = np.repeat(np.arange(times), num)

    # create row dictionary
    psm_dict = {r: a for r, a in zip(psms, assignments)}

    # initialize row lists
    samp_list = [[] for _ in range(times)]

    logging.info("Sampling %i PSMs, %i times...", num, times)
    # Read through PIN, appending selected rows to each list.
    with gzip.open(pin_file, "r") as pin:
        header = pin.readline()
        for samp in samp_list:
            samp.append(header)

        for idx, psm in tqdm(enumerate(pin)):
            if idx in psm_dict.keys():
                samp_list[psm_dict[idx]].append(psm)

    for out_file, samp in tqdm(zip(out_files, samp_list)):
        with open(out_file, "wb+") as out_pin:
            out_pin.writelines(samp)

    return out_files


def sample_confident_psms(num, total, times, conf, psms, pin_file, suffix):
    """Sample the number of confident PSMs"""
    out_files = [os.path.join("pin-out", f"train_conf_samp{num}_rep{i}{suffix}.pin")
                 for i in range(times)]

    all_exist = all([os.path.isfile(f) for f in out_files])
    if all_exist:
        logging.info("All files for num=%i, times=%i are present. Skipping...",
                     num, times)
        return out_files

    conf_set = set(conf)
    other = [p for p in psms if p not in conf_set]
    conf = np.random.permutation(conf)[:num*times]
    other = np.random.permutation(other)[:(total-num)*times]
    psms = np.concatenate([conf, other])

    conf_assignments = np.repeat(np.arange(times), num)
    other_assignments = np.repeat(np.arange(times), total-num)
    assignments = np.concatenate([conf_assignments, other_assignments])

    # create row dictionary
    psm_dict = {r: a for r, a in zip(psms, assignments)}

    # initialize row lists
    samp_list = [[] for _ in range(times)]

    logging.info("Sampling %i PSMs, %i times...", num, times)
    # Read through PIN, appending selected rows to each list.
    with gzip.open(pin_file, "r") as pin:
        header = pin.readline()
        for samp in samp_list:
            samp.append(header)

        for idx, psm in tqdm(enumerate(pin)):
            if idx in psm_dict.keys():
                samp_list[psm_dict[idx]].append(psm)

    for out_file, samp in tqdm(zip(out_files, samp_list)):
        with open(out_file, "wb+") as out_pin:
            out_pin.writelines(samp)

    return out_files


def percolate(pin_file, name, weights=None):
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
               "--output-dir", "perc-out"]

        if weights is not None:
            cmd += ["--init-weights", weights, "--static", "T"]

        subprocess.run(cmd + [pin_file], check=True)

    return out_files


def percolate_kim(gz_file, thold=0.01):
    """Percolate the Kim dataset, return line numbers of confident PSMs"""
    out_files = [os.path.join("perc-out", "kim.percolator.target.psms.txt"),
                 os.path.join("perc-out", "kim.percolator.decoy.psms.txt")]

    all_exist = all([os.path.isfile(f) for f in out_files])
    pin_file = os.path.join("pin-out", "kim.pin")
    if not all_exist:
        if not os.path.isfile(pin_file):
            logging.info("Decompressing %s...", gz_file)
            with open(pin_file, "w+b") as pin:
                with gzip.open(gz_file, "rb") as gz:
                    shutil.copyfileobj(gz, pin)

        cmd = ["crux", "percolator",
               "--fileroot", "kim",
               "--subset-max-train", "1000000",
               "--tdc", "T",
               "--output-dir", "perc-out"]

        logging.info("Percolating %s...", pin_file)
        subprocess.run(cmd + [pin_file], check=True)

    idx_out = os.path.join("pkl-out", "conf_idx.pkl")
    if os.path.isfile(idx_out):
        with open(idx_out, "rb") as idx_file:
            conf_idx = pickle.load(idx_file)

        return conf_idx

    # Get indices of confident PSMs
    logging.info("Finding confident PSMs...")
    perc_df = wispy.percolator.read(out_files[0])
    perc_expmass = (perc_df["spectrum neutral mass"] + 1.007276).round(4).astype(str)
    perc_df = perc_df.set_index(perc_df["scan"].astype(str) + "_" + perc_expmass)
    perc_df = perc_df[perc_df["percolator q-value"] <= thold]

    conf_id = set(perc_df.index)
    del perc_df # Clean up memory

    logging.info("Finding confident PSMs in PIN file...")
    with open(pin_file, "rt") as pin:
        _ = pin.readline()
        conf_loc = [(get_id(l) in conf_id) for l in tqdm(pin)]

    conf_idx = list(np.nonzero(conf_loc)[0])
    with open(idx_out, "w+b") as idx_file:
        pickle.dump(conf_idx, idx_file)

    return conf_idx


def get_id(line):
    """Extract an ID to match"""
    line = line.split("\t")
    expmass = str(round(float(line[3]), 4))
    return line[2] + "_" + expmass


def read_perc(psm_file):
    """Read a Percolator tab-delimited file"""
    psms_df = wispy.percolator.read(psm_file)
    file_base = os.path.basename(psm_file.split(".")[0])
    psms_df["name"] = file_base
    psms_df["major"] = "major" in file_base
    psms_df["samp"] = int(re.search("samp(.+?)_", file_base).group(1))
    psms_df["rep"] = re.search("rep(.)", file_base).group(1)
    return psms_df


# Figure-making Functions -----------------------------------------------------
def make_figures(all_df, conf_df, res=300):
    """Make cool plots"""
    # Plot seed vs seed q-value curves
    qlabels = [["A", "B", "C", "D"], ["E", "F", "G", "H"]]

    wide_all = widen(all_df[all_df.major])
    wide_conf = widen(conf_df[conf_df.major])

    _, axs = plt.subplots(2, 4, figsize=(7, 4), sharey=True)
    plot_qvalues(wide_all, axs[0, :].flatten(), qlabels[0])
    plot_qvalues(wide_conf, axs[1, :].flatten(), qlabels[1])
    plt.savefig("figures/seeds_qvalue_pairs.png", dpi=res)

    _, axs = plt.subplots(2, 4, figsize=(7, 4), sharey=True)
    plot_btw_qvalues(wide_all, axs[0, :].flatten(), qlabels[0])
    plot_btw_qvalues(wide_conf, axs[1, :].flatten(), qlabels[1])
    plt.savefig("figures/sample_qvalue_pairs.png", dpi=res)

    # Plot the number of passing PSMs
    _, axs = plt.subplots(2, 1, figsize=(3.33, 4.5))
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


def widen(pin_df, values="percolator q-value"):
    """Widen a concatenated pin dataframe"""
    groups = ["scan", "sequence", "samp", "rep"]
    wide = pin_df.groupby(groups)[values] \
                 .sum() \
                 .unstack("rep") \
                 .reset_index()

    return wide


def count_passing(df, thold=0.01):
    """Count the number of passing psms"""
    df["passing"] = df["percolator q-value"] <= thold
    ret_df = df.groupby(["samp", "rep"]).passing.sum()
    return ret_df.reset_index()


def plot_psms(df, ax, rev=True):
    """Plot the number of PSMs accepted"""
    ax.set(xscale="log")
    sns.lineplot(x=df.samp, y=df.passing, ci=95, marker="o", ax=ax)
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
        samp_df = group[1]
        ax.axhline(0.01, color="black", linewidth=1, linestyle="dashed")
        ax.axvline(0.01, color="black", linewidth=1, linestyle="dashed")
        ax.scatter(samp_df["1"].values, samp_df["2"].values, alpha=0.01, s=10, marker=".")
        ax.set_xlabel("q-value (Rep 1)")

        if i == len(axs) - 1:
            ax.set_ylabel("q-value (Rep 2)")

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
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    _ = wispy.theme.paper()
    prepare_dirs()

    # Percolate the full dataset ----------------------------------------------
    conf_psms = percolate_kim(SOURCE_FILE)

    # Check if results are already present
    out_dfs = ["df-out/overall_res.pkl", "df-out/conf_res.pkl"]
    if all([os.path.isfile(f) for f in out_dfs]):
        logging.info("Loading saved results...")
        return [pd.read_pickle(f) for f in out_dfs]

    # Make splits
    test, train = split_psms(NUM_LINES, TEST_SAMPLE)
    test_file = rows_to_pin(test, SOURCE_FILE, "pin-out/test.pin")
    test_set = set(test)
    conf_train = [p for p in conf_psms if p not in test_set]

    # Downsample overall PSMs -------------------------------------------------
    np.random.seed(SEED2)
    overall_res = []
    total_major = [int(1*10**i) for i in range(2, 6)]
    total_minor = [int(1*10**i) for i in (3.5, 2.5, 6)]
    total_type = ["major"]*len(total_major) + ["minor"]*len(total_minor)

    for num, mtype in zip(total_major + total_minor, total_type):
        pin_files = sample_psms(num, TIMES, train, SOURCE_FILE, f"_{mtype}")
        for pin_file in pin_files:
            out_root = pin_file.split(".")[0].split("_", 1)[1]
            train_res = percolate(pin_file, f"train_{out_root}")
            test_res = percolate(test_file, f"test_{out_root}",
                                 weights=train_res[4])
            overall_res.append(read_perc(test_res[0]))

    overall_res = pd.concat(overall_res)
    overall_res.to_pickle(out_dfs[0])

    # Downsample confident PSMs -----------------------------------------------
    np.random.seed(SEED3)
    conf_pin = os.path.join("pin-out", "train_samp1000000_rep0_minor.pin")
    conf_res = []
    conf_major = [int((HICONF_TOTAL * HICONF_FRAC) / 10**i) for i in range(4)]
    conf_minor = [int((HICONF_TOTAL * HICONF_FRAC) / 10**i) for i
                  in (1.5, 2.5, 3.5, 4)]

    conf_type = ["major"]*len(conf_major) + ["minor"]*len(conf_minor)

    for num, mtype in zip(conf_major + conf_minor, conf_type):
        pin_files = sample_confident_psms(num, HICONF_TOTAL, TIMES, conf_train,
                                          train, SOURCE_FILE, f"_{mtype}")
        for pin_file in pin_files:
            out_root = pin_file.split(".")[0].split("_", 1)[1]
            train_res = percolate(pin_file, f"train_{out_root}")
            test_res = percolate(test_file, f"test_{out_root}",
                                 weights=train_res[4])
            conf_res.append(read_perc(test_res[0]))

    conf_res = pd.concat(conf_res)
    conf_res.to_pickle(out_dfs[1])

    return overall_res, conf_res


if __name__ == "__main__":
    dfs = main()
    logging.info("Making Figures...")
    make_figures(*dfs)
