# Code to Evaluate Static Models in Percolator

This repository contains the code for the analyses presented in our manuscript,
["A machine learning strategy that leverages large datasets to boost statistical
power in small-scale experiments"](https://doi.org/10.1101/849331).

## Reproducing the manuscript  
The code provided can fully reproduce the figures and analyses presented in the
manuscript, provided that the necessary software are installed and data are
present. 

### Required Software
The analysis scripts are written for Python 3.7+ and also require a recent
version of the [Crux mass spectrometry toolkit](https://crux.ms).

Specifically, we recommend using the Anaconda Python distribution and running
the following commands to verify the prerequisite software are installed:

```bash
conda install numpy pandas matplotlib seaborn
conda install -c conda-forge tqdm
```

Additionally, you'll need the Python packages that I use internally to provide
common utility functions. These can be installed from GitHub using pip:

```bash
pip install git+git://github.com/wfondrie/wispy
pip install git+git://github.com/wfondrie/targetdecoy
```

Finally, you will also need a recent version of [Crux tookit](http://crux.ms). Navigate
to the "Download" page, select an appropriate version and click "I agree to the
licensing terms, download the most recent build of Crux". You will then need to
verify Crux has been added to your path.


### Data  
The directory structure is expected to look like this:
```
README.md (this file)
results (analysis scripts and where results are written)
data
|- 2019specht-high
|  `- mzML
|     `- (mzML.gz files for the SCoPE-MS data)
|- 2019basilicata-denovo
|  `- mzML
|     `- (mzML.gz files for the gel bands)
|- fasta
|  `- human.fasta (the fasta protein database)
`- pin
   `- kim.pin.gz (PSMs from the Kim et al. dataset)
```


The mass spectrometry data files for the Basilicata *et al.* gel band
experiments and the Specht *et al.* single-cell proteomics experiments are
required for the analysis scripts. These must be downloaded an converted to
gzipped mzML (*.mzML.gz) format for analysis. Make sure to put these in the
appropriate directory after conversion. All of the Thermo raw files can be
downloaded with the following (assuming wget is available):

```bash
mkdir -p data/2019basilicata-denovo/raw
wget -N -P data/2019basilicata-denovo/raw ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2018/09/PXD009317/*.raw

mkdir -p data/2019specht-high/raw
wget -N -P data/2019specht-high/raw ftp://massive.ucsd.edu/MSV000083945/raw/scope2_raw/*.raw
```

Given the size of the Kim *et al.* draft map of the human proteome dataset,
we've provided the Percolator input file (pin) for the entire dataset via
[figshare](https://figshare.com/articles/Percolator_input_file_from_the_Kim_et_al_Draft_Map_of_the_Human_Proteome/10565477).
Run the following commands to download it into the correct location:

```bash
mkdir -p ./data/pin
wget -N -O ./data/pin/kim.pin.gz https://ndownloader.figshare.com/files/19068101
```

Finally, you need the FASTA protein database to search against. For these
analyses, you need the Uniprot/Swiss-Prot canonical human proteome. This can be
obtained from the following address:
https://www.uniprot.org/uniprot/?query=proteome:UP000005640%20reviewed:yes

## Running the analyses
If all of the data are downloaded in the correct locations, running the scripts
in easy:

To run the downsampling experiments:
```
cd results/downsampling && python runall.py
```

To run the analysis of histone gel bands:
```
cd results/gel-bands && python runall.py
```

To run the analysis of SCoPE-MS experiments:
```
cd results/scope-ms && python runall.py
```

## Results

All of the result and intermediate files are written to the results directory in
appropriate subdirectories alongside the `runall.py` script that generated them.

## Questions?  

If you have problems or questions, feel free to ask Will Fondrie (wfondrie@uw.edu).
