# Setup

Everything here runs on a laptop. No cluster, no GPU, no data-access paperwork.

## 1. Clone

```shell
git clone https://github.com/CNNC-Lab/compbio-2026.git
cd compbio-2026
```

New to `git`? Read [this short introduction](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git), or use [GitHub Desktop](https://desktop.github.com/), which hides most of it.

## 2. Create the environment

```shell
conda env create -f environment.yml
conda activate compbio-2026
```

This installs the project itself in editable mode, so `import compbio2026` works from anywhere and your edits to `src/` take effect immediately.

The environment is deliberately small — around fifteen packages. The 2025 course environment carried the whole bioinformatics stack (scanpy, scvi-tools, pysam, samtools, opencv, pytorch) and took a long time to solve; none of it is used here.

If you would rather not use conda:

```shell
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install numpy scipy pandas scikit-learn umap-learn tables h5py matplotlib seaborn jupyterlab group-lasso celer
```

## 3. Get the data

```python
from compbio2026 import data
shd = data.load("train")          # downloads on first call, ~200 MB, then cached in data/
```

Already have the file from the 2025 repository? Point at it and skip the download:

```python
shd = data.load(path="~/repos/_education/computational-biology-2025/data/hdspikes/shd_train.h5")
```

## 4. Check it works

```shell
jupyter lab notebooks/01_data_and_statistics.ipynb
```

Run the first three cells. You should get a raster that looks like a spoken digit.

## Optional: LAUSCHER

Only needed if you want the *ground-truth* hair-cell layer rather than the estimate (see [06-hair-cell-recovery.md](06-hair-cell-recovery.md)). It is installed by `environment.yml`, but it is the one dependency that pulls from GitHub rather than a package index, so it is also the one most likely to fail behind a restrictive network:

```shell
pip install git+https://github.com/electronicvisions/lauscher
```

Nothing else in the project depends on it. If it fails, carry on.

## Troubleshooting

**`conda env create` hangs on "Solving environment".** Install [mamba](https://mamba.readthedocs.io/) and use `mamba env create -f environment.yml`. It solves the same file in seconds.

**`ImportError: No module named compbio2026`.** The editable install did not run. From the repository root, with the environment active: `pip install -e .`

**The download fails or is slow.** The file is hosted at <https://zenkelab.org/datasets>. Download `shd_train.h5.gz` by hand, put it in `data/hdspikes/`, and `data.load()` will find and decompress it.

**Windows.** Everything in this project is pure Python and works on Windows. (The 2025 Project 1 needed NEST, which does not; that constraint no longer applies.)
