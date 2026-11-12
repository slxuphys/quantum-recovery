# Quantum recovery

Research code and numerical examples for quantum recovery, including Petz
recovery, quantum singular value transformation (QSVT), and decoding methods for
approximate quantum error correction and information retrieval. Each project has
its own directory; reusable numerical tools are shared at the repository root.

## Table of contents

- [Petz recovery](#petz-recovery)
- [Setup](#setup)

## Petz recovery

The [working notebook](petz/Petz.ipynb) accompanies:

**Petz Recovery through a Single Polar Transformation for Approximate Quantum
Error Correction** — Hanzhang Zhao and Shenglong Xu,
[arXiv:2610.20515](https://arxiv.org/abs/2610.20515).

It introduces the Petz channel and its polar representation, block encoding and
QSVT, optimal interval polynomials, and the connection between spectrum-weighted
loss and diamond-distance bounds. Two numerical examples compare approximate
recovery with ideal Petz recovery:

- **Haar-random encoding:** recover the logical state from an accessible output
  subsystem after the remaining qubits are discarded.
- **Hayden–Preskill encoding:** recover using the accessible output and an
  entangled side-information register.

The dense 12-qubit Hayden-Preskill calculations can require several GB of memory and about 15 minutes of runtime on a typical modern laptop.

All the figures in the manuscript are reproduced in the notebook and saved in the
[figure folder](petz/figure/).

## Setup

Use Python 3.10 or newer. From the repository root, create a virtual environment
and install the dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then open the Petz notebook:

```sh
cd petz
python -m jupyter lab Petz.ipynb
```


Plotting cells use external LaTeX rendering. Install LaTeX or change
`"text.usetex": True` to `False` in those cells to use Matplotlib's built-in
rendering. 

