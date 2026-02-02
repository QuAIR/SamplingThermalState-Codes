# Thermal-Codes Documentation

Source code for the numerical experiments in the paper Random quantum thermal state sampling.

Preparing Gibbs states of many-body Hamiltonians is a central task in quantum simulation and finite-temperature quantum physics, while existing approaches typically suffer from unfavorable resource scaling at low temperatures.
This repository contains simulation code for a quantum thermal state sampling algorithm based on a measurement-controlled thermal drift channel. The algorithm prepares Gibbs states of local Hamiltonians with polynomial resources in inverse temperature and system size, and the accompanying notebooks reproduce the numerical experiments presented in the paper.

The code focuses on:
1. Sampling Gibbs states of local Hamiltonians using a thermal-drift-based algorithm
2. Verifying theoretical distributional predictions for Pauli observables
3. Studying error scaling with inverse temperature
4. Analyzing precision–variance trade-offs and level-spacing statistics

## Code–Paper Correspondence

| Figure/Protocol in Paper | Location in Repository | Hardware Requirements |
|--------------------------|------------------------|:---------------------:|
| [Figure 2(a)](./code/error%20beta.ipynb) | `./code/error beta.ipynb` | \ |
| [Figure 2(b)](./code/distribution.ipynb) | `./code/distribution.ipynb` | \ |
| [Figure 2(c)](./code/tradeoff.ipynb) | `./code/tradeoff.ipynb` | \ |
| [Figure 3](./code/level%20statistic.ipynb) | `./code/level statistic.ipynb` | \ |

## Repository Structure

```plaintext
Thermal-Codes/
├── code/                         # code for reproducible experiment for the paper
│   ├── data/                     # Cached numerical data (npz files)
├── gibbs/                        # source code for the thermal state sampling algorithm in the paper
```

## How to Run the Notebooks

We recommend running these files by creating a virtual environment using `conda` and install Jupyter Notebook. We recommend using Python `3.10` for compatibility.

### Create a Conda Environment

```bash
conda create -n thermal python=3.10
conda activate thermal
conda install jupyter notebook
```

These codes are highly dependent on the [QuAIRKit](https://github.com/QuAIR/QuAIRKit) package no lower than v0.4.4. This package is featured for batch and qudit computations in quantum information science and quantum machine learning. The minimum Python version required for QuAIRKit is `3.9`.

To install QuAIRKit, run the following commands:

```bash
pip install quairkit
```

## System and Package Versions

It is recommended to run these files on a server with high performance. Below are our environment settings:

**Package Versions**:

* quairkit: 0.5.0
* torch: 2.9.1
* numpy: 2.2.6
* scipy: 1.15.2
* matplotlib: 3.10.8

**System Information**:

* Python version: 3.10.19
* OS: Darwin
* OS version: Darwin Kernel Version 23.5.0
* CPU: Apple M2 Pro

These settings ensure compatibility and performance when running the codes.
