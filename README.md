<p align="center">
  <img src=".github/delphi-logo-white-bg.svg" width="400" alt="Delphi Logo"/>
</p>

## Learning the natural history of human disease with generative transformers

[![Paper](https://img.shields.io/badge/Paper-medRxiv-blue)](https://www.medrxiv.org/content/10.1101/2024.06.07.24308553v1)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/mit)

**Authors:** Artem Shmatko*, Alexander Wolfgang Jung*, Kumar Gaurav*, Søren Brunak, Laust Mortensen, Ewan Birney, Tom Fitzgerald, Moritz Gerstung (*Equal Contribution)

## Overview

This repository contains the code for **Delphi**, a modified GPT-2 model designed to learn the natural history of human disease using generative transformers. The implementation is based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and includes training code and analysis notebooks.

## Installation

### Option 1: Conda Environment

1. **Clone the repository:**

   ```bash
   git clone https://github.com/gerstung-lab/Delphi.git
   cd Delphi
   ```

2. **Create and activate the conda environment:**

   ```bash
   conda create -n delphi python=3.11
   conda activate delphi
   pip install -r requirements.txt
   ```

   > **Note:** Installing requirements typically takes a few minutes.

### Option 2: Docker

We provide a Dockerfile for containerized training and downstream analyses. See `containers/Dockerfile` for implementation details.

## Data

### UK Biobank Access

Delphi-2M is trained on 500K patient health trajectories from the UK Biobank dataset. Access to this data requires a research application through the [UK Biobank](https://www.ukbiobank.ac.uk/).

### Data Preparation

For detailed instructions on preparing training data, please refer to [`data/README.md`](data/README.md).

## Configuration and Training

### Prerequisites

Set the following environment variables:

- `DELPHI_DATA_DIR`: Directory containing training and validation data
- `DELPHI_CKPT_DIR`: Directory for storing model checkpoints

> **Tip:** We recommend using a `.env` file with [direnv](https://direnv.net) for environment management.

## Development

### Code Quality

This project uses [`pre-commit`](https://pre-commit.com) hooks to maintain code quality standards.

#### Setup Pre-commit Hooks

1. **Install pre-commit** (if not already available):

   ```bash
   # Via conda (recommended for base environment)
   conda install pre-commit
   # Or via Homebrew
   brew install pre-commit
   ```

2. **Install hooks in your local repository:**

   ```bash
   # From project root directory
   pre-commit install --install-hooks
   ```

3. **Manual execution** (optional):

   ```bash
   pre-commit run --all-files
   ```

## Citation

If you use this work, please cite our paper:

```bibtex
@article{Shmatko2024.06.07.24308553,
    title = {Learning the natural history of human disease with generative transformers},
    author = {Shmatko, Artem and Jung, Alexander Wolfgang and Gaurav, Kumar and Brunak, S{\o}ren and Mortensen, Laust and Birney, Ewan and Fitzgerald, Tom and Gerstung, Moritz},
    doi = {10.1101/2024.06.07.24308553},
    journal = {medRxiv},
    publisher = {Cold Spring Harbor Laboratory Press},
    year = {2024}
}
```

## License

This project is licensed under the MIT License - see the badge above for details.
