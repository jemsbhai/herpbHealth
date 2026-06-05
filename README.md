# herpbHealth

Privacy-preserving RAG for healthcare using two-server additive secret sharing.

## Overview

This repository contains the experimental evaluation and paper for applying a two-server additive secret sharing protocol to healthcare retrieval-augmented generation (RAG) workflows. The cryptographic primitives are implemented in the companion [herpb](https://github.com/jemsbhai/herpb) package; this repo focuses on healthcare datasets, medical embedding models, clinical deployment scenarios, and end-to-end evaluation.

**Target venue:** IEEE HealthCom (International Conference on e-Health Networking, Application & Services)

## Setup

```powershell
# Clone and install the companion package first
cd E:\data\code\claudecode
pip install -e .\herpb

# Then install this package
cd herpbHealth
pip install -e .
```

## Project Structure

```
herpbHealth/
  configs/          Experiment configurations (YAML)
  data/             Datasets (raw + processed)
  experiments/      Experiment outputs (checkpoints, logs, results)
  figures/          Publication-ready figures
  notebooks/        Exploratory analysis
  paper/            LaTeX source
  scripts/          Experiment entry points
  src/herpb_health/ Python package
  tests/            Test suite
```

## Reproducing Results

See `LOGBOOK.md` for the full experimental record. Each experiment (EXP-NNN) includes the exact configuration, environment, and commands needed to reproduce.

## License

MIT
