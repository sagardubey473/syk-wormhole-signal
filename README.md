# Coupled SYK Traversable Wormhole: Numerical Research

Research-grade numerical simulator for the Maldacena-Qi traversable wormhole
protocol in coupled Sachdev-Ye-Kitaev (SYK) systems.

## Research Questions

1. **RQ1 (Sparsity Dependence):** At what level of SYK sparsification does the
   wormhole transmission signal lose its holographic character?

2. **RQ2 (Mimicry Diagnostic):** What quantitative diagnostic reliably distinguishes
   authentic holographic wormhole dynamics from non-holographic systems that produce
   similar-looking transmission correlators?

3. **RQ3 (Noise Robustness):** How does the wormhole signal degrade under realistic
   decoherence, and what noise thresholds determine signal survival?

## Context

This investigation is motivated by the controversy surrounding the 2022 Google
quantum-processor wormhole experiment (Jafferis et al., Nature 612, 51, 2022),
which claimed to observe traversable wormhole dynamics on a heavily sparsified
SYK model implemented on the Sycamore processor.

## Project Structure

```
wormhole_research/
  src/          - Core simulation modules (Majorana, SYK, doubled system, TFD, etc.)
  tests/        - Unit tests for all modules
  notebooks/    - Jupyter notebooks for analysis and results
  data/         - Cached disorder-averaged results (NPZ/HDF5)
  results/      - Final plots (PNG) and source data
```

## Key References

- Maldacena & Stanford, arXiv:1604.07818 (SYK model)
- Maldacena & Qi, arXiv:1804.00491 (Eternal traversable wormhole)
- Maldacena, Shenker & Stanford, arXiv:1503.01409 (Chaos bound)
- Jafferis et al., Nature 612, 51 (2022) (Google experiment)
- Kobrin, Schuster & Yao, arXiv:2302.07897 (Critique)

## Conventions

All sign, normalization, and basis conventions are documented in CONVENTIONS.md.

## Requirements

Python 3.10+. See requirements.txt for dependencies.
