# Changelog

## 0.1.0 - 2026-05-31

Initial public release of the C-alpha relational-network package for protein residue interaction scoring.

### Added

- Multi-model PDB parsing for C-alpha trajectories
- Residue and pairwise distance-dynamics feature extraction
- Symmetric relational network for residue-pair scoring
- Self-supervised training and inference pipelines
- YAML-driven configuration and CLI entry point
- CSV score export and checkpoint save/load support
- Example config, example workflow, and integration tests

### Notes

- The package is designed to keep the core model reusable while making I/O and hyperparameters configurable from YAML.
- The release targets Python 3.11+.
