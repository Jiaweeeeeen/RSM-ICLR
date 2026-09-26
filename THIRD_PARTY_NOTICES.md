# Third-party notices

This project composes upstream packages; it does not claim ownership of their
source. The authoritative license text distributed with each installed package
controls. Direct dependencies include:

- AMAGO 3.4.0 — MIT License, copyright Jake Grigsby and contributors;
- POPGym 1.0.7 — MIT License;
- Gymnasium 0.29.1 — MIT License;
- NumPy and SciPy — BSD-family licenses;
- PyTorch — BSD-style license;
- Matplotlib — PSF-based license with separately licensed bundled assets;
- gin-config — Apache License 2.0.

`reasoned_icrl/model/memo_transformer.py` is a project-owned re-implementation
of the Memo method (Gupta, Yadav, Kira, Gal and Aljundi, 2025, arXiv
2510.19732) audited against the author-linked source
`https://github.com/Memory-icrl/memo` at commit
`9e7044f2f0e6791b33a6e5ffee6a6722b0b9fe94`, released under the MIT License as a
fork of AMAGO (its `LICENSE` names no holder; the AMAGO copyright is Jake
Grigsby and contributors). No file of that repository is copied; the adaptation
is described in `docs/METHOD.md`.

The native Dark Key-to-Door and MazeRunner tasks are AMAGO's built-in
environments and CountRecall is POPGym's, all used through their public APIs
behind project-owned adapters. `tests/model/reference/dual_attention/` vendors
files from `awni00/dual-attention` (MIT License, commit recorded in its
`SOURCES.json`) as test fixtures for the parity checks only; nothing in the
package imports them.

Before redistributing a wheel or environment archive, copy the full dependency
license files from the resolved environment and review any transitive binary
notices recorded by the package manager.
