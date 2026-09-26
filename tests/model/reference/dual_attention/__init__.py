"""Pinned upstream Dual Attention sources, vendored for numerical parity tests.

Copied verbatim from https://github.com/awni00/dual-attention at commit
``dce218cbf5ec9aa7f90687c1323050a1fba17966``, MIT licensed; see ``LICENSE`` in
this directory. These files are test fixtures only. Nothing under ``src/``
imports them, and the package is not a runtime dependency.

``tests/test_dat_transformer.py`` verifies each file's SHA-256 against
``docs/DAT_INTEGRATION_SOURCES.json`` before using it, so a silent upstream
substitution cannot pass as parity evidence.
"""
