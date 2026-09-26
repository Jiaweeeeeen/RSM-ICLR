"""The study protocol and the active study.

The root modules define what an experiment is — contracts and identities,
config resolution, environment contracts and rosters, run-directory layout,
event records, evaluation summaries and references, qualification records —
without importing the AMAGO runtime in :mod:`reasoned_icrl.runtime`. Each study
lives in its own sub-package; :mod:`reasoned_icrl.experiments.summary_memory`
is the active one.
"""
