"""Closed migration ledger: accepted (archived, migrated) source digest pairs.

Only three leaf studies participate. Both endpoints are fixed literals: never
compute the accepted live endpoint at import time. A subsequent edit must fail
against the historical archives. Numerical replay does not prove equivalence of
arbitrary source edits. Other experiment chains keep their original contracts.
"""

MIGRATION_BASELINE = {
    'tools/velocity_cost_study.py': (
        '1d245735aeadb3bd8c62b843ed576c884d44db4f862c0317da082a136b9be89c',
        '2fea7fda074813146be8197badfe2eed14bf270a025c33662df7377dbeb00dcc',
    ),
    'tools/onset_observer_study.py': (
        '059f11527aade733dddf416270ac575ccf336233f10392ee1f6c8a68152fb48a',
        '9563b05fe480836eb4bb5274f2551a1052cdf38977dd24af2fedf5a4db4d43e1',
    ),
    'tools/velocity_time_study.py': (
        '1af037c61f9bcf1a3a3996dd1c8b00f86b9da1bdbbe675979130d1a601900657',
        '9aced999af4a252920b9fc87b443787d562180ae7db526b1610ce0c1070577a3',
    ),
}

MAXIMUM_ENTRIES = 3


def baseline_for(*paths: str) -> dict[str, tuple[str, str]]:
    """Return only the reviewed transitions a study names explicitly."""
    missing = [path for path in paths if path not in MIGRATION_BASELINE]
    if missing:
        raise KeyError(f"no migration baseline recorded for: {missing}")
    return {path: MIGRATION_BASELINE[path] for path in paths}
