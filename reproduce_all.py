from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest():
    rows = list(csv.DictReader((ROOT / 'MANIFEST.csv').open(encoding='utf-8')))
    for row in rows:
        path = ROOT / row['path']
        if digest(path) != row['sha256']:
            raise ValueError('Changed packaged file: ' + row['path'])
    return len(rows)


def verify_results():
    import numpy as np
    import pandas as pd

    payload = ROOT / 'runtime_payload'
    sys.path.insert(0, str(payload / 'candidate/original/code/src'))
    from protocol_final.core import deterministic_inner_folds
    from phase_b.matrices import _cell_hash

    metrics = pd.read_csv(payload / 'expected_empirical/METRICS.csv')
    comparison = pd.read_csv(ROOT / 'validation/upstream/fresh_validation/PREDICTION_COMPARISON.csv')
    assert len(metrics) == 252 and metrics.job_id.is_unique and len(comparison) == 1008
    indexed = metrics.set_index('job_id')
    for metric, group in comparison.groupby('metric'):
        assert set(group.job_id) == set(metrics.job_id)
        np.testing.assert_allclose(group.archived, indexed.loc[group.job_id, metric], atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(group.rebuilt, group.archived, atol=1e-10, rtol=1e-8)
        assert group.status.eq('PASS').all()
    core = pd.read_csv(ROOT / 'validation/upstream/fresh_validation/CORE_COMPARISON.csv')
    assert len(core) == 24 and core.status.eq('PASS').all()
    np.testing.assert_allclose(core.rebuilt, core.archived, atol=1e-10, rtol=1e-8)
    samples = pd.read_csv(ROOT / 'sample_identities/core_countries.csv')
    rq1 = set(samples.loc[samples.rq1, 'iso3'])
    rq2 = set(samples.loc[samples.rq2, 'iso3'])
    assert len(rq1) == 66 and len(rq2) == 65 and rq1 - rq2 == {'TWN'}
    cells = pd.read_csv(ROOT / 'sample_identities/model_cells.csv')
    manifest = pd.read_csv(payload / 'expected_empirical/matrix_manifest.csv')
    for row in manifest.itertuples(index=False):
        group = cells[(cells.scenario_id == row.scenario_id) & (cells.target_direction == row.target_direction) & (cells.construction_role == row.construction_role)]
        assert len(group) == row.rows and group.iso3.nunique() == row.countries
        assert _cell_hash(group) == row.cell_identity_sha256
    primary = cells[(cells.scenario_id == 'PRIMARY') & (cells.target_direction == 'HLO_to_P')]
    assert len(primary) == 53 and primary.iso3.nunique() == 38
    splits = pd.read_csv(payload / 'expected_empirical/SPLITS.csv', keep_default_na=False)
    assert len(splits) == 233688
    assert not splits.inner_country.eq(splits.outer_country).any()
    fold_groups = 0
    for _, group in splits.groupby(['job_id', 'fit_id'], sort=False):
        expected = deterministic_inner_folds(group.inner_country)
        assert (group.inner_country.map(expected).to_numpy() == group.inner_fold.to_numpy()).all()
        fold_groups += 1
    for name, total in [('PHASE_B_CALIBRATION_SOURCE.csv', 1252), ('PHASE_B_METRICS_SOURCE.csv', 1512), ('PHASE_B_RANK_SUMMARY_SOURCE.csv', 192)]:
        assert len(pd.read_csv(ROOT / 'tables/archived' / name)) == total
    tails = pd.read_csv(ROOT / 'tables/archived/TAIL_SUMMARIES.csv')
    assert len(tails) == 1696
    assert not {'target_threshold', 'prediction_threshold'} & set(tails.columns)
    upstream = json.loads((ROOT / 'validation/upstream/fresh_validation/VALIDATION.json').read_text())
    assert upstream['status'] == 'PASS' and upstream['prediction_jobs'] == 252 and upstream['independent_checks'] == 11
    return {'status': 'PASS', 'scope': 'archived results, exact sample identities and deterministic folds; no model refit', 'core_comparisons_checked': 24, 'loss_comparisons_checked': 1008, 'prediction_jobs': 252, 'model_matrices': len(manifest), 'split_rows': len(splits), 'fold_groups': fold_groups, 'primary_cells': 53, 'primary_countries': 38, 'absolute_tolerance': 1e-10, 'relative_tolerance': 1e-8, 'fold_tolerance': 0, 'models_refitted': False}


def main():
    parser = argparse.ArgumentParser(description='Check archived results and deterministic sample/fold identities without refitting.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    files = verify_manifest()
    args.output.mkdir(parents=True)
    results = verify_results()
    results.update(packaged_files_verified=files, elapsed_seconds=time.perf_counter() - start)
    (args.output / 'VALIDATION.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
