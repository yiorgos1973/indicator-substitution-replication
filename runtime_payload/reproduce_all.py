from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent


def verify_manifest():
    rows = list(csv.DictReader((ROOT / 'MANIFEST.csv').open(encoding='utf-8')))
    for row in rows:
        path = ROOT / row['path']
        if not path.is_file():
            raise FileNotFoundError(f"Required input missing: {row['path']}; see MISSING_INPUTS.csv")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f"Changed frozen file: {row['path']}")
    return len(rows)


class LedgerRecorder:
    def __init__(self, output, jobs, run_job, write_job, plan_hash):
        self.output = output
        self.jobs = jobs
        self.run_job = run_job
        self.write_job = write_job
        self.plan_hash = plan_hash
        self.started = time.perf_counter()
        self.count = 0

    def __call__(self, matrix, job):
        result = self.run_job(matrix, job)
        self.write_job(self.output, job, result, self.plan_hash)
        self.count += 1
        if self.count % 10 == 0:
            elapsed = time.perf_counter() - self.started
            remaining = elapsed * (len(self.jobs) - self.count) / self.count
            print(f'Ledger jobs {self.count}/{len(self.jobs)}; elapsed {elapsed:.1f}s; linear ETA {remaining:.1f}s (job costs differ)', flush=True)
        return result


def frozen_run(output):
    sys.path.insert(0, str(ROOT / 'candidate'))
    import reproduce
    import pandas as pd
    from phase_b.orchestrator import _write_job, assemble_ledgers
    from phase_b.plan import plan_hash

    reproduce.check_manifest()
    output.mkdir(parents=True)
    empirical = output / 'empirical'
    empirical.mkdir()
    frame = pd.read_csv(ROOT / 'candidate/original/data/prediction/model_matrices.csv')
    matrices = {tuple(key): group.reset_index(drop=True) for key, group in frame.groupby(['scenario_id', 'target_direction', 'construction_role'], sort=True)}
    jobs = reproduce.build_execution_plan(reproduce.MatrixBundle(matrices, pd.DataFrame()))
    recorder = LedgerRecorder(empirical, jobs, reproduce.run_job, _write_job, plan_hash(jobs))
    reproduce.run_job = recorder
    core_count = reproduce.core_checks(output)
    job_count, comparison_count = reproduce.prediction_checks(output, True)
    for name in ('execution_plan.csv', 'matrix_manifest.csv', 'preflight_status.json', 'gate_results.json'):
        shutil.copyfile(ROOT / 'expected_empirical' / name, empirical / name)
    assemble_ledgers(empirical, jobs, plan_hash(jobs))
    validation = finalize_and_validate(empirical, output)
    return {'core_comparisons': core_count, 'prediction_jobs': job_count, 'prediction_comparisons': comparison_count, **validation}


def finalize_and_validate(empirical, output):
    sys.path.insert(0, str(ROOT / 'candidate'))
    import reproduce
    from phase_b.finalize_run import finalize_run
    from validate_empirical_run import write_validation_outputs
    from clean_rebuild_phase_b import compare_outputs

    finalize_run(empirical, 'phase_b_20260930_13fb5775_v2')
    compare_outputs(ROOT / 'expected_empirical', empirical, ROOT / 'COMPARISON_RULES.csv', output / 'FULL_LEDGER_COMPARISON.csv')
    checks, status = write_validation_outputs(empirical, output / 'independent_empirical')
    if status['status'] != 'PASS':
        raise ValueError('Independent empirical validation failed')
    return {'independent_checks': len(checks), 'full_ledger_comparison': 'PASS'}


def source_check(repository, output):
    rows = list(csv.DictReader((ROOT / 'SOURCE_ALLOWLIST.csv').open(encoding='utf-8')))
    output.mkdir(parents=True)
    missing = []
    for row in rows:
        path = repository / row['source_path']
        if not path.is_file():
            missing.append({**row, 'status': 'missing'})
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            missing.append({**row, 'status': 'hash_mismatch'})
    with (output / 'MISSING_INPUTS.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) + ['status'])
        writer.writeheader()
        writer.writerows(missing)
    if missing:
        raise FileNotFoundError(f'{len(missing)} frozen repository inputs missing or changed; see {output.name}/MISSING_INPUTS.csv')
    return {'verified_repository_inputs': len(rows), 'source_acquisition_performed': False, 'construction_rebuild_performed': False}


def main():
    parser = argparse.ArgumentParser(description='Frozen-input reproduction and authorized-source inventory; no automatic source downloads.')
    parser.add_argument('--mode', choices=['frozen', 'source-check', 'saved-run'], default='frozen')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repository-root', type=Path)
    parser.add_argument('--run-directory', type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    count = verify_manifest()
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    if args.mode == 'source-check':
        if args.repository_root is None:
            raise ValueError('--repository-root is required for source-check')
        results = source_check(args.repository_root, args.output)
    elif args.mode == 'saved-run':
        if args.run_directory is None:
            raise ValueError('--run-directory is required for saved-run')
        args.output.mkdir(parents=True)
        results = finalize_and_validate(args.run_directory, args.output)
        results['empirical_fitting_performed'] = False
    else:
        results = frozen_run(args.output)
    results.update(status='PASS', mode=args.mode, manifest_files_checked=count, elapsed_seconds=time.perf_counter() - start)
    (args.output / 'VALIDATION.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
