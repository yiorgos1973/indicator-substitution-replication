from dataclasses import asdict
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile
import json
import shutil
import sys

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'construction'))
sys.path.insert(0, str(ROOT / 'runtime_payload/candidate/original/code/src'))

import measurement
import context
import context_manifest
from phase_b.matrices import build_modelling_matrices
from phase_b.plan import build_execution_plan, plan_hash
from phase_b.engine import run_job
from phase_b.orchestrator import _matrix_for_job, _write_job, assemble_ledgers
from phase_b_reporting.processor import compute_results
from phase_b.finalize_run import _metrics, _reporting_ledgers, _paired


SOURCES = {
    'niq_v135.zip': ('https://viewoniq.org/wp-content/uploads/2023/10/NIQ-DATASET-V1.3.5.zip', 'e786460da3ff8a03065251160168b88bce8884321c1427901492e8842731666a'),
    'niq_v133.zip': ('https://viewoniq.org/wp-content/uploads/2019/07/NIQ-DATASET-V1.3.3.zip', 'bfd444e1c7ad4c17a50c7b236f50d379464dd27fd58ba9be49a0850b27a14888'),
    'hlo_disag.dta': ('https://raw.githubusercontent.com/measuringhumancapital/MHC/a61b6452555a080f113687ba6427927bba2d18f9/analysis%20data/hlo_disag.dta', '46aa4b601800cf96a43b4961c8953a4afccf9c9b1c77281fcc9f39d082ef299a'),
    'warne.xlsx': ('https://static-content.springer.com/esm/art%3A10.1007%2Fs40806-022-00351-y/MediaObjects/40806_2022_351_MOESM1_ESM.xlsx', '9eb05090b71a4fd728e13872e788eed9bd382594871265a941acedc297c4989d'),
}


def load_module(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator='\n')


def wdi_table(path):
    with ZipFile(path) as archive:
        name = next(x for x in archive.namelist() if x.startswith('API_') and x.endswith('.csv'))
        return pd.read_csv(BytesIO(archive.read(name)), skiprows=4)


class Replication:
    def __init__(self, output='replication_output', data_version='current'):
        self.data_version = data_version
        self.output = Path(output).resolve()
        self.raw = self.output / 'downloads'
        self.work = self.output / 'workspace'
        self.temporal = self.work / 'rewrite/analysis/temporal'
        self.context = self.work / 'rewrite/analysis/context'
        self.protocol = self.work / 'rewrite/analysis/protocol_final'
        self.core = self.work / 'rewrite/analysis/core'
        self.run = self.output / 'model_run'
        self.tables = self.output / 'tables'
        self.figures = self.output / 'figures'
        for path in [self.raw, self.temporal, self.context, self.protocol, self.core,
                     self.run, self.tables, self.figures, self.work / 'outputs']:
            path.mkdir(parents=True, exist_ok=True)
        self.started = perf_counter()
        self.checks = []

    def download(self):
        records = []
        started = perf_counter()
        for name, (url, expected) in SOURCES.items():
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            digest = sha256(response.content).hexdigest()
            (self.raw / name).write_bytes(response.content)
            records.append(dict(file=name, url=url, sha256=digest, bytes=len(response.content), matches_paper_source=digest == expected))
            print(f'Downloaded {name}: {len(response.content):,} bytes; matches paper source: {digest == expected}.', flush=True)
        for edition in ['135', '133']:
            with ZipFile(self.raw / f'niq_v{edition}.zip') as archive:
                destination = self.raw / f'niq_v{edition}'
                destination.mkdir(exist_ok=True)
                for name in archive.namelist():
                    if name.endswith(('.xlsx', '.txt')):
                        (destination / Path(name).name).write_bytes(archive.read(name))
        attribution = pd.read_csv(ROOT / 'data/wdi_frozen/ATTRIBUTION.csv')
        self.wdi_raw = self.raw / 'wdi'
        self.wdi_raw.mkdir(exist_ok=True)
        for row in attribution.itertuples():
            path = ROOT / row.file
            assert sha256(path.read_bytes()).hexdigest() == row.sha256
            target = self.wdi_raw / path.name
            url = f'https://api.worldbank.org/v2/en/indicator/{row.indicator}?downloadformat=csv'
            if self.data_version == 'current':
                response = requests.get(url, timeout=180)
                response.raise_for_status()
                target.write_bytes(response.content)
            else:
                shutil.copyfile(path, target)
            digest = sha256(target.read_bytes()).hexdigest()
            records.append(dict(file='wdi/' + path.name, url=url, sha256=digest,
                                bytes=target.stat().st_size, matches_paper_source=digest == row.sha256))
            print(f'World Bank {row.indicator}: {self.data_version} data; matches paper source: {digest == row.sha256}.', flush=True)
        from datetime import datetime, timezone
        retrieved = datetime.now(timezone.utc).isoformat()
        (self.output / 'DATA_VERSION.json').write_text(json.dumps(dict(data_version=self.data_version,
            retrieved_utc=retrieved, paper_WDI_vintages='July 2026 GDP/population; September 2026 context',
            note='The paper reports its archived analysis. Revised provider data can change reproduced estimates.'), indent=2), encoding='utf-8')
        csv(pd.DataFrame(records), self.output / 'SOURCE_DOWNLOADS.csv')
        print(f'Downloads and World Bank snapshot checks: {perf_counter() - started:.1f}s.', flush=True)

    def reconstruct(self):
        started = perf_counter()
        measurement.INPUTS = self.work / 'rewrite/analysis/inputs/measurement'
        measurement.TEMPORAL = self.temporal
        measurement.CORE = self.core
        measurement.NIQ_WORKBOOK = self.raw / 'niq_v135/NIQ-DATA (V1.3.5).xlsx'
        measurement.HLO_FILE = self.raw / 'hlo_disag.dta'
        measurement.RQ1_FILE = self.work / 'outputs/headline_analysis_rq1_exact_sample.csv'
        measurement.RQ2_FILE = self.work / 'outputs/headline_analysis_rq2_exact_sample.csv'
        measurement.ARCHIVED_RANKING = ROOT / 'tables/archived/headline_analysis_ranking_summary.csv'
        measurement.ARCHIVED_GDP = ROOT / 'tables/archived/headline_analysis_correlation_difference.csv'
        samples, fav, nat = measurement.recover_niq()
        hlo = measurement.recover_hlo()
        identities = pd.read_csv(ROOT / 'sample_identities/reporting_economies.csv')
        years = hlo.groupby(['code', 'year', 'subject', 'level'], as_index=False).agg(hlo=('hlo', 'mean'))
        components = years.groupby(['code', 'subject', 'level'], as_index=False).agg(hlo=('hlo', 'mean'))
        levels = components.groupby(['code', 'level'], as_index=False).agg(hlo=('hlo', 'mean'))
        aggregate = levels.groupby('code', as_index=False).agg(hlo=('hlo', 'mean'))
        counts = hlo.groupby('code', as_index=False).agg(subjects=('subject', 'nunique'), levels=('level', 'nunique'))
        aggregate = aggregate.merge(counts, on='code').rename(columns={'code': 'iso3'})
        aggregate['eligible_primary_hlo'] = aggregate['subjects'].ge(2) & aggregate['levels'].eq(2)
        latest = years[years['year'].eq(years.groupby(['code', 'subject', 'level'])['year'].transform('max'))]
        latest = latest.groupby(['code', 'level'], as_index=False).agg(hlo=('hlo', 'mean'))
        latest = latest.groupby('code', as_index=False).agg(latest_period_hlo=('hlo', 'mean')).rename(columns={'code': 'iso3'})
        balanced = components.pivot(index='code', columns=['subject', 'level'], values='hlo')
        balanced['balanced_period_hlo'] = balanced.mean(axis=1).where(balanced.notna().sum(axis=1).eq(6))
        balanced = balanced['balanced_period_hlo'].rename_axis('iso3').reset_index()
        old = measurement.read_two_row_header(self.raw / 'niq_v133/NIQ-DATA (V1.3.3).xlsx', 'FAV')
        old = old.iloc[:, [0, 4]].copy()
        old.columns = ['iso3', 'niq_qnw_v133']
        old['niq_qnw_v133'] = pd.to_numeric(old['niq_qnw_v133'], errors='coerce')
        gdp = wdi_table(self.wdi_raw / 'wdi_NY.GDP.PCAP.PP.KD.zip')[['Country Code', '2017']]
        gdp.columns = ['iso3', 'gdp_pc_ppp_2017']
        population = wdi_table(self.wdi_raw / 'wdi_SP.POP.TOTL.zip')[['Country Code', '2017']]
        population.columns = ['iso3', 'population_2017']
        base = identities.merge(aggregate, on='iso3', how='left').merge(latest, on='iso3', how='left')
        base = base.merge(balanced, on='iso3', how='left').merge(fav[['iso3', 'niq_qnw', 'niq_qnw_sas', 'niq_qnw_sas_geo']], on='iso3', how='left')
        base = base.merge(old, on='iso3', how='left').merge(gdp, on='iso3', how='left').merge(population, on='iso3', how='left')
        base['log_gdp_pc_ppp_2017'] = np.log(base['gdp_pc_ppp_2017'])
        base['direct_qnw'] = base['niq_qnw'].notna() & ~(base['niq_qnw_sas'].isna() & base['niq_qnw_sas_geo'].notna())
        base['eligible_rq1'] = base['include_primary_reporting_economy'] & base['eligible_primary_hlo'].eq(True) & base['direct_qnw']
        base['eligible_rq2'] = base['eligible_rq1'] & base['gdp_pc_ppp_2017'].gt(0)
        self.base = base.sort_values('iso3').reset_index(drop=True)
        self.rq1 = self.base[self.base['eligible_rq1']].copy()
        self.rq2 = self.base[self.base['eligible_rq2']].copy()
        assert len(self.rq1) == 66 and len(self.rq2) == 65
        expected_ids = pd.read_csv(ROOT / 'sample_identities/core_countries.csv')
        assert set(self.rq1.iso3) == set(expected_ids.loc[expected_ids.rq1, 'iso3'])
        assert set(self.rq2.iso3) == set(expected_ids.loc[expected_ids.rq2, 'iso3'])
        csv(self.rq1, measurement.RQ1_FILE)
        csv(self.rq2, measurement.RQ2_FILE)
        self.core_results, reconciliation = measurement.reproduce_core()
        expected = pd.read_csv(ROOT / 'runtime_payload/candidate/original/data/core/core_results.csv')
        self.compare_frame(self.core_results, ROOT / 'runtime_payload/candidate/original/data/core/core_results.csv', ['result_id'])
        self.grid, self.windows, self.schemes = measurement.build_temporal(samples, hlo, nat)
        print(f'Reconstructed {len(self.rq1)} core economies, {len(self.rq2)} GDP economies and temporal cells in {perf_counter() - started:.1f}s.', flush=True)
        self.reconstruct_prior()

    def reconstruct_prior(self):
        warne = measurement.read_two_row_header(self.raw / 'warne.xlsx', 'National scores')
        iso = warne.columns[0]
        hlo = warne.columns[83]
        ghw = warne.columns[85]
        niqb = warne.columns[8]
        warne[hlo] = pd.to_numeric(warne[hlo], errors='coerce')
        warne[ghw] = pd.to_numeric(warne[ghw], errors='coerce')
        specifications = ['qnw', 'sas', 'qnw_plus_sas', 'qnw_plus_sas_plus_geo', 'geo_iq_only']
        rows = []
        for suffix in specifications:
            column = 'niq_values_from_lynn_and_becker__' + suffix
            pair = warne[[hlo, column]].apply(pd.to_numeric, errors='coerce').dropna()
            rows.append(dict(source='Warne', pair=suffix, r=pair[hlo].corr(pair[column]), n=len(pair)))
        henss = warne[[iso, niqb, hlo, ghw]].copy()
        for column in [niqb, hlo, ghw]:
            henss[column] = pd.to_numeric(henss[column], errors='coerce')
        henss.loc[henss[iso].eq('KHM'), niqb] = np.nan
        henss.loc[henss[iso].isin(['CUB', 'PAK']), [hlo, ghw]] = np.nan
        for left, right, label in [(niqb, hlo, 'NIQB / HLO IQ'), (niqb, ghw, 'NIQB / GHW IQ'), (hlo, ghw, 'HLO IQ / GHW IQ')]:
            pair = henss[[left, right]].dropna()
            rows.append(dict(source='Henss', pair=label, r=pair[left].corr(pair[right]), n=len(pair)))
        self.prior = pd.DataFrame(rows)
        csv(self.prior, self.output / 'PRIOR_RECONSTRUCTION.csv')
        assert self.prior.n.tolist() == [115, 100, 133, 161, 28, 158, 153, 148]
        expected = pd.read_csv(ROOT / 'tables/Table_2.csv', dtype=str)
        expected['Reconstructed r (N)'] = [f'{x.r:.3f}'.replace('0.', '.') + f' ({x.n})' for x in self.prior.itertuples()]
        self.assert_table(expected, 'Table_2.csv')

    def build_context(self):
        started = perf_counter()
        rows = []
        for code, name in context.FEATURES.items():
            frame = wdi_table(self.wdi_raw / f'wdi_{code}.zip')
            frame = frame[frame['Country Code'].isin(self.rq2.iso3)]
            frame = frame[['Country Code', *map(str, range(1980, 2018))]].melt(id_vars='Country Code', var_name='year', value_name='value')
            frame = frame.rename(columns={'Country Code': 'iso3'})
            frame['year'] = frame['year'].astype(int)
            frame['indicator_code'] = code
            frame['indicator_name'] = name
            rows.append(frame)
        self.wdi = pd.concat(rows, ignore_index=True)
        csv(self.wdi, self.work / 'data/processed/wdi_bridge_1980_2017_long.csv')
        block = context.build_block_ledger(self.wdi)
        block_coverage, _, _ = context.summarize_ledger(block, assessment_specific=False)
        eligible = pd.read_csv(self.temporal / 'temporal_eligible_cells.csv')
        keys = eligible[eligible['scheme'].eq('5_year')][['iso3', 'window_start', 'window_end']].drop_duplicates()
        aligned = block_coverage.merge(keys, on=['iso3', 'window_start', 'window_end'])
        csv(aligned, self.context / 'aligned_cell_block_preceding_coverage_long.csv')
        schedule = context.load_assessment_schedule(self.temporal / 'p_record_contribution_ledger.csv', self.temporal / 'hlo_record_contribution_ledger.csv', self.temporal / 'temporal_eligible_cells.csv')
        assessment = context.build_assessment_ledger(self.wdi, schedule)
        assessment_coverage, _, _ = context.summarize_ledger(assessment, assessment_specific=True)
        csv(assessment, self.context / 'assessment_specific_annual_provenance_ledger.csv')
        csv(assessment_coverage, self.context / 'assessment_specific_coverage_long.csv')
        context_manifest.ROOT = self.work
        context_manifest.CONTEXT = self.context
        context_manifest.TEMPORAL = self.temporal
        five, four = context_manifest.load_cells()
        block_features = context_manifest.prepare_block_features()
        assessment_features, contributions = context_manifest.prepare_assessment_features()
        four_features = context_manifest.build_four_year_features(four)
        cells, features = context_manifest.build_scenarios(five, four, block_features, assessment_features, four_features)
        cells['retained'] = cells['included']
        csv(cells, self.protocol / 'scenario_sample_manifest.csv')
        csv(features, self.protocol / 'scenario_feature_manifest.csv')
        self.bundle = build_modelling_matrices(self.protocol / 'scenario_sample_manifest.csv', self.protocol / 'scenario_feature_manifest.csv')
        self.jobs = build_execution_plan(self.bundle)
        primary = self.bundle.matrices[('PRIMARY', 'HLO_to_P', 'primary_5y')]
        self.compare_frame(self.bundle.manifest, ROOT / 'runtime_payload/expected_empirical/matrix_manifest.csv', ['scenario_id', 'target_direction', 'construction_role'])
        plan = pd.DataFrame([asdict(job) for job in self.jobs])
        self.compare_frame(plan, ROOT / 'runtime_payload/expected_empirical/execution_plan.csv', ['job_id'])
        csv(self.bundle.manifest, self.run / 'matrix_manifest.csv')
        csv(plan, self.run / 'execution_plan.csv')
        self.make_temporal_table()
        print(f'Built {len(self.bundle.matrices)} modelling matrices and {len(self.jobs)} jobs; primary sample {len(primary)} cells / {primary.iso3.nunique()} countries; {perf_counter() - started:.1f}s.', flush=True)

    def fit(self):
        started = perf_counter()
        digest = plan_hash(self.jobs)
        print(f'Running {len(self.jobs)} jobs with the original nested country-holdout method. Previous archived-data run: approximately 22 minutes.', flush=True)
        with threadpool_limits(limits=1):
            for number, job in enumerate(self.jobs, 1):
                result = run_job(_matrix_for_job(self.bundle, job), job)
                assert result['complete'], f'Incomplete prediction job: {job.job_id}'
                _write_job(self.run, job, result, digest)
                elapsed = perf_counter() - started
                remaining = elapsed / number * (len(self.jobs) - number)
                print(f'{number}/{len(self.jobs)} {job.job_id}: elapsed {elapsed / 60:.1f} min; estimated remaining {remaining / 60:.1f} min.', flush=True)
        assemble_ledgers(self.run, self.jobs, digest)
        print(f'All {len(self.jobs)} jobs completed in {(perf_counter() - started) / 60:.1f} minutes.', flush=True)

    def gdp_results(self):
        core = self.core_results.set_index('result_id').estimate
        inference = load_module('paired_bootstrap', ROOT / 'runtime_payload/candidate/original/code/build_postreview_empirical_inference.py')
        archived = pd.read_csv(ROOT / 'tables/archived/headline_analysis_correlation_difference.csv').iloc[0]
        bootstrap = inference.PairedCorrelationBootstrap.__new__(inference.PairedCorrelationBootstrap)
        bootstrap.sample = self.rq2
        bootstrap.values = self.rq2[inference.VARIABLES].to_numpy(dtype=float)
        assert len(self.rq2) == 65 and np.isfinite(bootstrap.values).all()
        bootstrap.observed = bootstrap.correlations(bootstrap.values)
        distribution = bootstrap.distribution()
        self.bootstrap = bootstrap.summary(distribution, bootstrap.jackknife())
        (self.output / 'GDP_BOOTSTRAP.json').write_text(json.dumps(self.bootstrap, indent=2), encoding='utf-8')
        csv(distribution, self.output / 'GDP_BOOTSTRAP_DISTRIBUTION.csv')
        table = pd.read_csv(ROOT / 'tables/archived/T_GDP_CONTRAST_source.csv')
        table[['estimate', 'lower', 'upper', 'n_economies']] = np.nan
        primary = [core.CORE_QNW_GDP_PEARSON, core.CORE_HLO_GDP_PEARSON,
                   core.CORE_GDP_CORRELATION_DIFFERENCE_QNW_MINUS_HLO,
                   core.CORE_GDP_CORRELATION_DIFFERENCE_QNW_MINUS_HLO,
                   core.CORE_GDP_CORRELATION_DIFFERENCE_QNW_MINUS_HLO, core.CORE_WILLIAMS_P]
        table.loc[:5, 'estimate'] = primary
        table.loc[:5, 'n_economies'] = 65
        table.loc[2, ['lower', 'upper']] = [core.CORE_ZOU_LOWER, core.CORE_ZOU_UPPER]
        table.loc[3, ['lower', 'upper']] = [self.bootstrap['percentile_ci_95_lower'], self.bootstrap['percentile_ci_95_upper']]
        table.loc[4, ['lower', 'upper']] = [self.bootstrap['bca_ci_95_lower'], self.bootstrap['bca_ci_95_upper']]
        variants = [
            (6, 'niq_qnw', 'hlo', self.base.eligible_rq2, True),
            (7, 'niq_qnw_v133', 'hlo', self.base.eligible_rq2, True),
            (8, 'niq_qnw', 'latest_period_hlo', self.base.eligible_rq2, True),
            (9, 'niq_qnw', 'hlo', self.base.eligible_rq2 & self.base.include_sensitivity_un_member_observer, False),
            (10, 'niq_qnw_sas', 'hlo', self.base.include_primary_reporting_economy & self.base.eligible_primary_hlo.eq(True), False),
            (11, 'niq_qnw_sas_geo', 'hlo', self.base.include_primary_reporting_economy & self.base.eligible_primary_hlo.eq(True), False),
            (12, 'niq_qnw', 'balanced_period_hlo', self.base.include_primary_reporting_economy & self.base.direct_qnw, False),
            (13, 'niq_qnw', 'hlo', self.base.include_primary_reporting_economy & self.base.direct_qnw, False),
            (14, 'niq_qnw', 'hlo', self.base.eligible_rq2, True),
        ]
        for number, qnw, hlo, mask, fixed in variants:
            frame = self.base[mask & self.base.gdp_pc_ppp_2017.gt(0)].dropna(subset=[qnw, hlo])
            table.loc[number, ['estimate', 'n_economies']] = [frame[qnw].corr(frame.log_gdp_pc_ppp_2017) - frame[hlo].corr(frame.log_gdp_pc_ppp_2017), len(frame)]
        frame = self.rq2[self.rq2.population_2017.gt(0)]
        def weighted_r(values):
            weights = frame.population_2017.to_numpy()
            x, y = values.to_numpy(), frame.log_gdp_pc_ppp_2017.to_numpy()
            x = x - np.average(x, weights=weights)
            y = y - np.average(y, weights=weights)
            return np.average(x * y, weights=weights) / np.sqrt(np.average(x ** 2, weights=weights) * np.average(y ** 2, weights=weights))
        table.loc[15, ['estimate', 'n_economies']] = [weighted_r(frame.niq_qnw) - weighted_r(frame.hlo), len(frame)]
        self.compare_frame(table, ROOT / 'tables/archived/T_GDP_CONTRAST_source.csv', ['section', 'scenario', 'interval_or_test'])
        csv(table, self.tables / 'Table_S7.csv')
        self.checks.append(dict(check='GDP_contrasts_and_bootstrap_generated', comparisons=len(table), status='PASS'))

    def make_temporal_table(self):
        expected = pd.read_csv(ROOT / 'tables/archived/T_TEMPORAL_COVERAGE_source.csv').iloc[:, :6]
        table = expected.copy()
        table[['candidate_cells', 'aligned_eligible_cells', 'countries']] = np.nan
        for i, row in expected.iterrows():
            scheme = row.scheme_or_construction
            if row.section == 'Temporal scheme':
                result = self.schemes[self.schemes.scheme.eq(scheme)].iloc[0]
                table.loc[i, ['candidate_cells', 'aligned_eligible_cells', 'countries']] = [result.country_window_cells, result.aligned_eligible_cells, result.countries_with_aligned_cell]
            elif row.section == 'Five-year window':
                result = self.windows[self.windows.scheme.eq('5_year') & self.windows.window_label.eq(row.period)].iloc[0]
                table.loc[i, ['candidate_cells', 'aligned_eligible_cells']] = [result.countries, result.aligned_eligible_cells]
            elif row.section == 'Fractional-date assignment sensitivity':
                result = self.schemes[self.schemes.scheme.eq(scheme)].iloc[0]
                table.loc[i, ['candidate_cells', 'aligned_eligible_cells']] = [result.country_window_cells, result.aligned_eligible_cells]
            else:
                table.loc[i, 'candidate_cells'] = int((self.grid.scheme.eq('5_year') & self.grid.eligible_aligned_cell).sum())
        self.compare_frame(table, ROOT / 'tables/archived/T_TEMPORAL_COVERAGE_source.csv', ['section', 'scheme_or_construction', 'period'], columns=expected.columns)
        csv(table, self.tables / 'Table_S8.csv')

    def report(self):
        predictions = pd.read_csv(self.run / 'PREDICTIONS.csv')
        plan = pd.read_csv(self.run / 'execution_plan.csv')
        status = json.loads((self.run / 'RUN_STATUS.json').read_text())
        csv(_metrics(plan, predictions), self.run / 'METRICS.csv')
        calibration, ranks, tails = _reporting_ledgers(plan, predictions)
        for name, frame in [('CALIBRATION.csv', calibration), ('RANK_LEDGER.csv', ranks), ('TAIL_GROUPS.csv', tails),
                            ('PAIRED_COMPLETENESS.csv', _paired(plan, status))]:
            csv(frame, self.run / name)
        results, sources = compute_results(predictions, 'download_reconstruction', sha256((self.run / 'PREDICTIONS.csv').read_bytes()).hexdigest())
        for name, frame in sources.items():
            csv(frame, self.output / 'reporting_sources' / name)
        metrics = pd.read_csv(self.run / 'METRICS.csv')
        for name, keys in [
            ('METRICS.csv', ['job_id']), ('SPLITS.csv', ['job_id', 'fit_id', 'inner_country']),
            ('TUNING.csv', ['job_id', 'fit_id', 'alpha']), ('CALIBRATION.csv', ['job_id', 'scope']),
            ('RANK_LEDGER.csv', ['job_id', 'cell_id']),
            ('PAIRED_COMPLETENESS.csv', ['scenario_id', 'direction', 'construction_role', 'deleted_country', 'estimator']),
        ]:
            fresh = pd.read_csv(self.run / name)
            self.compare_frame(fresh, ROOT / 'runtime_payload/expected_empirical' / name, keys)
        for name, file in [('PHASE_B_RANK_SUMMARY_SOURCE.csv', 'PHASE_B_RANK_SUMMARY_SOURCE.csv'), ('PHASE_B_TAIL_SOURCE.csv', 'TAIL_SUMMARIES.csv')]:
            fresh = sources[name]
            expected = pd.read_csv(ROOT / 'tables/archived' / file)
            self.compare_frame(fresh[expected.columns], ROOT / 'tables/archived' / file, ['job_id'] + (['window_start', 'tail'] if 'tail' in expected else []))
        primary = metrics[metrics.scenario_id.eq('PRIMARY')].copy()
        csv(primary.round(3), self.tables / 'Table_S1.csv')
        csv(metrics[~metrics.scenario_id.isin(['PRIMARY', 'PRIMARY_DELETE'])].round(3), self.tables / 'Table_S2.csv')
        deletion = metrics[metrics.scenario_id.eq('PRIMARY_DELETE')].pivot(index='deleted_country', columns='model_id', values='mse')
        deletion['M1_minus_M3'] = deletion.M1 - deletion.M3
        deletion['M2_minus_M3'] = deletion.M2 - deletion.M3
        csv(deletion.reset_index().round(3), self.tables / 'Table_S3.csv')
        ranks = sources['PHASE_B_RANK_SUMMARY_SOURCE.csv']
        csv(ranks[ranks.scenario_id.eq('PRIMARY')].round(6), self.tables / 'Table_S4.csv')
        calibration = pd.read_csv(self.run / 'CALIBRATION.csv')
        csv(calibration[calibration.scenario_id.eq('PRIMARY') & calibration.estimator.isin(['ridge', 'reference'])].round(3), self.tables / 'Table_S5.csv')
        tails = sources['PHASE_B_TAIL_SOURCE.csv']
        primary_ridge = set(primary.loc[primary.estimator.eq('ridge'), 'job_id'])
        csv(tails[tails.job_id.isin(primary_ridge)].round(6), self.tables / 'Table_S6.csv')
        references = primary[primary.model_id.eq('M0')].set_index('direction').mse
        primary['reference_mse'] = primary.direction.map(references)
        primary['normalized_mse'] = primary.mse / primary.reference_mse
        primary['mse_improvement_vs_m0'] = 1 - primary.normalized_mse
        csv(primary, self.figures / 'INTELLIGENCE_Figure_2_data.csv')
        csv(pd.read_csv(self.core / 'core_rank_ledger.csv'), self.figures / 'INTELLIGENCE_Figure_1_data.csv')
        self.main_tables(primary)
        plots = load_module('empirical_plots', ROOT / 'figures/plot_figures.py')
        plots.ROOT = self.figures
        plots.ranks()
        plots.performance()
        shutil.copyfile(ROOT / 'figures/METHODOLOGY_Workflow.svg', self.figures / 'METHODOLOGY_Workflow.svg')
        self.checks.append(dict(check='main_table_display_cells', comparisons=4, status='GENERATED'))
        self.checks.append(dict(check='figures_generated_from_refitted_results', comparisons=3, status='PASS'))
        csv(pd.DataFrame(self.checks), self.output / 'VALIDATION.csv')
        primary_matrix = self.bundle.matrices[('PRIMARY', 'HLO_to_P', 'primary_5y')]
        summary = dict(status='COMPLETED', data_version=self.data_version,
                       source='public provider downloads; paper mode uses attributed World Bank snapshots',
                       core_economies=len(self.rq1), GDP_economies=len(self.rq2), primary_cells=len(primary_matrix), primary_countries=int(primary_matrix.iso3.nunique()),
                       matrices=len(self.bundle.matrices), fitted_jobs=len(self.jobs), main_tables=4, supplementary_tables=8, figures=3,
                       elapsed_minutes=(perf_counter() - self.started) / 60, checks=self.checks)
        (self.output / 'VALIDATION.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)

    def main_tables(self, primary):
        shutil.copyfile(ROOT / 'tables/Table_1.csv', self.tables / 'Table_1.csv')
        v = self.core_results.set_index('result_id').estimate
        def coefficient(x):
            return ('\\textminus{}' if x < 0 else '') + f'{abs(x):.3f}'.removeprefix('0')
        table = pd.read_csv(ROOT / 'tables/Table_3.csv', dtype=str)
        table['Result'] = [
            coefficient(v.CORE_QNW_HLO_PEARSON), coefficient(v.CORE_QNW_HLO_SPEARMAN),
            f'{v.CORE_RANK_MEAN_ABS:.2f} / {v.CORE_RANK_MEDIAN_ABS:g} / {v.CORE_RANK_MAX_ABS:g} places',
            f'{int(v.CORE_ORDER_REVERSALS):,} / {int(v.CORE_STRICT_PAIRS):,}',
            f'{int(v.CORE_TOP_OVERLAP)} / {int(v.CORE_TAIL_DENOMINATOR)}; {int(v.CORE_BOTTOM_OVERLAP)} / {int(v.CORE_TAIL_DENOMINATOR)}',
            coefficient(v.CORE_QNW_GDP_PEARSON), coefficient(v.CORE_HLO_GDP_PEARSON),
            coefficient(v.CORE_GDP_CORRELATION_DIFFERENCE_QNW_MINUS_HLO), coefficient(v.CORE_WILLIAMS_P),
            f'[{coefficient(v.CORE_ZOU_LOWER)}, {coefficient(v.CORE_ZOU_UPPER)}]',
            f'[{coefficient(self.bootstrap["bca_ci_95_lower"])}, {coefficient(self.bootstrap["bca_ci_95_upper"])}]',
        ]
        self.assert_table(table, 'Table_3.csv')
        table = pd.read_csv(ROOT / 'tables/Table_4.csv', dtype=str)
        for index, model in enumerate(['M0', 'M1', 'M2', 'M3']):
            for direction, value_column, reduction_column in [
                ('HLO_to_P', table.columns[1], table.columns[2]),
                ('P_to_HLO', table.columns[3], table.columns[4]),
            ]:
                row = primary[primary.direction.eq(direction) & primary.model_id.eq(model) & primary.estimator.isin(['ridge', 'reference'])].iloc[0]
                table.loc[index, value_column] = f'{row.mse:,.3f}'
                table.loc[index, reduction_column] = '—' if model == 'M0' else f'{100 * row.mse_improvement_vs_m0:.1f}%'
        self.assert_table(table, 'Table_4.csv')

    def assert_table(self, table, name):
        expected = pd.read_csv(ROOT / 'tables' / name, dtype=str)
        exact = table.equals(expected)
        self.checks.append(dict(check=name, comparisons=table.size, status='MATCH' if exact else 'UPDATED_DATA_DIFFERENCE'))
        csv(table, self.tables / name)

    def compare_frame(self, fresh, path, keys, columns=None):
        expected = pd.read_csv(path)
        if columns is not None:
            expected = expected[list(columns)]
        fresh = fresh[expected.columns].copy()
        for column in expected.columns:
            if expected[column].dtype == object or fresh[column].dtype == object:
                expected[column] = expected[column].fillna('').astype(str)
                fresh[column] = fresh[column].fillna('').astype(str)
        fresh = fresh.sort_values(keys, kind='mergesort').reset_index(drop=True)
        expected = expected.sort_values(keys, kind='mergesort').reset_index(drop=True)
        numeric = expected.select_dtypes(include='number').columns
        same_ids = fresh[keys].equals(expected[keys])
        paired = fresh.merge(expected, on=keys, suffixes=('_fresh', '_paper'))
        changes = []
        for column in numeric:
            if column in keys:
                continue
            a = paired[column + '_fresh'].to_numpy(float)
            b = paired[column + '_paper'].to_numpy(float)
            finite = np.isfinite(a) & np.isfinite(b)
            difference = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
            changes.append(dict(table=path.name, column=column, maximum_absolute_difference=difference,
                                differing_categories=0,
                                changed_missingness=int((np.isfinite(a) != np.isfinite(b)).sum()),
                                identical_row_identities=same_ids, fresh_rows=len(fresh), paper_rows=len(expected)))
        for column in expected.columns.difference(numeric).difference(keys):
            a = paired[column + '_fresh'].fillna('').astype(str)
            b = paired[column + '_paper'].fillna('').astype(str)
            changes.append(dict(table=path.name, column=column, maximum_absolute_difference=0.0,
                                differing_categories=int(a.ne(b).sum()), changed_missingness=0,
                                identical_row_identities=same_ids, fresh_rows=len(fresh), paper_rows=len(expected)))
        if changes:
            csv(pd.DataFrame(changes), self.output / 'comparisons' / path.name)
        matches = same_ids and all(x['maximum_absolute_difference'] <= 1e-10 and x['changed_missingness'] == 0 and x['differing_categories'] == 0 for x in changes)
        self.checks.append(dict(check=path.name, comparisons=len(fresh), status='MATCH' if matches else 'DIFFERENCE_REPORTED'))
