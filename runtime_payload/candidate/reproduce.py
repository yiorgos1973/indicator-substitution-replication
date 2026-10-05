"""Reproduce reported comparisons from frozen analysis inputs.

The wrapper calls the archived numerical routines without changing model rules.
It checks the input/code manifest and compares against archived results.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, platform, sys, time

ROOT = Path(__file__).resolve().parent
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name] = '1'
sys.path[:0] = [str(ROOT/'original/code/src'), str(ROOT/'original/code')]
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata
from threadpoolctl import threadpool_limits
from phase_b.engine import run_job
from phase_b.plan import build_execution_plan
from phase_b.matrices import MatrixBundle, _validate_matched_pairs
from phase_b.gate import EXPECTED_PROTOCOL_HASH, canonical_protocol_hash
from regenerate_primary_headline import williams_and_zou
from build_postreview_empirical_inference import PairedCorrelationBootstrap

def check_manifest():
    rows = list(csv.DictReader((ROOT/'MANIFEST.csv').open()))
    for row in rows:
        p = (ROOT/row['path']).resolve()
        if ROOT not in p.parents or not p.is_file():
            raise ValueError('Missing or invalid path: '+row['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Changed file: '+row['path'])
    protocol=json.loads((ROOT/'original/protocol/protocol_for_author_approval.json').read_text())
    if canonical_protocol_hash(protocol) != EXPECTED_PROTOCOL_HASH:
        raise ValueError('Protocol hash mismatch')
    prior=json.loads((ROOT/'original/manifests/VALIDATION_ATTESTATION.json').read_text())
    if prior['status'] != 'PASS' or prior['protocol_hash'] != EXPECTED_PROTOCOL_HASH:
        raise ValueError('Original validation attestation mismatch')
    return len(rows)

def core_checks(output):
    base=ROOT/'original/data/core'
    a=pd.read_csv(base/'core_rq1_exact_sample.csv')
    b=pd.read_csv(base/'core_rq2_exact_sample.csv')
    expected=pd.read_csv(base/'core_results.csv').set_index('result_id')['estimate'].to_dict()
    x,y=a.niq_qnw.to_numpy(),a.hlo.to_numpy()
    disp=np.abs(rankdata(-x)-rankdata(-y))
    dx,dy=x[:,None]-x,y[:,None]-y
    strict=np.triu((dx!=0)&(dy!=0),1)
    reverse=int(np.sum(strict&(dx*dy<0)))
    topx,topy=x>=np.quantile(x,.9),y>=np.quantile(y,.9)
    botx,boty=x<=np.quantile(x,.1),y<=np.quantile(y,.1)
    rq,rh=pearsonr(b.niq_qnw,b.log_gdp_pc_ppp_2017).statistic,pearsonr(b.hlo,b.log_gdp_pc_ppp_2017).statistic
    _,wt,wp,delta,zlo,zhi=williams_and_zou(rq,rh,pearsonr(b.niq_qnw,b.hlo).statistic,len(b))
    values={'CORE_RQ1_N':len(a),'CORE_QNW_HLO_PEARSON':pearsonr(x,y).statistic,'CORE_QNW_HLO_SPEARMAN':spearmanr(x,y).statistic,'CORE_QNW_HLO_KENDALL':kendalltau(x,y).statistic,'CORE_RANK_MEAN_ABS':disp.mean(),'CORE_RANK_MEDIAN_ABS':np.median(disp),'CORE_RANK_MAX_ABS':disp.max(),'CORE_RANK_MEAN_NORMALIZED':disp.mean()/(len(a)-1),'CORE_STRICT_PAIRS':strict.sum(),'CORE_ORDER_REVERSALS':reverse,'CORE_ORDER_REVERSAL_SHARE':reverse/strict.sum(),'CORE_TOP_OVERLAP':np.sum(topx&topy),'CORE_BOTTOM_OVERLAP':np.sum(botx&boty),'CORE_TAIL_DENOMINATOR':topx.sum(),'CORE_RQ2_N':len(b),'CORE_QNW_GDP_PEARSON':rq,'CORE_HLO_GDP_PEARSON':rh,'CORE_GDP_CORRELATION_DIFFERENCE_QNW_MINUS_HLO':delta,'CORE_WILLIAMS_T':wt,'CORE_WILLIAMS_P':wp,'CORE_ZOU_LOWER':zlo,'CORE_ZOU_UPPER':zhi}
    if not (a.iso3.is_unique and b.iso3.is_unique and set(a.iso3)-set(b.iso3)=={'TWN'}):
        raise ValueError('Core sample identity mismatch')
    rows=[]
    for k,v in values.items():
        np.testing.assert_allclose(v,expected[k],atol=1e-10,rtol=1e-8)
        rows.append({'result':k,'rebuilt':v,'archived':expected[k],'status':'PASS'})
    archived=pd.Series({'common_sample_iso3':' '.join(b.iso3),'r_qnw_gdp':rq,'r_hlo_gdp':rh,'difference_r_qnw_gdp_minus_r_hlo_gdp':delta})
    bootstrap=PairedCorrelationBootstrap(b,archived)
    summary=bootstrap.summary(bootstrap.distribution(),bootstrap.jackknife())
    gdp=pd.read_csv(ROOT/'original/tables/T_GDP_CONTRAST_source.csv')
    row=gdp[gdp.interval_or_test.eq('Paired economy bootstrap BCa 95%')].iloc[0]
    for name,ex in [('bca_ci_95_lower',row.lower),('bca_ci_95_upper',row.upper)]:
        np.testing.assert_allclose(summary[name],ex,atol=1e-10,rtol=1e-8)
        rows.append({'result':name,'rebuilt':summary[name],'archived':ex,'status':'PASS'})
    pd.DataFrame(rows).to_csv(output/'CORE_COMPARISON.csv',index=False)
    (output/'BOOTSTRAP_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    return len(rows)

def prediction_checks(output,full):
    frame=pd.read_csv(ROOT/'original/data/prediction/model_matrices.csv')
    matrices={tuple(k):d.reset_index(drop=True) for k,d in frame.groupby(['scenario_id','target_direction','construction_role'],sort=True)}
    _validate_matched_pairs(matrices)
    primary=matrices[('PRIMARY','HLO_to_P','primary_5y')]
    assert len(primary)==53 and primary.iso3.nunique()==38
    jobs=build_execution_plan(MatrixBundle(matrices,pd.DataFrame()))
    expected=pd.read_csv(ROOT/'original/results/model_runs/phase_b_20260930_13fb5775_v2/METRICS.csv').set_index('job_id')
    assert set(j.job_id for j in jobs)==set(expected.index)
    if not full: jobs=[j for j in jobs if j.scenario_id=='PRIMARY']
    comparisons=[]; primary_predictions=[]
    with threadpool_limits(limits=1):
        for index,job in enumerate(jobs,1):
            identity=('PRIMARY' if job.scenario_id=='PRIMARY_DELETE' else job.scenario_id,job.direction,job.construction_role)
            matrix=matrices[identity]
            if job.deleted_country: matrix=matrix[matrix.iso3.ne(job.deleted_country)].reset_index(drop=True)
            result=run_job(matrix,job)
            if not result['complete']: raise ValueError('Incomplete job '+job.job_id)
            assert result['planned_cell_count']==len(matrix)==result['predicted_cell_count']
            for metric in ('mae','mse','rmse','bias_pred_minus_observed'):
                v=result['metrics'][metric]; ex=expected.loc[job.job_id,metric]
                np.testing.assert_allclose(v,ex,atol=1e-10,rtol=1e-8,err_msg=job.job_id+' '+metric)
                comparisons.append({'job_id':job.job_id,'metric':metric,'rebuilt':v,'archived':ex,'absolute_difference':abs(v-ex),'status':'PASS'})
            if job.scenario_id=='PRIMARY':
                primary_predictions.extend({'job_id':job.job_id,**row} for row in result['predictions'])
            if index%10==0 or index==len(jobs):
                print(f'Prediction jobs {index}/{len(jobs)} passed',flush=True)
    pd.DataFrame(comparisons).to_csv(output/'PREDICTION_COMPARISON.csv',index=False)
    pd.DataFrame(primary_predictions).to_csv(output/'PRIMARY_HELD_OUT_PREDICTIONS.csv',index=False)
    return len(jobs),len(comparisons)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true',help='Also rerun all 80 sensitivity and 152 country-deletion jobs.')
    parser.add_argument('--output',type=Path,default=ROOT/'reproduced')
    args=parser.parse_args(); start=time.perf_counter()
    count=check_manifest()
    output=args.output.resolve()
    if output.exists(): raise FileExistsError('Choose a new output directory: '+str(output))
    output.mkdir(parents=True)
    result={'status':'RUNNING','manifest_files_checked':count,'mode':'all_252_jobs' if args.full else 'primary_20_jobs','protocol_hash':EXPECTED_PROTOCOL_HASH}
    try:
        result['core_comparisons']=core_checks(output)
        result['prediction_jobs'],result['prediction_comparisons']=prediction_checks(output,args.full)
        result['status']='PASS'
    except Exception as error:
        result.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result.update(elapsed_seconds=time.perf_counter()-start,environment={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},absolute_tolerance=1e-10,relative_tolerance=1e-8)
        (output/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
