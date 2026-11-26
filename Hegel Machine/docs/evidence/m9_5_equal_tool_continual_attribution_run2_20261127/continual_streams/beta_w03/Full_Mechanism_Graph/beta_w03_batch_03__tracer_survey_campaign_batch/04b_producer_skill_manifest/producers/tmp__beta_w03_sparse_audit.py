import importlib.util, json, pickle, sys
from pathlib import Path
import numpy as np

def load(path):
    spec = importlib.util.spec_from_file_location('solver',path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

root=Path('/holdout_task')
base=load(root/'beta_w03_batch_03_solver.py')
cand=load(sys.argv[1])
ns={}; exec((root/'validate_public_schema.py').read_text().split('\n_d =')[0],ns)
verify=ns['verify_box_simplex_kkt_certificate']
a=pickle.loads((root/'public_beta_w03_batch_03.pkl').read_bytes())[0]
s,g=a['study'],a['gt']; rows=s['row_bank_id']; y=base._calibrate(s)
history=json.loads(Path('/tmp/hegel_continual_archive/archive_calibration_history.json').read_text())
metrics={}

def run(label, study):
    out=cand.solve_beta_w03_batch_03(study)
    w=max(np.linalg.norm(u-v)/np.linalg.norm(v) for u,v in zip(out['weights_by_budget'],g['weights_by_budget'],strict=True))
    f=np.linalg.norm(out['terminal_field_mid_budget']-g['terminal_field_mid_budget'])/np.linalg.norm(g['terminal_field_mid_budget'])
    checks=[verify(g['S_tilde'],g['r_y'],s['source_ridge'],s['u_box'],b,c) for b,c in zip(s['budget_grid'],out['kkt_certificates'],strict=True)]
    reasons={x['reason'].split(':')[0] for x in checks if not x['valid']}
    prev=metrics.get(label,(0.,0.,0.,set()))
    metrics[label]=(max(w,prev[0]),max(f,prev[1]),max([prev[2]]+[x['gap'] for x in checks if x['valid']]),prev[3]|reasons)

run('public',s)
for h in history:
    ids,x,r=map(np.asarray,(h['bank_id'],h['ref_signal'],h['raw_readout']))
    p=np.array([np.polyfit(x[ids==b],r[ids==b],1) for b in range(4)])
    raw=y*p[rows,0]+p[rows,1]
    for wing in (0,1):
        for n in (0,1,2,3,12):
            for spike in ((False,True) if n else (False,)):
                for placement in ((0,-1) if spike and n>1 else (0,)):
                    mask=ids//2!=wing; qraw=r.copy()
                    for b in range(2*wing,2*wing+2):
                        j=np.flatnonzero(ids==b); j=j[np.argsort(x[j])]; k=j[np.linspace(0,len(j)-1,n,dtype=int)]; mask[k]=True
                        if spike: qraw[k[placement]] += (0.6 if b%2==0 else -0.6)
                    u=dict(s,y_obs_raw=raw,sensor_calibration_sweeps=dict(bank_id=ids[mask],ref_signal=x[mask],raw_readout=qraw[mask])); run((n,spike),u)
for k,v in metrics.items(): print(k,'weights rel, field rel, max passed KKT gap, strict failures:',v)
