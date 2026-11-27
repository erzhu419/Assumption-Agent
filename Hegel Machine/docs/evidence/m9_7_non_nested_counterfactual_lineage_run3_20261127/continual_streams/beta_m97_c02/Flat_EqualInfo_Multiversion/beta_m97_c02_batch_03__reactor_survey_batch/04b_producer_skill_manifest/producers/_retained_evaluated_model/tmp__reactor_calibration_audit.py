import importlib.util, pickle
from pathlib import Path
import numpy as np
p=Path('/holdout_task')
spec=importlib.util.spec_from_file_location('solver',p/'beta_m97_c02_batch_03_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
v={}; exec((p/'validate_public_schema.py').read_text().split('\n_d =',1)[0],v)
check=v['verify_box_simplex_kkt_certificate']
c=pickle.load((p/'public_beta_m97_c02_batch_03.pkl').open('rb'))[0]; s,g=c['study'],c['gt']
f=m.solve_beta_m97_c02_batch_03; base=f(s)
sw={k:np.asarray(v) for k,v in s['sensor_calibration_sweeps'].items()}; b,x,y=(sw[k] for k in ('bank_id','ref_signal','raw_readout'))
metrics=[]
for i in range(len(b)):
    keep=np.arange(len(b))!=i; subset=keep & (b==b[i]); a,d=m._fit_line(x[subset],y[subset]); held=abs(a*x[i]+d-y[i]); t=dict(s,sensor_calibration_sweeps={k:v[keep] for k,v in sw.items()}); out=f(t)
    kkt=check(g['S_tilde'],g['r_y'],s['source_ridge'],s['u_box'],s['total_budget'],out['kkt_certificate'])
    metrics.append((kkt['gap'],int(b[i]),i,float(held),float(np.max(np.abs(out['weights']-g['w_opt']))),kkt['reason']))
print('Public leave-one-sweep-out worst gap per bank:'); [print(max(r for r in metrics if r[1]==i)) for i in range(4)]
print('Worst held-out raw calibration residual:',max(row[3] for row in metrics)); assert all(row[-1]=='ok' for row in metrics)
for mode in (0.,0.50,0.500001,0.85,1.):
    t=dict(s,recirculation_loop_mode=mode); design,fields=m._forward(t); z=s['Z_nuisance']; a=np.column_stack((m._calibrate(t),design)); a-=z@np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@a)
    out=f(t); report=check(a[:,1:],a[:,0],s['source_ridge'],s['u_box'],s['total_budget'],out['kkt_certificate']); assert report['valid'],(mode,report)
    np.testing.assert_allclose(out['terminal_field'],(fields@out['weights']).reshape(s['nx'],s['ny'],order='F')); print('Loop mode',mode,'KKT gap',report['gap'])
