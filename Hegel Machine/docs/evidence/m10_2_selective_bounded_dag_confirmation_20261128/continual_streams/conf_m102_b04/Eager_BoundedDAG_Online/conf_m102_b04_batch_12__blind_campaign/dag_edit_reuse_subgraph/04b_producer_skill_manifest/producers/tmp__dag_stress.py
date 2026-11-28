import importlib.util, pickle, pathlib, subprocess, os
import numpy as np
p=pathlib.Path('/holdout_task/node_01.py'); base=p.read_text(); opts=pathlib.Path('/tmp/scalar_options.txt').read_text()
def load(path):
    spec=importlib.util.spec_from_file_location(path.stem,path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
m=load(p); frozen=load(p.parent/'_frozen_nodes/node_00.py'); exec(opts,m.__dict__)
rows=[]
for b in pickle.load(open(p.parent/'public_batch.pkl','rb')):
    st=b['study']; tr=st['train']; n=len(tr['x']); ids=np.arange(n); coef, prev=m._calibration(st['panel_a']); split=max(40,n-20)
    folds=[ids[split:], *np.array_split(ids,4), *[ids[i::4] for i in range(4)], *[c[-max(10,n//5):] for c in (ids[np.argsort(abs(tr['x'][:,i]))] for i in range(tr['x'].shape[1]))]]
    def inverse(x, st):
        F=frozen.run_node({'__input__':st}); d=x.shape[1]; start=F['hybrid'][:,-((d+1)*(d+2)*(d+3)//6):][:,1:d+1];return m._invert(x,start,coef,prev)
    for i,te in enumerate(folds):
        ti=np.setdiff1d(ids,te); y=tr['y'].reshape(-1); x=tr['x'][np.r_[ti,te]]; study=dict(st, train=dict(tr,x=tr['x'][ti], y=tr['y'][ti],traj=tr['traj'][ti]), test={'x':tr['x'][te]}); z=inverse(x,study); nn=len(ti); meta=st['telemetry_vec']; a=m._scalar(z,y[ti],nn,meta); v=m._scalar_vp(z,y[ti],nn,meta)
        rows.append([np.mean(abs(a-y[te])),np.mean(abs(v-y[te])),np.mean(abs(.5*(a+v)-y[te])),np.mean(abs((.25*a+.75*v)-y[te])),np.mean(abs(m._trajectory(z,tr['traj'][ti],nn)-tr['traj'][te]))]); print(i,np.round(rows[-1],6),flush=True)
a=np.array(rows); print('columns: baseline, projection, equal blend, projection-weighted blend, trajectory');print('worst standard',a[:9].max(0),'worst including coordinate tails',a.max(0),'mean',a.mean(0))
choices=[i for i in (1,2,3) if a[0,i]<.038 and a[:9,i].max() < min(.038,.9*a[:9,0].max()) and a[:,i].max() < .95*a[:,0].max() and a[:,i].mean() < a[:,0].mean()]
if choices:
    j=min(choices,key=lambda i:(a[:,i].max(),a[:,i].mean())); expression={1:'_scalar_vp(z, target, n, telemetry)',2:'.5 * _scalar(z, target, n, telemetry) + .5 * _scalar_vp(z, target, n, telemetry)',3:'.25 * _scalar(z, target, n, telemetry) + .75 * _scalar_vp(z, target, n, telemetry)'}[j]
    text=base.replace('\ndef run_node(', '\n'+opts+'\n\ndef run_node(').replace("    y = _scalar(z, np.asarray(train['y']).reshape(n), n, np.asarray(study['telemetry_vec']))", "    target = np.asarray(train['y']).reshape(n)\n    telemetry = np.asarray(study['telemetry_vec'])\n    y = "+expression)
    p.write_text(text); print('PROMOTED',j,flush=True)
try:
    for flags in ([],['-I']):
        result=subprocess.check_output(['python3',*flags,str(p.parent/'test_dag_candidate.py')],cwd=p.parent,text=True); print(result,end='');import json; assert json.loads(result)['accepted']
except Exception:
    if not choices: raise
    p.write_text(base); print('RESTORED BASELINE'); subprocess.run(['python3','-I',str(p.parent/'test_dag_candidate.py')],cwd=p.parent,check=True)
