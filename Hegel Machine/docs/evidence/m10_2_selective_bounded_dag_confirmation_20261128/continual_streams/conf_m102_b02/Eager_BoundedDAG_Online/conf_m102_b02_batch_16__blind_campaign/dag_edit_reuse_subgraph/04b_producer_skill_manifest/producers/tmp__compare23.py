import sys,pickle,re,runpy,numpy as np
sys.path.insert(0,'/tmp')
import challenger23 as m
import base23 as old
from pathlib import Path
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['train']; N=len(a['y']); i=np.arange(N)
text=Path('/holdout_task/test_dag_candidate.py').read_text(); print([l for l in text.splitlines() if 'sp_idx =' in l],flush=True)
match=re.search(r'sp_idx\s*=\s*int\([^\n]*?\*\s*(0\.\d+)',text); n=int(N*float(match[1])) if match else 52
nodes=[runpy.run_path('/holdout_task/_frozen_nodes/node_%02d.py'%j)['run_node'] for j in [1,2,3]]
def check(tr,va):
    sub={**s,'train':{'x':a['x'][tr],'y':a['y'][tr],'traj':a['traj'][tr]},'test':{'x':a['x'][va]}}
    sub['train']['summary']=np.array([a['y'][tr].mean(),a['y'][tr].std(),a['traj'][tr,-1].mean(),a['traj'][tr,-1].std()]); args={'__input__':sub}; q=nodes[0](args); outs=[]
    for v in [0,1,2]:
        m.VARIANT=v; r=(old if v==0 else m).run_node(args)
        o=nodes[2]({'node_00':r,'node_02':nodes[1]({'node_00':r,'node_01':q})}); outs.append((o['y_hat'],o['state_pred']))
    return outs
splits=[]
for off in range(4):
    va=i[i%4==off]; tr=i[i%4!=off]; outs=check(tr,va); splits.append((va,outs)); print('independent',off,[(float(abs(y-a['y'][va]).mean()),float(abs(t-a['traj'][va]).mean())) for y,t in outs],flush=True)
def mix(items, weights, j):return sum(w*r[j] for w,r in zip(weights,items))
opts=[np.eye(3)[k] for k in range(3)]+[np.array(t) for t in [(0,.5,.5),(.2,.4,.4),(.5,.25,.25),(0,.25,.75),(0,.75,.25)]]
w=[]
for j,label in enumerate(['y','traj']):
    ranked=[]
    for weights in opts:
        errs=np.array([abs(mix(o,weights,j)-a[label][va]).mean() for va,o in splits]); ranked.append((.65*errs.mean()+.35*errs.max(),weights,errs))
    _,b,e=min(ranked,key=lambda k:k[0]); w.append(b); print(label,'weights',b,'mean/worst',e.mean(),e.max(),flush=True)
# The frozen trajectory kernel is nonlinear in the design: promote a single trajectory model.
j=1
idx=min(range(3),key=lambda k:.65*np.mean([abs(o[k][j]-a['traj'][va]).mean() for va,o in splits])+.35*max(abs(o[k][j]-a['traj'][va]).mean() for va,o in splits))
print('selected trajectory',idx,flush=True)
code=Path('/tmp/challenger23.py').read_text().split('\ndef run_node')[0]; code+='\n'+Path('/tmp/base23.py').read_text().split('\ndef run_node')[1].join(['def _baseline','']) if False else ''
base=Path('/tmp/base23.py').read_text().split('\ndef run_node')[1]
code+='\ndef _baseline'+base
code+='\n_Y_WEIGHTS='+repr(w[0].tolist())+'\n_TRAJECTORY='+str(idx)+'\n'
code+='''
def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; pred=None; design=None
    for k,weight in enumerate(_Y_WEIGHTS):
        if not weight and k!=_TRAJECTORY:
            continue
        if k==0:
            out=_baseline(inputs)
        else:
            p,m,z,b=predict_local(s,k); n=len(m)
            out={'y_hat':p[n:,0].copy(),'trajectory_design':np.c_[p[:,1:]/np.maximum(m.std(0),1e-5),.035*z,.035*b]}
        if weight:
            pred=weight*out['y_hat'] if pred is None else pred+weight*out['y_hat']
        if k==_TRAJECTORY: design=out['trajectory_design']
    return {'y_hat':pred,'trajectory_design':design}
'''
fp=Path('/holdout_task/node_00.py'); fp.write_text(code)
