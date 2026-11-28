import importlib.util, numpy as np, pickle, shutil
from pathlib import Path
p=Path('/holdout_task')
def load(f):
    q=importlib.util.spec_from_file_location('candidate',f); m=importlib.util.module_from_spec(q); q.loader.exec_module(m); return m
s=pickle.load(open(p/'public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; n=len(x)
order=np.argsort(np.linalg.norm(x,axis=1)); rng=np.random.default_rng(810)
splits=[np.arange(n-20,n),*np.array_split(rng.permutation(n),4),order[-12:]]
sink=load(p/'_frozen_nodes/node_01.py'); trials=[]
for f in ('/tmp/node08_final_baseline.py','/tmp/node08_final_challenger.py'):
    m=load(f); rows=[]
    for va in splits:
        it=np.setdiff1d(np.arange(n),va); st={**s,'train':{k:tr[k][it] for k in ('x','y','traj')},'test':{'x':x[va]}}
        yy,tt=st['train']['y'],st['train']['traj'][:,-1]
        st['train']['summary']=np.array([yy.mean(),yy.std(),tt.mean(),tt.std()]); a=m.run_node({'__input__':st}); b=sink.run_node({'node_00':a})
        rows.append([np.mean(abs(b['y_hat']-tr['y'][va])),np.mean(abs(b['state_pred']-tr['traj'][va]))])
    r=np.array(rows); print(Path(f).stem,'official',r[0].round(6),'worst random',r[1:-1].max(0).round(6),'high radius',r[-1].round(6),flush=True)
    trials.append((r,f))
base,challenger=trials
r,f=challenger
if (r[0].max()<=.038 and base[0][0].max()>.038) or (r.max()<base[0].max() and np.all(r[0]<=base[0][0]+.001)):
    shutil.copyfile(f,p/'node_00.py'); print('Retained challenger')
else:
    print('Retained baseline')
