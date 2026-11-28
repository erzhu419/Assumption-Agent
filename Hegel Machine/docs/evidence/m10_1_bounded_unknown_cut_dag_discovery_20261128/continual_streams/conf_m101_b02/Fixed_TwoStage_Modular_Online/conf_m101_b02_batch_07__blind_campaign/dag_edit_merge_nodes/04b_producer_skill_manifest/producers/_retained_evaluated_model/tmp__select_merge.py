import sys,pickle,itertools as it,numpy as np
from pathlib import Path
sys.path.insert(0,'/holdout_task')
from node_00 import _calibrate,_readout_features,_predict_readout
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; Y=np.c_[y,t]; n=len(x)
aw,_,_,_,odd=_calibrate(s); folds=[np.arange(n-20,n),*np.array_split(np.arange(n),4)]; rows=[]
for group in it.product(range(3),('odd','cube')):
 a,b=_readout_features(x,group,aw,odd)
 for mode,l,inter,fitted,reg in it.product(('rbf','matern','poly'),(4.,6.,8.,10.,12.),('mixed','limited'),(True,False),(1e-5,3e-5,1e-4,3e-4,1e-3)):
  cfg=(group,mode,l,inter,fitted,reg); p=[]
  for v in folds:
   i=np.setdiff1d(np.arange(n),v);p.append(_predict_readout((a[i],b[i]),(a[v],b[v]),Y[i],cfg))
  rows.append((cfg,p))

def selection(j):
 sl=0 if j==0 else slice(1,None)
 def errors(p):return np.array([np.abs(q[:,sl]-Y[v,sl]).mean() for q,v in zip(p,folds)])
 def rank(e):return (0 if e[0]<=.0379 else 1,e[1:].max() if e[0]<=.0379 else e[0],e.mean())
 singles=sorted([(rank(e),[k],e) for k,(_,p) in enumerate(rows) for e in [errors(p)]],key=lambda r:r[0]); best=singles[0]
 shortlist=set(k[0] for _,k,_ in singles[:85]);shortlist.update(k[0] for _,k,_ in sorted(singles,key=lambda r:r[2][1:].max())[:70]);shortlist.update(k[0] for _,k,_ in sorted(singles,key=lambda r:r[2][0])[:45]);
 for a,b in it.combinations(sorted(shortlist),2):
  if rows[a][0][:4]==rows[b][0][:4]:continue
  e=errors([(p+q)/2 for p,q in zip(rows[a][1],rows[b][1])]); r=rank(e)
  if r<best[0]:best=(r,[a,b],e)
 cfg=[rows[i][0] for i in best[1]];print(j,cfg,'MAEs',best[2].round(6));return cfg
specs=[selection(j) for j in range(2)]
p=Path('/holdout_task/node_00.py'); text=p.read_text(); i=text.index('def _kernel('); j=text.index('def _readout_gram('); text=text[:i]+text[j:];p.write_text(text+f'''\n\n_READOUTS = {specs!r}\n\n\ndef run_node(inputs: dict) -> dict:\n    study = inputs.get('__input__', inputs)\n    y, state = _predict(study, _READOUTS)\n    terminal = state[:, -1]\n    return {{'y_hat': y, 'state_pred': state,\n            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}}\n''')
