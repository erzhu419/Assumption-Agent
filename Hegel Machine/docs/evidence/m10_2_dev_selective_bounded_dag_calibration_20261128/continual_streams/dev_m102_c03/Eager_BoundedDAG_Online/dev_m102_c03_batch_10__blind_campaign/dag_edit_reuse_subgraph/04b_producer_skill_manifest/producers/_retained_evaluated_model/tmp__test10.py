import numpy as np,pickle
from cal10 import amap,other
from read10 import candidates
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; Y=np.c_[t['y'],t['traj']]; Z=amap(s)(t['x']); D=other(s,Z); R={}
for j,ix in enumerate([np.arange(72),*np.array_split(np.random.default_rng(813).permutation(72),3)]):
 if not j: a,b=ix[:52],ix[52:]
 else: b=ix; a=np.setdiff1d(np.arange(72),b)
 ix=np.r_[a,b]; ss={**s,'train':{'x':t['x'][a]},'test':{'x':t['x'][b]}}
 for k,p in candidates(ss,{h:v[ix] for h,v in D.items()},len(a),Y[a]):
  e=abs(p-Y[b]); R.setdefault(k,[]).append([e[:,0].mean(),e[:,1:].mean()])
for col in (0,1):
 for k,v in sorted(R.items(),key=lambda kv:np.max(np.array(kv[1])[:,col]))[:10]: print(col,k,np.round(v,5).tolist())
