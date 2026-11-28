import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
u,v=[s['panel_a'][key] for key in 'uv']; delta=np.linalg.norm(v-u,axis=1); med=np.median(delta); good=delta<=med+max(1,8*np.median(abs(delta-med))); print('A rejected',np.where(~good)[0],np.sort(delta)[-10:]); terms=[t for d in range(4) for t in cr(range(4),d)]; f=np.column_stack([np.prod(u[good][:,t],axis=1) for t in terms]);
for n in (5,15,35):
 w=ls(f[:,:n],v[good]); print('A',n,np.sqrt(np.mean((f[:,:n]@w-v[good])**2,axis=0))); print([(t,np.round(c,4)) for t,c in zip(terms,w) if max(abs(c))>.018])
u,v=[s['panel_b'][key] for key in 'uv'];
for name,feat in [('cos',lambda a,p:np.cos(a*p)),('sin',lambda a,p:np.sin(a*p)),('square',lambda a,p:a*a),('cos0',lambda a,p:np.cos(a*p))]:
 def design(q,p): return np.c_[np.ones(len(q)),q,feat(q,p)] if name!='cos0' else np.c_[np.ones(len(q)),feat(q,p)]
 def fun(p):
  f=design(u,p);return (f@ls(f,v)-v).ravel()
 xx=min([least_squares(fun,[p],bounds=(.02,15)) for p in [1,2,3,5,8,10]],key=lambda x:x.cost); print('B',name,xx.x,np.sqrt(np.mean(xx.fun**2)),np.round(ls(design(u,xx.x),v),3))
for key in ('v1','v2'):
 a,b=s['panel_c']['u'],s['panel_c'][key]; print('C',key,np.sqrt(np.mean((np.c_[np.ones(len(a)),a]@ls(np.c_[np.ones(len(a)),a],b)-b)**2)))
