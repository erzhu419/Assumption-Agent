import sys,pickle,itertools,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import _panel_b,_quadratic,_affine,_fit,_relaxation,_forecast
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; tr=s['train']['traj']; t=s['eval_times']; u=s['panel_b']['u']; v=s['panel_b']['v']
def warp(z,p):return _affine(z+p[0]*np.sin(p[1]*z))
def res(p):
 a=warp(u,p); return (a@_fit(a,v)-v).ravel()
p=least_squares(res,[-.4,2.8]).x
z0=_panel_b(s['panel_b'],x); z1=warp(x,p)@_fit(warp(u,p),v)
dx=cdist(x,x,'sqeuclidean'); zs=[z0,z1]; dz=[cdist(z,z,'sqeuclidean') for z in zs]; P=[_quadratic(z) for z in zs]; X=_quadratic(x)[:,1:]
# Compact grid chosen from the previous development experiments. Each outer split selects settings using only its training rows.
specs=[(0,False,0.,.075,.0003),(0,False,0.,.035,.0003),(0,True,1.,.035,.002),(0,True,0.,.035,.015),(0,True,.25,.075,.015),(1,False,0.,.075,.0003),(1,False,0.,.035,.0003),(1,True,1.,.035,.002),(1,True,0.,.035,.015),(1,True,.25,.075,.015)]
def scores(a,b):
 modes,B=_relaxation(tr[a],t,[1.8,.41]); target=np.c_[y[a],modes]; out=[]
 for i,raw,w,l,r in specs:
  p=np.c_[P[i],X] if raw else P[i]; d=dx+w*dz[i]; pred=_forecast(d[np.ix_(a,a)],d[np.ix_(b,a)],p[a],p[b],target,l,r); e=np.abs(pred[:,1:]@B.T-tr[b]);out.append([np.abs(pred[:,0]-y[b]).mean(),e.mean(),e[:,-1].mean()])
 return np.array(out)
rng=np.random.default_rng(120917); fixed=[]; nested=[]; chosen=[]
orderings=[np.arange(len(x)),*[rng.permutation(len(x)) for _ in range(23)]]
for ix in orderings:
 a,b=ix[:52],ix[52:]; err=scores(a,b); fixed.append(err)
 inner=np.mean([scores(c[:40],c[40:]) for c in (rng.permutation(a) for _ in range(6))],axis=0); pick=inner[:,:2].argmin(axis=0);chosen.append(pick);nested.append([err[pick[0],0],*err[pick[1],1:]])
fixed=np.array(fixed);nested=np.array(nested);print('columns: documented split, across-split mean, across-split max; metrics: y, state, terminal MAE')
for j,spec in enumerate(specs): print(j,spec,np.round(np.stack([fixed[0,j],fixed[:,j].mean(0),fixed[:,j].max(0)]),5).tolist())
print('nested',np.round(np.stack([nested[0],nested.mean(0),nested.max(0)]),5).tolist(),'selections',np.array(chosen).T.tolist())
