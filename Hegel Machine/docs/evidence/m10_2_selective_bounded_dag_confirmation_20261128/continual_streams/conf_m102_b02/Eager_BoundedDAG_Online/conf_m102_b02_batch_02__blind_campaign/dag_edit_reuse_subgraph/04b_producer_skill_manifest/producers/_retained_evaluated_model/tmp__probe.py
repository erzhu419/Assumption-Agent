import numpy as np,pickle,sys
from scipy.spatial.distance import cdist
from scipy.special import erf
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
sys.path.insert(0,'/holdout_task/_frozen_nodes'); import node_00
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=node_00._affine(s['panel_a'],True); b=node_00._affine(s['panel_b']); x=np.linalg.solve(a[1:].T,(s['train']['x']-a[0]).T).T; z=node_00.run_node({'__input__':s})['z']; Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(Y); ids=np.arange(n); splits=[(ids[(ids<l)|(ids>=r)],ids[l:r]) for l,r in [(52,72),(0,20),(20,40),(40,60)]]
for k in ['panel_a','panel_b','panel_c']:
 p=s[k]; u=p['u']; print(k)
 for j in p.keys()-{'u'}:
  v=p[j]; print(j,[(d,round(np.mean(abs((f:=PolynomialFeatures(d).fit_transform(u))@np.linalg.lstsq(f,v,rcond=None)[0]-v)),6)) for d in (1,2,3)])
print('x range',x.min(0),x.max(0)); print('trajectory temporal modes',np.linalg.svd(Y[:,1:],full_matrices=False)[2][:2]); print('Y correlations linear',np.round(np.corrcoef(np.c_[x,Y[:,0],Y[:,1],Y[:,-1]].T)[:4,-3:],3))
def score(K,l):
 scores=[]
 for i,j in splits:
  E=abs(K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+l*np.eye(len(i)),Y[i])-Y[j]); scores.append([E[:,0].mean(),E[:,1:].mean()])
 S=np.array(scores); return max(S.max(0)),S[0],S.max(0)
res=[]
for nm,w in [('x',x),('z',z),('both',np.c_[x,z])]:
 for conv in ('identity','tanh','asinh','erf','sin'):
  for a in ([1] if conv=='identity' else [.75,1.5,2.5,3.5,5]):
   h={'identity':lambda u:u,'tanh':np.tanh,'asinh':np.arcsinh,'erf':erf,'sin':np.sin}[conv](a*w)
   h=h/np.maximum(h.std(0),.01); v=w/np.maximum(w.std(0),.01)
   for form,g in [('h',h),('vh',np.c_[v,h])]:
    lin=g@g.T/g.shape[1]; D=cdist(g,g,'sqeuclidean')/g.shape[1];
    for typ,K in [(f'p{d}',(1+lin)**d) for d in (1,2,3,4)] + [(f'g{d}',np.exp(-d*D)+(1+lin)**2) for d in (.1,.5,2,5)]:
     for l in (1e-7,1e-4,.002,.02,.15): res.append((score(K,l), (nm,conv,a,form,typ,l)))
for r in sorted(res,key=lambda r:r[0][0])[:18]: print(r)
for w in [x,z]:
 F=PolynomialFeatures(2).fit_transform(w); C=np.linalg.lstsq(F,Y[:,[0,1,-1]],rcond=None)[0]; print('quadratic terms',np.round(C,3)); print('quad train MAE',np.mean(abs(F@C-Y[:,[0,1,-1]]),0))
