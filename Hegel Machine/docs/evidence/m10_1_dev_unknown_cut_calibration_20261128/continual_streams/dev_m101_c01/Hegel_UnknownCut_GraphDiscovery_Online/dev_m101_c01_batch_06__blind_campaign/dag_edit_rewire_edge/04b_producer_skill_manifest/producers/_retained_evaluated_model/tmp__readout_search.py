import runpy,pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']({'__input__':s}); banks=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']({'node_00':a}); Y=np.column_stack([s['train']['y'],s['train']['traj']]); n=len(Y); x=a['x'][:n]; raw=s['train']['x']; one=lambda z:np.ones((len(z),1)); lin=lambda z:np.column_stack([one(z),z]); quad=lambda z:np.column_stack([lin(z),*[z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(z.shape[1]),2)]])
b=s['panel_b']; coef=np.linalg.lstsq(lin(b['u']),b['v'],rcond=None)[0]; print('B affine',coef.round(5),'rotation singular values',np.linalg.svd(coef[1:],compute_uv=False))
c=s['panel_c']; u=c['u']; coords={'direct':lambda z:z,'Bforward':lambda z:lin(z)@coef,'Binverse':lambda z:(z-coef[0])@np.linalg.inv(coef[1:])}; C={}
for key in ('v1','v2'):
 out=c[key]; scores=[]
 for coord,trans in coords.items():
  z=trans(u)
  for fam,fn,lo,hi in [('tanh',np.tanh,.2,2.5),('sin',np.sin,.2,2.5),('rational',lambda z:z/np.sqrt(1+z*z),.03,3)]:
   for form,basis in [('linear',lin),('quad',quad),('linear+raw',lambda h,z=z:np.column_stack([lin(h),z]))]:
    def score(scale):
     F=basis(fn(scale*z)); pinv=np.linalg.pinv(F); h=(F*pinv.T).sum(1); r=out-F@(pinv@out); return np.mean(np.abs(r/(1-h[:,None])))
    res=minimize_scalar(score,bounds=(lo,hi),method='bounded'); scores.append((res.fun,coord,fam,form,res.x))
 scores.sort(); print(key,'best bases',scores[:7]); C[key]=scores[0]
print('trajectory spectrum',np.linalg.svd(Y[:,1:],compute_uv=False))
# Compare the official-size split with random blocks and coordinate-edge holdouts.
split=max(40,n-24); rng=np.random.default_rng(42); rows=np.arange(n); radius=np.linalg.norm(x-x.mean(0),axis=1); outer=np.argsort(radius)[-12:]; splits=[(rows[:split],rows[split:])]+[(p[:split],p[split:]) for p in (rng.permutation(n) for _ in range(10))]+[(np.setdiff1d(rows,outer),outer)]; features={'none':one(x),'linear':lin(x),'quadratic':quad(x),'cubic':np.column_stack([quad(x),*[np.prod(x[:,inds],axis=1) for inds in combinations_with_replacement(range(4),3)]])};
for key,(_,coord,fam,form,scale) in C.items():
 fn={'tanh':np.tanh,'sin':np.sin,'rational':lambda z:z/np.sqrt(1+z*z)}[fam]; z=coords[coord](x); h=fn(scale*z); basis={'linear':lambda:lin(h),'quad':lambda:quad(h),'linear+raw':lambda:np.column_stack([lin(h),z])}[form](); mapping=np.linalg.lstsq({'linear':lambda:lin(fn(scale*coords[coord](u))),'quad':lambda:quad(fn(scale*coords[coord](u))),'linear+raw':lambda:np.column_stack([lin(fn(scale*coords[coord](u))),coords[coord](u)])}[form](),c[key],rcond=None)[0]; features[key]=np.column_stack([lin(x),basis@mapping]); features[key+'_basis']=basis
features['panels']=np.column_stack([features['v1'],features['v2'][:,5:]]); features['panelquad']=np.column_stack([quad(x),features['v1'][:,5:],features['v2'][:,5:]]);
d=cdist(x,x,'sqeuclidean'); kernels={'zero':np.zeros_like(d),'rbf':banks['base'][1,:n,:n],'wide':banks['wide'][3,:n,:n],'matern':banks['base'][2,:n,:n],'mix':banks['base'][1,:n,:n]+.1*banks['base'][2,:n,:n],**{f'rbf{w}':np.exp(-d/w) for w in (2,4,12,40)}}
records=[]
for fname,F in features.items():
 for kname,K in kernels.items():
  for lam in ([1.] if kname=='zero' else [1e-7,1e-5,1e-4,1e-3,1e-2]):
   fold=[]
   for tr,te in splits:
    fac=cho_factor(K[np.ix_(tr,tr)]+lam*np.eye(len(tr))); X=F[tr]; KiX=cho_solve(fac,X); KiY=cho_solve(fac,Y[tr]); coef2=np.linalg.lstsq(X.T@KiX,X.T@KiY,rcond=1e-9)[0]; pred=F[te]@coef2+K[np.ix_(te,tr)]@(KiY-KiX@coef2); err=np.abs(pred-Y[te]); fold.append([err[:,0].mean(),err[:,1:].mean(),err[:,1:].mean(0).max()]);
   g=np.asarray(fold); records.append((fname,kname,lam,g[0],g[:-1].mean(0),g[:-1].max(0),g[-1]));
for col in (0,1):
 print('output',col,'rows: public, mean, worst, edge');
 for r in sorted(records,key=lambda r:r[4][col]+.20*r[6][col])[:14]:print(r[:3],*[np.round(q,5).tolist() for q in r[3:]])
print('kernel-only reference');
for col in (0,1):
 for r in sorted((r for r in records if r[0]=='none'),key=lambda r:r[4][col]+.20*r[6][col])[:2]:print(r[:3],*[np.round(q,5).tolist() for q in r[3:]])
