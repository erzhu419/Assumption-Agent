import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; X=s['train']['x']; p=s['panel_a']; u,v=p['u'],p['v']; good=np.linalg.norm(v,axis=1)<7; u,v=u[good],v[good]
print('A sample pairs',np.c_[u,v][:20].round(3));print('bounds A',u.min(0),u.max(0),v.min(0),v.max(0)); print('X quantiles',np.quantile(X,[0,.05,.25,.5,.75,.95,1],axis=0).round(3)); print('telemetry',s['telemetry_vec']);
def fit(F,v):return np.linalg.lstsq(F,v,rcond=None)[0]
def score(F,v):
 w=fit(F,v); h=np.sum(F*np.linalg.pinv(F).T,axis=1); e=F@w-v; print(np.sqrt(np.mean(e**2,axis=0)).round(4),np.sqrt(np.mean((e/np.maximum(1-h[:,None],1e-6))**2,axis=0)).round(4));return w
for name,U,V in [('A',u,v),('B',s['panel_b']['u'],s['panel_b']['v']),('C1',s['panel_c']['u'],s['panel_c']['v1']),('C2',s['panel_c']['u'],s['panel_c']['v2'])]:
 print(name,'std',U.std(0).round(3),V.std(0).round(3), 'corr',np.corrcoef(U.T,V.T)[:4,4:].round(2))
 for g in (lambda x:x,np.tanh,np.arctan,lambda x:x/(1+np.abs(x)),lambda x:np.sin(x),lambda x:x/np.sqrt(1+x*x)):
  score(np.c_[np.ones(len(U)),g(U)],V)
 if name!='A': continue
 for deg in (2,3):
  print('complete polynomial',deg); F=np.column_stack([np.ones(len(U))]+[np.prod(U[:,ix],axis=1) for k in range(1,deg+1) for ix in combinations_with_replacement(range(4),k)]); w=score(F,V); print(w[:15].round(3))
 for label,g in [('tanh',np.tanh),('arctan',np.arctan),('softsign',lambda x:x/(1+abs(x))),('rational',lambda x:x/np.sqrt(1+x*x)),('sin',np.sin)]:
  for mode in ('before','after','add'):
   errs=[]; params=[]
   for j in range(4):
    z=np.c_[np.ones(len(U)),U]; y=V[:,j]; w=fit(z,y)
    if mode=='after': fun=lambda a: a[5]+a[6]*g(z@a[:5])-y; init=np.r_[w*.5,0,2]
    if mode=='before': fun=lambda a: np.c_[np.ones(len(U)),g(a[5:9]*U+a[9:13])]@a[:5]-y; init=np.r_[w*2,np.ones(4)*.5,np.zeros(4)]
    if mode=='add': fun=lambda a: z@a[:5]+a[5]*g(z@a[6:])-y; init=np.r_[w,1,np.zeros(5)]; init[7+j]=.6
    opt=least_squares(fun,init,max_nfev=250); errs.append(np.sqrt(np.mean(opt.fun**2))); params.append(opt.x.round(3))
   print(label,mode,np.round(errs,4), np.asarray(params) if max(errs)<.05 else '')
