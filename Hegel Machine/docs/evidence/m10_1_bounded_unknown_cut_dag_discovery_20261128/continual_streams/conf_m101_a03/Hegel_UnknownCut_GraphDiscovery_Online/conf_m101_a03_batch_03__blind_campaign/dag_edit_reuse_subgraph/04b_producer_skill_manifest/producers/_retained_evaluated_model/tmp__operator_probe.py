import pickle,itertools,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
aff=lambda u: np.c_[np.ones(len(u)),u]
funcs={'sin':np.sin,'tanh':np.tanh,'cos':np.cos,'rat':lambda x:x/(1+x*x)}
for key,out in [('b','v'),('c','v1'),('c','v2')]:
 p=s['panel_'+key]; u,v=p['u'],p[out]; a=aff(u); res=[]
 qs={'0':u[:,:0],'sq':u*u,'cross':np.column_stack([u[:,i]*u[:,j] for i,j in itertools.combinations(range(4),2)])}; qs['all']=np.c_[qs['sq'],qs['cross']]
 for qn,q in qs.items():
  for fn,f in funcs.items():
   def calc(t, ret=False):
    d=np.c_[a,q,f(t*u)]; w=np.linalg.lstsq(d,v,rcond=None)[0]; r=v-d@w
    if not ret:return np.mean(r*r)
    h=np.sum(d*np.linalg.pinv(d).T,axis=1); return np.sqrt(np.mean(r*r)),np.max(np.abs(r/(1-h[:,None]))),w
   grid=np.linspace(.1,5,99); i=np.argmin([calc(t) for t in grid]); opt=minimize_scalar(calc,bounds=(grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]),method='bounded'); rms,worst,w=calc(opt.x,True);res.append((rms,worst,qn,fn,opt.x,w))
 print(key,out,'best by fit and jackknife:')
 for z in sorted(res,key=lambda x:x[0])[:4]+sorted(res,key=lambda x:x[1])[:2]: print(z[:5]);
 print('best weights\n',sorted(res,key=lambda x:x[0])[0][5].round(5))
