import pickle,numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def fit(b,v):
 c=np.linalg.lstsq(b,v,rcond=None)[0]; r=v-b@c; h=(np.linalg.svd(b,full_matrices=False)[0]**2).sum(1)
 return np.mean(r*r)**.5,np.mean(abs(r/np.maximum(1-h[:,None],1e-9))),c
if __name__=='__main__':
 np.set_printoptions(precision=3,suppress=True,linewidth=135)
 for p in ['panel_a','panel_b','panel_c']:
  u=s[p]['u']; print(p,{k:(a.shape,float(a.min()),float(a.max())) for k,a in s[p].items()});
  if p=='panel_a':
   q=np.column_stack([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)]); base=np.c_[np.ones(len(u)),u]; v=s[p]['v'];
   for name,b in [('linear',base),('diag2',np.c_[base,u*u]),('quadratic',np.c_[base,q]),('quad+cube',np.c_[base,q,u**3])]:
    c=np.linalg.lstsq(b,v,rcond=None)[0]
    for _ in range(40):
     r=np.linalg.norm(v-b@c,axis=1); w=1/np.sqrt(1+(r/max(np.median(r),.0002))**2); c=np.linalg.lstsq(b*w[:,None],v*w[:,None],rcond=None)[0]
    r=np.linalg.norm(v-b@c,axis=1); print(name,np.quantile(r,[.1,.5,.8,.9,.95,1]),'coef',c.T)
 u=s['panel_c']['u']; v=s['panel_c']['v2']; z=u*u; print('u2 ranges',z.max(0)); print('u,v2\n',np.c_[u,v]);
 funcs={'square':lambda a:z,'abs':lambda a:abs(u),'quartic':lambda a:z+a*z*z,'rational':lambda a:z/(1+a*z),'sqrt':lambda a:np.sqrt(1+a*z),'cos':lambda a:np.cos(a*u),'cos_sq':lambda a:np.cos(a*z),'sin_sq':lambda a:np.sin(a*z),'tanh_sq':lambda a:np.tanh(a*z),'exp_sq':lambda a:np.exp(-a*z),'gauss_sat':lambda a:np.exp(-a*z*z),'log':lambda a:np.log1p(a*z),'exp+':lambda a:np.exp(a*z),'pow':lambda a:abs(u)**a}
 for typ in [0,1,2]:
  l=[np.ones((len(u),1)),np.c_[np.ones(len(u)),u],np.c_[np.ones(len(u)),z]][typ]; print('nuisance',typ)
  for name,fn in funcs.items():
   grid=np.linspace(-.3 if name=='quartic' else .001,5,250); fun=lambda a:fit(np.c_[l,fn(a)],v)[0]; j=np.argmin([fun(a) for a in grid]); a=minimize_scalar(fun,bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded').x; print(name,round(a,6),np.round(fit(np.c_[l,fn(a)],v)[:2],7))
 for k in ['train','test']:
  print(k,'x ranges',s[k]['x'].min(0),s[k]['x'].max(0))
