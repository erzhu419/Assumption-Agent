import numpy as np,pickle,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aff(u):return np.c_[np.ones(len(u)),u]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def cubic_inv(y,g):return 2/np.sqrt(3*g)*np.sinh(np.arcsinh(1.5*np.sqrt(3*g)*y)/3)
funcs={'sinh_inv':lambda y,g:np.arcsinh(g*y)/g,'cubic_inv':cubic_inv,'atan_inv':lambda y,g:np.tan(g*y)/g,'tanh_inv':lambda y,g:np.arctanh(g*y)/g,'asinh_inv':lambda y,g:np.sinh(g*y)/g,'exp_inv':lambda y,g:np.log1p(g*y)/g}
for k in ['panel_b','panel_c']:
 p=s[k];u=p['u'];a=aff(u);d=u.shape[1];qu=np.c_[a,*[u[:,i]*u[:,j] for i,j in itertools.combinations_with_replacement(range(d),2)]]
 for key,v in p.items():
  if key=='u':continue
  print(k,key,'range',np.min(v,axis=0),np.max(v,axis=0)); print('aff coefficients',ls(a,v));print('aff and quad residual',np.sqrt(np.mean((a@ls(a,v)-v)**2,axis=0)),np.sqrt(np.mean((qu@ls(qu,v)-v)**2,axis=0)))
  print('aff residual by row',np.round(np.linalg.norm(a@ls(a,v)-v,axis=1),4))
  for name,f in funcs.items():
   def loss(g,cols=slice(None)):
    vv=v[:,cols]; z=f(vv,g); r=z-a@ls(a,z);return np.mean(r*r)/max(np.var(z),1e-12)
   lim=12
   if name=='atan_inv':lim=1.56/abs(v).max()
   if name in ['tanh_inv','exp_inv']:lim=.999/abs(v).max()
   g=minimize_scalar(loss,bounds=(1e-6,lim),method='bounded'); print(name,round(g.x,6),round(g.fun,9),[(round(h.x,5),round(h.fun,8)) for i in range(d) for h in [minimize_scalar(lambda q:loss(q,i),bounds=(1e-6,lim),method='bounded')]])
