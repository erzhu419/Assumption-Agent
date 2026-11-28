import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];np.set_printoptions(precision=4,suppress=True,linewidth=150)
def fit(d,v):
 w=np.linalg.lstsq(d,v,rcond=None)[0];r=v-d@w;h=np.sum(d*np.linalg.pinv(d).T,axis=1);return np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/(1-h).reshape((-1,)+(1,)*(r.ndim-1)))**2)),w
u,v=[s['panel_a'][k] for k in ('u','v')];keep=np.max(abs(v),axis=1)<3; print('A excluded',np.where(~keep),v[~keep]);u,v=u[keep],v[keep];p=[(i,j) for i in range(4) for j in range(i,4)];d=np.column_stack([np.ones(len(u)),u,*[u[:,i]*u[:,j] for i,j in p]]);print('A quadratic',p,fit(d,v));
for j in range(4):
 own=u[:,j];nb=u[:,(j+1)%4];prev=u[:,(j-1)%4];print('A channel',j)
 for nm,g in [('none',lambda x:x),('sin',np.sin),('tanh',np.tanh),('cube',lambda x:x**3)]:
  for a in [.22,1,1.25,2]:
   best=[]
   for txt,z in [('own',own),('nb',nb),('prev',prev),('sum',own+nb),('diff',own-nb),('prod',own*nb)]:
    e,l,w=fit(np.column_stack([d,g(a*z)]),v[:,j]);best.append((l,txt))
   print(nm,a,sorted(best)[:2],end='; ')
  print()
for pn,pk in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=[s[pn][k] for k in ('u',pk)];a=np.column_stack([np.ones(len(u)),u]);print(pn,pk)
 for name,g in [('sin',np.sin),('tanh',np.tanh),('rational',lambda x:x/(1+x*x)),('sqrt',lambda x:x/np.sqrt(1+x*x)),('cube',lambda x:x**3),('square',lambda x:x*x),('asinh',np.arcsinh),('atan',np.arctan)]:
  def ds(k):return np.column_stack([a,g(k*u)])
  z=least_squares(lambda z:(v-ds(np.exp(z))@fit(ds(np.exp(z)),v)[2]).ravel(),[0.],bounds=([-2.],[2.])).x
  print(name,'best',np.exp(z),fit(ds(np.exp(z)),v)[:2],'fixed',[(k,fit(ds(k),v)[:2]) for k in [1,1.25,2,2.5,3]],'w',fit(ds(np.exp(z)),v)[2] if name in ['sin','tanh'] and pn=='panel_b' else '')
