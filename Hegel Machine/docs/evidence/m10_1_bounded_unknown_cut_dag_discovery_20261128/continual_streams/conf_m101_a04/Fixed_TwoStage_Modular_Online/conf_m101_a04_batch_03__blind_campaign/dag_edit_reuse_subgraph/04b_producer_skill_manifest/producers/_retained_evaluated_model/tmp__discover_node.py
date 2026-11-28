import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=5,suppress=True,linewidth=130)
p=s['panel_a']; u,v=p['u'],p['v']; r=np.roll(u,-1,axis=1)
X=np.stack([np.ones_like(u),u,r*(u-1),u**3],axis=-1)
w=np.ones(len(u))
for _ in range(24):
 z=np.sqrt(w); g=np.array([np.linalg.lstsq(X[:,j]*z[:,None],v[:,j]*z,rcond=None)[0] for j in range(4)])
 e=np.linalg.norm(v-np.einsum('njk,jk->nj',X,g),axis=1); w=1/(1+(e/(2*np.median(e)))**2)
print('A sparse coeffs',g,'residual largest',np.sort(e)[-6:]); print('A clean ranges',v[e<.001].min(0),v[e<.001].max(0))
p=s['panel_c']; u=p['u']; one=np.ones((len(u),1)); lin=np.c_[one,u]
for key in ['v1','v2']:
 v=p[key]; print(key)
 for name,fn in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('atan',np.arctan),('softsign',lambda x:x/np.sqrt(1+x*x))]:
  for shape in ['common','shift','coord','mix']:
   def design(a):
    if shape=='common': return np.c_[lin,fn(a[0]*u)]
    if shape=='shift': return np.c_[lin,fn(a[0]*u+a[1])]
    if shape=='coord': return np.c_[lin,fn(u*a)]
    return np.c_[one,u+a[2]*fn(a[0]*u+a[1])]
   def calc(a,ret=False):
    x=design(a); g=np.linalg.lstsq(x,v,rcond=None)[0]; e=v-x@g
    if ret:
     h=np.sum(x*np.linalg.pinv(x).T,axis=1); return g,np.mean(abs(e)/(1-h[:,None]))
    return e.ravel()
   starts={'common':[[1.],[2.],[3.]],'shift':[[1.,0],[2.,0],[3.,.25]],'coord':[[1.]*4,[2.]*4,[3.]*4],'mix':[[1.,0.,1.],[2.,0.,-.65],[1.,0.,-.65]]}[shape]
   opts=[least_squares(calc,a,max_nfev=160) for a in starts]; opt=min(opts,key=lambda o:o.cost)
   g,press=calc(opt.x,True)
   print(name,shape,opt.x,'fit rmse',round(np.sqrt(np.mean(opt.fun**2)),7),'conditional PRESS',round(press,7))
   if (name,shape) in [('tanh','common'),('cos','common')] or press<.0005: print(g)
