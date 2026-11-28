import pickle,numpy as n
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; d=n.load('/tmp/modeldata.npz'); x=d['x']; y=d['y']; t=s['eval_times']; z=s['train']['traj']; ls=lambda a,b:n.linalg.lstsq(a,b,rcond=None)[0]; A=lambda x:n.c_[n.ones(len(x)),x]; i,j=n.triu_indices(4); Q=lambda x:n.c_[A(x),x[:,i]*x[:,j]]
u,v=s['panel_b']['u'],s['panel_b']['v']; h=lambda x,p:n.c_[A(x),n.cos(p*x)]
w=ls(h(u,2.6),v); print('B COS coefs',n.round(w,4),'mae',n.mean(abs(h(u,2.6)@w-v),0)); print('B examples',n.round(n.c_[u,v,h(u,2.6)@w-v],3))
B=lambda p:-n.expm1(-t[:,None]*p)
for cut in (52,72):
 ix=slice(0,cut); jx=slice(52,None) if cut==52 else ix
 rates=least_squares(lambda p:(B(p)@ls(B(p),z[ix].T)-z[ix].T).ravel(),[.41,1.8]).x
 for p in (2.4,2.6):
  w=ls(h(u,p),v); c=h(x,p)@w; k=n.cos(p*x); wt=ls(h(x[ix],p),z[ix]); print(cut,p,'state',n.mean(abs(h(x[jx],p)@wt-z[jx])), 'cpanel',n.mean(abs(h(u,p)@w-v)))
  for name,f in [('cos',h(x,p)),('cosquad',Q(k)),('Bquad',Q(c)),('Baffcosquad',n.c_[Q(c),h(x,p)[:,1:]]),('cossin',n.c_[h(x,p),n.sin(p*x)]),('cosx2',n.c_[h(x,p),x*x]),('cosxi',n.c_[h(x,p),x*k]),('cos2harm',n.c_[h(x,p),n.cos(2*p*x)]),('cosxhalf',n.c_[h(x,p),x*n.cos(p*x/2)])]:
   wt=ls(f[ix],y[ix]); print(name,round(n.mean(abs(f[jx]@wt-y[jx])),5),n.round(wt,2) if cut==72 else '')
