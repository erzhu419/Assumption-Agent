import numpy as np,pickle,runpy,itertools
from scipy.optimize import least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=S['train']; X=T['x']; n=len(X); A=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_b01_batch_06_solver.py'); head=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; ls=A['_ls']; aff=A['_aff']; pair=A['_pairmap']; z=A['_calibrate'](S['panel_a'],X); u,v=S['panel_b']['u'],S['panel_b']['v']; C=S['panel_c']; p,q=C['u'],C['v2']; cfit=ls(pair(p),C['v1']); dfit=ls(A['_even'](p),q); d0=A['_even'](z)@dfit; d=np.tanh(1.25*d0)
print('A latent range',z.min(0),z.max(0),'C residual', np.std(pair(p)@cfit-C['v1']),np.std(A['_even'](p)@dfit-q)); print('B sv',np.linalg.svd(v-v.mean(0),compute_uv=False));
def cv(w,folds=1):
 out=[]
 for te in [np.arange(52,72)]+list(np.array_split(np.random.default_rng(111).permutation(n),4))[:folds-1]:
  tr=np.setdiff1d(np.arange(n),te); a=head({'__input__':{'train':{k:T[k][tr] for k in ('y','traj')}},'node_00':{'w':w[np.r_[tr,te]]}});out.append([np.mean(abs(a[k]-T[t][te])) for k,t in [('y_hat','y'),('state_pred','traj')]])
 return np.round(out,5)
def profile(name,f,starts):
 def resid(h):
  k=aff(f(u,h));return (k@ls(k,v)-v).ravel()
 fits=[least_squares(resid,h,max_nfev=260) for h in starts]; r=min(fits,key=lambda x:np.mean(x.fun**2));h=r.x; k=aff(f(u,h)); coef=ls(k,v); hat=k@np.linalg.pinv(k); loo=np.std(r.fun.reshape(v.shape)/(1-hat.diagonal())[:,None]); print(name,np.round(h,5),'rmse,press',np.std(r.fun),loo);return aff(f(z,h))@coef,loo
opts={
 'lin':(lambda x,h:x+h[0]*0,[[0]]),
 'sinmix':(lambda x,h:x+h[0]*np.sin(h[1]*x)+h[2]*np.cbrt(x),[[a,b,0] for a in [-.4,.4] for b in [1.25,2.15,3.5,5.5,8]]),
 'rational':(lambda x,h:x/(1+np.exp(h[0])*x*x)+h[1]*x,[[a,b] for a in [-2,0,1] for b in [-.5,0,.5]]),
 'sine':(lambda x,h:np.sin(h[0]*x+h[1])+h[2]*x,[[b,a,0] for a in [0,.5,1] for b in [1.25,2.15,3.5,5.5,8]]),
 'tanh':(lambda x,h:np.tanh(h[0]*x+h[1])+h[2]*x,[[b,a,0] for a in [0,.5] for b in [1.25,2.15,3.5,5.5]]),
 'oddpoly':(lambda x,h:x+h[0]*x**3+h[1]*x**5+h[2]*x**7,[[0,0,0]]),
 'atan':(lambda x,h:np.arctan(h[0]*x+h[1])+h[2]*x,[[b,a,0] for a in [0,.5] for b in [1.25,2.15,3.5,5.5]]),
 'polysep':(lambda x,h:np.c_[x,x*x,x**3,x**5]+h[0]*0,[[0]]),
 'trigrat':(lambda x,h:np.sin(h[0]*x)/(1+np.exp(h[1])*x*x)+h[2]*x,[[b,a,0] for a in [-2,0,1] for b in [1.25,2.15,3.5,5.5,8]]),
}
best=[]
for name,(f,starts) in opts.items():
 b,e=profile(name,f,starts); c0=pair(b)@cfit;c=np.arcsinh(1.25*c0)
 ws={'b':b,'b+d':np.c_[b,d],'c':c,'d':d,'c+d':np.c_[c,d],'c0+d':np.c_[c0,d],'c+d0':np.c_[c,d0],'c0+d0':np.c_[c0,d0],'c0':c0,'tanhc+d':np.c_[np.tanh(1.25*c0),d]}
 scores=sorted((max(cv(w)[0]),k,cv(w)[0]) for k,w in ws.items());print('best',scores[:4]);best.extend((a,name,k,w if False else ws[k]) for a,k,_ in scores)
for a,name,k,w in sorted(best,key=lambda t:t[0])[:12]:print('multi',name,k,cv(w,5))
