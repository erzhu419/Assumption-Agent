import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; ls=lambda a,y:np.linalg.lstsq(a,y,rcond=1e-10)[0]; one=lambda z:np.c_[np.ones(len(z)),z]
p=s['panel_a'];u,v=p['u'],p['v'];print('A ranges',u.min(),u.max(),np.round(np.sort(np.linalg.norm(v,axis=1)),2));
for j in range(4):
 print('A',j,np.round(np.c_[u[:,j],v[:,j]][np.argsort(u[:,j])][::3],3).T)
Y=s['train']['traj']; t=s['eval_times']; print('SVD',np.linalg.svd(Y,compute_uv=False)); tf=lambda q:-np.expm1(-t[:,None]*q); r=lambda q:(tf(q)@ls(tf(q),Y[:52].T)-Y[:52].T).ravel(); q=least_squares(r,[.4,1.8],bounds=([.01,.8],[.79,5])).x;print('time',q,np.std(r(q)),np.std((tf(q)@ls(tf(q),Y[52:].T)-Y[52:].T)))
for key in ['v1','v2']:
 p=s['panel_c'];u,v=p['u'],p[key]; ix,iy=np.triu_indices(4,1); pairs=lambda x:x[:,ix]*x[:,iy]; cyc=lambda x:x*np.roll(x,-1,axis=1)
 funcs={'aff':lambda x,q:x,'tanh':lambda x,q:np.tanh(q[0]*x),'tanh+lin':lambda x,q:np.c_[x,np.tanh(q[0]*x)],'sin+lin':lambda x,q:np.c_[x,np.sin(q[0]*x)],'sin+tanh':lambda x,q:np.c_[np.tanh(q[0]*x),np.sin(q[1]*x)],'ratabs':lambda x,q:x/(1+q[0]*np.abs(x))+q[1]*x,'rat2':lambda x,q:np.c_[x,x/(1+(q[0]*x)**2)],'tanh+pair':lambda x,q:np.c_[np.tanh(q[0]*x),pairs(x)],'tanh+lin+pair':lambda x,q:np.c_[x,np.tanh(q[0]*x),pairs(x)],'lin+pair':lambda x,q:np.c_[x,pairs(x)],'tanh+square':lambda x,q:np.c_[np.tanh(q[0]*x),x*x],'lin+square+rat':lambda x,q:np.c_[x,x*x,x*x/(1+q[0]*x*x)],'tanh+cos':lambda x,q:np.c_[np.tanh(q[0]*x),np.cos(q[1]*x)],'tanh+lin+square':lambda x,q:np.c_[x,np.tanh(q[0]*x),x*x],'lin+cos':lambda x,q:np.c_[x,np.cos(q[0]*x)],'tanh+cyc':lambda x,q:np.c_[np.tanh(q[0]*x),cyc(x)],'sin':lambda x,q:np.sin(q[0]*x),'cos':lambda x,q:np.cos(q[0]*x),'lin+sq+pair':lambda x,q:np.c_[x,x*x,pairs(x)]}
 out=[]
 for name,f in funcs.items():
  def fit(mask):
   def resid(q):
    a=one(f(u[mask],q));return (a@ls(a,v[mask])-v[mask]).ravel()
   q=min([least_squares(resid,start,bounds=(.001,6),max_nfev=100) for start in ([.7,.3],[1.5,2.3],[2.6,1.6])],key=lambda a:a.cost).x
   return q,ls(one(f(u[mask],q)),v[mask]),np.std(resid(q))
  q,w,e=fit(slice(None)); errs=[]
  for i in range(4):
   m=np.arange(len(u))%4!=i; h,c,_=fit(m);errs.extend(one(f(u[~m],h))@c-v[~m]);
  out.append((np.sqrt(np.mean(np.square(errs))),e,name,np.round(q,5)))
 print(key,*sorted(out,key=lambda x:x[0])[:12],sep='\n')
