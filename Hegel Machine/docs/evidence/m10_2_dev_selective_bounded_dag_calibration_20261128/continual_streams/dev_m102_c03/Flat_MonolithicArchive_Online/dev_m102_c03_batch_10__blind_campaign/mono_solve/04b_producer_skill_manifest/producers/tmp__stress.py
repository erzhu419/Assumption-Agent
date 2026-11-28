import pickle,importlib.util,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; u=s['panel_c']['u']; v=s['panel_c']['v1']; t=s['eval_times']; spec=importlib.util.spec_from_file_location('base','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m); fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]; z=m._calibrate(s['panel_a'],np.r_[tr['x'],s['test']['x']]); print('latent train and test extrema',[(x.min(0),x.max(0)) for x in (z[:72],z[72:])]);

def proj(D, v, p, bounds):
 def err(p):
  a=D(u,p);return (a@fit(a,v)-v).ravel()
 o=least_squares(err,p,bounds=bounds,gtol=1e-12); a=D(u,o.x); b=fit(a,v);return D(z,o.x)@b,np.sqrt(np.mean(err(o.x)**2)),o.x,b

d,err,p,b=proj(lambda x,p:m._aff(np.cos(x*p)),s['panel_c']['v2'],[2.4],(1.5,3.2)); print('cos',err,p)
models=[('tanh_lin',lambda x,p:np.c_[m._aff(x),np.tanh(x*p)],np.full(4,1.5),(.1,4)),('tanh_135',lambda x,p:np.c_[m._aff(np.tanh(x*p)),x,x**3,x**5],[1.72],(.5,3)),('rat_135',lambda x,p:np.c_[m._aff(x/(1+(x*p)**2)),x,x**3,x**5],[1.09],(.3,2)),('polynomial',lambda x,p:np.c_[m._aff(x),x**3,x**5,x**7,x**9],[1],(.1,2)),('tanh_35',lambda x,p:np.c_[m._aff(np.tanh(x*p)),x**3,x**5],[1.7],(.3,4)),('rat_35',lambda x,p:np.c_[m._aff(x/(1+(x*p)**2)),x**3,x**5],[1.1],(.3,4)),('tanh_15',lambda x,p:np.c_[m._aff(np.tanh(x*p)),x,x**5],[1.7],(.3,4))];
slices=[np.arange(52,72),np.arange(20),np.arange(26,46),np.argsort(np.linalg.norm(z[:72],axis=1))[-15:],np.argsort(np.max(abs(z[:72]),axis=1))[-15:]]
for name,D,p,bounds in models:
 c,e,p,b=proj(D,v,p,bounds); F=m._design(z,c,d); print(name,'panel',e,p); 
 if name=='tanh_135': print('odd coefficients',np.round(b,3))
 for jj in slices:
  ii=np.setdiff1d(np.arange(72),jj); T=tr['traj'][ii]; rates=least_squares(lambda p:(m._time(t,p)@fit(m._time(t,p),T.T)-T.T).ravel(),np.log([.4,1.8]),bounds=(np.log([.12,1.]),np.log([.9,4.]))).x; Q=np.linalg.qr(m._time(t,rates))[0]; yy=tr['y']; Y=np.c_[yy[ii],T@Q]; B=fit(F[ii],Y); P=F[jj]@B; et=abs(P[:,1:]@Q.T-tr['traj'][jj]); print(' ',np.round([np.mean(abs(P[:,0]-yy[jj])),et.mean(),max(et.mean(0)),np.sqrt(np.mean((F[ii]@B-Y)**2))],5),end='')
 print()
