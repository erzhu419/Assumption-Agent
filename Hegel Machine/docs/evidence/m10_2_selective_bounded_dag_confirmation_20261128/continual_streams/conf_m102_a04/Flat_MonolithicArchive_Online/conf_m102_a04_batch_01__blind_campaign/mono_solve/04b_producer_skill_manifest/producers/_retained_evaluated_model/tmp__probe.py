import pickle, importlib.util, numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
q=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/conf_m102_a04_batch_00_solver.py'); old=importlib.util.module_from_spec(q);q.loader.exec_module(old)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; print('t telemetry',t,s['telemetry_vec']); o,k=old._align(s['panel_a']); print('align',o,k); x=(tr['x']-o)/k
lin=np.linalg.lstsq
def fit(f,y):return lin(f,y,rcond=1e-10)[0]
def one(f):return np.c_[np.ones(len(f)),f]
def time(r):return -np.expm1(-t[:,None]*r)
def res(r):
 f=time(r); return (f@fit(f,tr['traj'][:52].T)-tr['traj'][:52].T).ravel()
r=least_squares(res,[.45,1.8],bounds=([.01,.5],[1.6,8])).x; print('rates',r,'std',np.std(res(r))); T=time(r); a=fit(T,tr['traj'].T).T; Y=np.c_[tr['y'],a]
u=s['panel_b']['u']; B=fit(one(u),s['panel_b']['v']); b=one(x)@B; print('B ortho',B[1:]@B[1:].T)
c=s['panel_c']; u=c['u']; maps={}
for key in ('v1','v2'):
 y=c[key]; fs={ 'sin':(lambda x,q:one(np.sin(q[0]*x)),[2]),'cos':(lambda x,q:one(np.cos(q[0]*x)),[2]),'sincos':(lambda x,q:one(np.c_[np.sin(q[0]*x),np.cos(q[0]*x)]),[2]),'sinlin':(lambda x,q:one(np.c_[x,np.sin(q[0]*x)]),[2]),'coslin':(lambda x,q:one(np.c_[x,np.cos(q[0]*x)]),[2]),'rational':(lambda x,q:one(np.c_[x,x*x/(1+q[0]*x*x)]),[.3]),'archivev2':(lambda x,q:one(np.c_[x,x*x,x*x/(1+q[0]*x*x)]),[.3]),'archivev1':(lambda x,q:one(x/(1+q[0]*abs(x))+q[1]*x),[.8,.0])}
 scores=[]
 for name,(f,init) in fs.items():
  def fun(q):
   F=f(u,q);return (F@fit(F,y)-y).ravel()
  p=least_squares(fun,init,max_nfev=200).x; score=np.std(fun(p)); scores.append((score,name,p,f))
 for err,name,p,_ in sorted(scores,key=lambda zz:zz[0]): print('C',key,name,p,err)
 err,name,p,f=min(scores,key=lambda zz:zz[0]);maps[key]=f(x,p)@fit(f(u,p),y)
def report(name,F):
 p=F[52:]@fit(F[:52],Y[:52]); print(name,np.round(np.mean(abs(p-Y[52:]),0),5),round(np.mean(abs((p[:,1:]-Y[52:,1:])@T.T)),5))
for name,z in [('x',x),('b',b),('b_center',b-B[0])]+list(maps.items()):
 report(name,one(z)); report(name+' diag2',one(np.c_[z,z*z])); report(name+' full2',one(np.c_[z,*[z[:,i]*z[:,j] for i,j in combinations_with_replacement(range(4),2)]])); report(name+' rational/trig',one(np.c_[z,z*z,z*z/(1+.3*z*z),np.sin(2*z),np.cos(2*z),np.tanh(z)]));
print('amplitude range',Y.min(0),Y.max(0)); print('correlation',np.corrcoef(np.c_[Y,x,b,*maps.values()].T)[:3]); print('x range',x.min(0),x.max(0))
