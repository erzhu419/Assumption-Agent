import pickle, numpy as np, importlib.util
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; n=52
spec=importlib.util.spec_from_file_location('a','/tmp/hegel_continual_archive/conf_m102_a02_batch_09_solver.py'); a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
ls=lambda f,y:np.linalg.lstsq(f,y,rcond=1e-10)[0]
aff=lambda z:np.c_[np.ones(len(z)),z]
def quad(z):
 i,j=np.triu_indices(z.shape[1]); return np.c_[aff(z),z[:,i]*z[:,j]]
z=a._inverse(tr['x'],a._calibrate(s['panel_a'])); print('calibration',a._calibrate(s['panel_a']),flush=True)
ops={'z':z}
for panel,vk in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[panel]['u'],s[panel][vk]; cand=[]
 for typ,fun,starts in [('rational',lambda x,p:x/(1+p[0]*abs(x)**p[1]),[(.5,1),(1,2),(2,3)]),('gauss',lambda x,p:x*np.exp(-p[0]*x*x),[(.5,),(3,)]),('tanh',lambda x,p:np.tanh(p[0]*x),[(.5,),(2,)]),('sin',lambda x,p:np.sin(p[0]*x),[(1,),(3,)]),('both',lambda x,p:np.c_[x*np.exp(-p[0]*x*x),x/(1+p[1]*abs(x)**p[2])],[(2,.5,1),(4,1,2)])]:
  def f(x,p):return np.c_[aff(x),fun(x,p)]
  def resid(p):
   F=f(u,p); return (F@ls(F,v)-v).ravel()
  fit=min((least_squares(resid,p,bounds=(.01,10),max_nfev=100) for p in starts),key=lambda q:q.cost); p=fit.x; F=f(u,p); w=ls(F,v); print(panel,vk,typ,np.round(p,4),np.sqrt(np.mean(resid(p)**2)),flush=True); cand.append((fit.cost,typ,f(z,p)@w))
 for _,typ,c in cand:ops[vk+'_'+typ]=c
for name,x in list(ops.items()):
 for use,f in [('aff',aff(x)),('quad',quad(x)),('mix',np.c_[quad(z),x])]:
  w=ls(f[:n],y[:n]); err=abs(f@w-y); st=np.mean(abs(f@ls(f[:n],tr['traj'][:n])-tr['traj'])[n:]); print(name,use,'y',np.round([err[:n].mean(),err[n:].mean()],5),'st',round(st,5))
