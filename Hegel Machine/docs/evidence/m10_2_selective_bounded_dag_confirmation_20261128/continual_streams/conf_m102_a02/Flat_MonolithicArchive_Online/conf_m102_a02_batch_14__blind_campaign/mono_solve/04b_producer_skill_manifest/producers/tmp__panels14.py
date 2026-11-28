import numpy as np,pickle
from scipy.optimize import minimize_scalar,least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
F={'sin':lambda x,p:np.sin(p*x),'cos':lambda x,p:np.cos(p*x),'tanh':lambda x,p:np.tanh(p*x),'rat':lambda x,p:x/(1+p*x*x),'rsq':lambda x,p:x/np.sqrt(1+p*x*x),'gauss':lambda x,p:x*np.exp(-p*x*x),'exp':lambda x,p:np.exp(-p*x*x),'sinh':lambda x,p:np.sinh(p*x),'atan':lambda x,p:np.arctan(p*x),'soft':lambda x,p:x/(1+p*abs(x)),'soft2':lambda x,p:x/(1+p*abs(x)**1.5),'expm':lambda x,p:np.expm1(p*x)}
def combos(x,kind):
 i,j=np.triu_indices(4,1)
 return {'single':lambda:x,'diff':lambda:x[:,i]-x[:,j],'sum':lambda:x[:,i]+x[:,j],'product':lambda:x[:,i]*x[:,j],'nextdiff':lambda:x-np.roll(x,-1,axis=1),'nextsum':lambda:x+np.roll(x,-1,axis=1),'oppdiff':lambda:x-np.roll(x,-2,axis=1)}[kind]()
def scan(u,v):
 out=[]
 for typ in ('single','diff','sum','product','nextdiff','nextsum','oppdiff'):
  a=combos(u,typ)
  for name,f in F.items():
   for lin in (0,1):
    def err(p):
     z=np.c_[aff(u) if lin else np.ones(len(u)),f(a,p)]; return np.mean((z@ls(z,v)-v)**2)
    grid=np.geomspace(.03,9,29); idx=np.argmin([err(p) for p in grid]); opt=minimize_scalar(err,bounds=(grid[max(idx-1,0)]*.999,grid[min(idx+1,len(grid)-1)]*1.001),method='bounded');out.append((opt.fun**.5,typ,name,lin,opt.x))
 i,j=np.triu_indices(4)
 for name,a in [('quad',np.c_[aff(u),u[:,i]*u[:,j]]),('cubicsep',np.c_[aff(u),u*u,u**3]),('cyclicquad',np.c_[aff(u),u*u,u*np.roll(u,-1,axis=1)])]:out.append((np.mean((a@ls(a,v)-v)**2)**.5,name,'-',0,0))
 return sorted(out)
if __name__=='__main__':
 for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u,v=S[p]['u'],S[p][k];print(p,k,scan(u,v)[:13]);
  for name,g in F.items():
   def res(w):return (g(aff(u)@w[1:21].reshape(5,4),w[0])*w[21:25]+w[25:]-v).ravel()
   w=np.r_[1.,ls(aff(u),v).ravel(),np.ones(4),np.zeros(4)]; fit=least_squares(res,w,max_nfev=150);print(name, np.mean(fit.fun**2)**.5,end='; ')
  print()
 import importlib.util
 f='/tmp/hegel_continual_archive/conf_m102_a02_batch_09_solver.py';sp=importlib.util.spec_from_file_location('a',f);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 p,b,w=m._calibrate(S['panel_a']);u,v=S['panel_a'].values();print('A',p,b,w,np.sort(np.linalg.norm(b+w*u+p[2]*np.tanh(p[0]*u)+p[3]*np.sin(p[1]*np.roll(u,-2,axis=1))-v,axis=1)))
