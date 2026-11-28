from probe import *
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy.linalg import solve
R=np.c_[y,T]; ones=np.ones(len(x)); f=lambda a:np.c_[np.ones(len(a)),a,np.cos(1.27*a)]; j=f(x)@np.linalg.lstsq(f(u),C['v2'],rcond=None)[0]
# Panel v1: identify whether a single rational coordinate accounts for the residual.
for q in range(4):
 def vfit(p):
  F=np.c_[np.ones(len(u)),u,uz,u[:,q]/(1+p[0]*u[:,q]**2)]; return F@np.linalg.lstsq(F,C['v1'],rcond=None)[0]-C['v1']
 r=least_squares(lambda p:vfit(p).ravel(),[1.],bounds=(.05,10));print('v1 rational',q,r.x,np.std(r.fun))
base={'z':z,'diag':x*z,'next':x*np.roll(z,-1,1),'prev':x*np.roll(z,1,1),'opp':x*np.roll(z,2,1),'zz':z*z,'zznext':z*np.roll(z,-1,1),'zzopp':(z*np.roll(z,2,1))[:,:2],'xx':x*x,'xxnext':x*np.roll(x,-1,1),'x':x,'j':j,'cos':np.cos(1.27*x),'sinz':np.sin(z),'ratx':x/(1+x*x),'ratz':z/(1+z*z),'tanhz':np.tanh(z),'sinx':np.sin(x),'cubez':z**3,'cubex':x**3,'xzz':x*z*z,'xxz':x*x*z,'xcnext':x*np.cos(np.roll(z,-1,1)),'xsinz':x*np.sin(z),'zsinx':z*np.sin(x),'xztanh':x*np.tanh(z),'zxtanh':z*np.tanh(x),'sinxz':np.sin(x*z),'ratxz':x*z/(1+(x*z)**2),'ztanhj':z*np.tanh(j),'jj':j*j,'zj':z*j,'xj':x*j,'zjnext':z*np.roll(j,-1,1)}
outer=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=281).split(x))
# Use inner leave-one-out for grammar and shrinkage selection separately in every outer split.
def model(ind,groups,lam,yl=y):
 F=np.column_stack([base[k] for k in groups]); mu=F[ind].mean(0); sc=np.maximum(F[ind].std(0),.01); F=(F-mu)/sc; L=F[ind]; m=yl[ind].mean(0); P=solve(L.T@L+lam*np.eye(L.shape[1]),L.T,assume_a='pos'); coef=P@(yl[ind]-m); G=F@coef+m; h=1/len(ind)+np.sum(L*P.T,axis=1); e=(G[ind]-yl[ind])/(1-h); return np.mean(abs(e)),G,coef/sc
seeds=[('z',),('z','diag','next','prev','zz','zznext'),('z','diag','next','prev','opp','x'),('z','diag','next','prev','zz','zznext','j')]
for ii,(a,b) in enumerate(outer):
 pool=set(tuple(sorted(g)) for g in seeds); best=[]; seen=set()
 for depth in range(6):
  for g in pool-seen:
   for lam in (.001,.025,.15,.6):
    e,p,c=model(a,g,lam);best.append((e,g,lam,np.mean(abs(p[b]-y[b]))))
  seen|=pool;best.sort(); pool=set()
  for e,g,lam,_ in best[:18]:
   pool.update(tuple(sorted((*g,k))) for k in base if k not in g and sum(base[j].shape[1] for j in g)+base[k].shape[1]<=38);pool.update(tuple(j for j in g if j!=k) for k in g if k!='z')
 print('fold',ii,[(round(e,5),g,l,round(o,5)) for e,g,l,o in best[:5]],flush=True)
 if ii==0:
  for _,g,l,_ in best[:3]:print('coef',g,model(a,g,l)[2].round(3),flush=True)
