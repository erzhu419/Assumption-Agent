import sys;sys.path.insert(0,'/holdout_task')
from batch_solver import *
from scipy.optimize import least_squares
from itertools import combinations_with_replacement as cr
import pickle
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=s['train']; y=T['y']; g=T['traj']; z,b,e,C,W=features(s,T['x']); p=P(b); c=D(p)@C; eb=E(b)@W; ce=D(P(b+.5*e))@C; n=len(z)
sp=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(88).permutation(n),4)]
print('telemetry',s['telemetry_vec'],'summary',T['summary']); print('yq',np.quantile(y,[0,.1,.5,.9,1])); print('c moments',c.mean(0),c.std(0)); print('z coef y',fit(D(z),y)); print('p coef y',fit(D(np.c_[z,p]),y));
def quad(x):return np.c_[D(x),*[x[:,a]*x[:,b] for a,b in cr(range(x.shape[1]),2)]]
x=np.c_[.22*z,p]; X=quad(x); coef=fit(X,g); print('state coef last',coef[:,-1].round(4)); print('state temporal rank',np.linalg.svd(coef,compute_uv=False)); print('y/amplitude',np.c_[y,g[:,-1],y-D(z)@fit(D(z),y)][:20].round(3));
ff={'z':z,'b':b,'p':p,'c':c,'e':e,'eb':eb,'ce':ce,'zp':np.c_[z,p],'zb':np.c_[z,b],'zbc':np.c_[z,b,c],'zc':np.c_[z,c],'zce':np.c_[z,ce]}; out=[]
for key in ['p','c','b','eb','ce','zp','zb','zce']:
 v=ff[key];
 for nk in ['z','zc','zbc','zce']:
  A=D(ff[nk]);
  for mode in [0,1,2]:
   er=[]; params=[]; train=[]
   for i,j in sp:
    U=D(v)/(np.sqrt(np.mean(D(v)[i]**2,axis=0))+.03); H=A/(np.sqrt(np.mean(A[i]**2,axis=0))+.03); L=D(z); m=U.shape[1]; I=np.eye(H.shape[1]+(L.shape[1] if mode else 0));
    def design(q):
     den=1+(U@q)**2
     if mode==2: den=np.sqrt(den)
     F=H/den[:,None]; return np.c_[L,F] if mode else F
    def fun(q,ret=False):
     F=design(q); X=F[i]; coef=np.linalg.solve(X.T@X+.004*I,X.T@y[i]); pred=F@coef
     return pred if ret else np.r_[pred[i]-y[i],.02*q,.04*coef]
    rng=np.random.default_rng(12); starts=[np.eye(m)[k]*.6 for k in range(m)]+list(rng.normal(0,.45,(4,m))); r=min((least_squares(fun,st,max_nfev=95) for st in starts),key=lambda r:r.cost); yh=fun(r.x,True); er.append(np.mean(abs(yh[j]-y[j])));train.append(np.std(yh[i]-y[i]));params.append(r.x)
   out.append((np.mean(er),max(er),er[0],key,nk,mode,np.mean(train),params[0].round(3)))
for a in sorted(out)[:12]:print('R mean/worst/gate',a)
for a in sorted(out,key=lambda x:x[2])[:5]:print('R gate',a)
