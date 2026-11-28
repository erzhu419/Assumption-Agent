from sink_probe import *
from scipy.spatial.distance import cdist
import runpy
mod=runpy.run_path('/holdout_task/node_01.py'); folds=[np.arange(52,N)]+list(np.array_split(np.random.default_rng(30).permutation(N),6))

def evaluate(K,B,lam):
 out=[]
 for v in folds:
  a=np.setdiff1d(np.arange(N),v); ka=K[np.ix_(a,a)]+lam*np.eye(len(a)); ba=B[a]; M=np.linalg.solve(ka,np.c_[ba,Y[a]]); P=M[:,:ba.shape[1]]; R=M[:,ba.shape[1]:]; c=np.linalg.lstsq(ba.T@P,ba.T@R,rcond=1e-10)[0]; e=abs(B[v]@c+K[np.ix_(v,a)]@(R-P@c)-Y[v]);out.append([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max()])
 return np.array(out)
res=[]
for key in ('xa','xab','z','white'):
 x=d[key] if key in d else (d['xa']-d['xa'].mean(0))@np.linalg.inv(np.linalg.cholesky(np.cov(d['xa'].T))).T
 for kind,xx in [('raw',x),('tanh',np.tanh(2.45*x)),('cos',np.cos(1.33*x)),('mixed',np.c_[x,np.tanh(2.45*x),np.cos(1.33*x)])]:
  D=cdist(xx,xx,'sqeuclidean'); dist=np.median(D)
  for shape in (.15,.45,1.3,4.):
   R=D/(dist*shape); rt=np.sqrt(3*R); kernels=[np.exp(-R/2),(1+rt)*np.exp(-rt),(1+R/4)**-2]
   for ix,K in enumerate(kernels):
    for deg in (0,1,2):
     for lam in (.0003,.003,.03):
      m=evaluate(K,poly(x,deg),lam);res.append(((key,kind,shape,ix,deg,lam),m))
for q,col in [('scalar',0),('traj',1)]:
 for cfg,m in sorted(res,key=lambda r: r[1][:,col].mean()+.5*r[1][:,col].max())[:7]: print(q,cfg,np.round(m[:,col],4))
for names,f,p,col in [(('xa','xab'),1.8,.1,0),(('z','xab'),.6,.01,1)]:
 m=[]
 for v in folds:
  a=np.setdiff1d(np.arange(N),v); x,w=(d[k] for k in names); e=abs(mod['_predict'](x[a],x[v],w[a],w[v],Y[a,0] if col==0 else Y[a,1:],f,p)-(Y[v,0] if col==0 else Y[v,1:]));m.append((e.mean(),np.max(e.mean(0))))
 print('baseline',names,np.round(m,5))
