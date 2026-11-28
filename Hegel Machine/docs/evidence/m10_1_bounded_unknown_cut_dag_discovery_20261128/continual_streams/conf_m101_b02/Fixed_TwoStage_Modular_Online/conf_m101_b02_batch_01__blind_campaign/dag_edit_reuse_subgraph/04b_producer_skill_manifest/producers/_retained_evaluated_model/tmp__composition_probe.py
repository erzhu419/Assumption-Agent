import pickle,runpy,itertools as it
import numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; f=runpy.run_path('/holdout_task/node_00.py'); sink=runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['run_node']; n=len(s['train']['y'])
a,A=f['_calibrate'](s['panel_a']); z=np.linalg.solve(A.T,(s['train']['x']-a).T).T
b,c=s['panel_b'],s['panel_c']; pf=f['_panel_fit']; B=pf(b['u'],b['v'],z,lambda u:(np.tanh(3.51*u),u/(1+(2.434*u)**2))); C=pf(c['u'],c['v1'],z,lambda u:(np.sin(1.5*u),u*abs(u))); D=pf(c['u'],c['v2'],z,lambda u:(u*u,np.cos(2.5*u)))
Y=np.column_stack((s['train']['y'],s['train']['traj'])); print('outcome corr',np.round(np.corrcoef(Y.T)[0],3)); U,S,V=np.linalg.svd(Y[:,1:],full_matrices=False); T=np.c_[Y[:,0],U[:,:2]*S[:2]]; Q=np.c_[z,B,C,D]; Q=(Q-Q.mean(0))/Q.std(0); E=np.c_[np.ones(n),Q]; ts=T.std(0); T=T/ts
rng=np.random.default_rng(38); folds=[np.arange(n-20,n),*np.array_split(rng.permutation(n),5)]
def score(fun):
 err=[]
 for v in folds:
  t=np.setdiff1d(np.arange(n),v); G=fun(t); h=sink({'node_00':{'g':G[np.r_[t,v]],'fd':B[np.r_[t,v]],'cd':np.c_[C,D][np.r_[t,v]],'y':Y[t,0],'traj':Y[t,1:]}}); e=abs(Y[v]-np.c_[h['y_hat'],h['state_pred']]);err.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.round(err[0],5),np.round(np.max(err,axis=0),5)
blocks=[z,B,C,D]; print('updated panels',score(lambda t:E))
for name,G in [('channel products',np.column_stack([j*k for j,k in it.combinations(blocks,2)])),('panel norms',np.c_[*[np.sum(j*j,axis=1) for j in blocks],*[np.sum(j*k,axis=1) for j,k in it.combinations(blocks,2)]]),('all triple z',np.c_[*[np.prod(z[:,i],axis=1) for i in it.combinations(range(4),3)]]),('channel cross saturations',np.c_[*[np.tanh(j+k) for j,k in it.combinations(blocks,2)],*[np.sin(j-k) for j,k in it.combinations(blocks,2)]])]: print(name,score(lambda t:np.c_[E,G]))
# Learn only projection directions inside each study split, then let the frozen sink fit its readouts.
for k,atoms in [(4,['sin','tanh','square']),(8,['sin','tanh','square']),(16,['sin','tanh','square'])]:
 X=Q[:,:k];
 for atom in atoms:
  def feature(t):
   L=E[t]; pinv=np.linalg.pinv(L); proj=lambda v:v-L@(pinv@v); R=proj(T[t]); out=[]
   fn={'sin':np.sin,'tanh':np.tanh,'square':np.square}[atom]
   for j in range(3):
    def cost(p):
     w,h=p[:-1],p[-1]; q=proj(fn(X[t]@w+h)); coef=(q@R[:,j])/(q@q+1e-8);return np.r_[q*coef-R[:,j],.018*w,.008*h]
    r=min((least_squares(cost,p,max_nfev=90) for p in np.random.default_rng(19).normal(0,.42,(5,k+1))),key=lambda r:r.cost)
    out.append(fn(X@r.x[:-1]+r.x[-1]));
   return np.c_[E,*out]
  print('index',k,atom,score(feature),flush=True)
