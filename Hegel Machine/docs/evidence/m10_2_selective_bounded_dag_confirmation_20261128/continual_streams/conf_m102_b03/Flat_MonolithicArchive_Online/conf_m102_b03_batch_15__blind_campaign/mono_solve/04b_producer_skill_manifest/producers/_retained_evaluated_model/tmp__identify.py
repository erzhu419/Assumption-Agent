import itertools as it,pickle,numpy as np
from scipy.optimize import least_squares,minimize_scalar
from scipy.special import expit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; D=lambda z:np.c_[np.ones(len(z)),z]; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
def calibrate(panel,x,verbose=False):
 u,v=panel['u'],panel['v']; rng=np.random.default_rng(7); pairs=list(it.combinations(range(4),2))
 def des(u):return D(np.c_[u,u**3,*[u[:,i]*u[:,j] for i,j in pairs]])
 A=des(u); fits=[ls(A[g],v[g]) for g in (rng.choice(len(u),17,False) for _ in range(450))]; e=lambda c:np.max(abs(A@c-v),axis=1); c=min(fits,key=lambda c:np.quantile(e(c),.72)); good=e(c)<.001;c=ls(A[good],v[good]); c[abs(c)<.003]=0
 for j in range(4):
  i=c[:,j]!=0;c[i,j]=ls(A[good][:,i],v[good,j])
 if verbose:
  print('calibration n/rmse',sum(good),np.mean((A[good]@c-v[good])**2)**.5);print(c.T.round(5));print(pairs)
 z=(x-c[0])@np.linalg.inv(c[1:5]);eye=np.eye(4)
 for _ in range(30):
  jac=np.broadcast_to(c[1:5].T,(len(z),4,4)).copy()+c[5:9].T[None,:,:]*(3*z*z)[:,None,:]
  for (i,j),q in zip(pairs,c[9:]):jac[:,:,i]+=z[:,j,None]*q;jac[:,:,j]+=z[:,i,None]*q
  delta=np.linalg.solve(jac,(des(z)@c-x)[...,None])[...,0];z-=delta/np.maximum(1,np.linalg.norm(delta,axis=1)[:,None]);
 if verbose:print('inverse max',np.max(abs(des(z)@c-x)))
 return z
fun={'tanh':np.tanh,'sin':np.sin,'cos':np.cos,'atan':np.arctan,'softsign':lambda x:x/(1+abs(x)),'swish':lambda x:x*expit(x),'softplus':lambda x:np.logaddexp(0,x),'cubic':lambda x:x**3,'rational':lambda x:x/(1+x*x),'square':lambda x:x*x}
def panel_candidates(panel,key,z):
 u=panel['u'];v=panel[key]; outs=[]
 for name,f in fun.items():
  for typ in ('alone','linear','adjacent','linear_adjacent','opposite','linear_opposite','cubic','constrained_rational'):
   def a(x,k):
    w=f(k*x);adj=x*np.roll(x,-1,axis=1);opp=(x*np.roll(x,2,axis=1))[:,:2]
    return D({'alone':w,'linear':np.c_[x,w],'adjacent':np.c_[adj,w],'linear_adjacent':np.c_[x,adj,w],'opposite':np.c_[opp,w],'linear_opposite':np.c_[x,opp,w],'cubic':np.c_[x**3,w],'constrained_rational':np.c_[x/(1+.22*x*x),w]}[typ])
   def err(k):
    A=a(u,k);Q=np.linalg.svd(A,full_matrices=False)[0][:,:np.linalg.matrix_rank(A)];return np.mean(((Q@(Q.T@v)-v)/(1-(Q**2).sum(axis=1,keepdims=True)))**2)**.5
   grid=np.linspace(.3,3.8,12);i=np.argmin([err(k) for k in grid]);k=minimize_scalar(err,bounds=(grid[max(0,i-1)],grid[min(i+1,len(grid)-1)]),method='bounded').x
   A=a(u,k); c=ls(A,v);outs.append((err(k),name,typ,k,a(z,k)@c,np.mean((A@c-v)**2)**.5,c))
 return sorted(outs,key=lambda a:a[0])
if __name__=='__main__':
 tr=s['train']; z=calibrate(s['panel_a'],np.r_[tr['x'],s['test']['x']],True)
 P=panel_candidates(s['panel_c'],'v1',z);Q=panel_candidates(s['panel_c'],'v2',z)
 for k,C in enumerate((P,Q)):
  print('branch',k);print(*[(round(e,7),n,t,round(f,5),round(r,7)) for e,n,t,f,_,r,c in C[:12]],sep='\n');print(C[0][-1].round(4))
 np.savez('/tmp/probe_features.npz',z=z,p=np.array([m[4] for m in P[:12]]),q=np.array([m[4] for m in Q[:12]]))
