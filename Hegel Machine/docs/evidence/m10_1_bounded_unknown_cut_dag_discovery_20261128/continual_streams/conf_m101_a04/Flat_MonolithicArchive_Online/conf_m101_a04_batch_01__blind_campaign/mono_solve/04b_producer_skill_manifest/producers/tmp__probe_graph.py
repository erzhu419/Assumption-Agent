import pickle,itertools
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import qr
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
T=S['train']; t=S['eval_times']; Y=np.c_[T['y'],T['traj']]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def poly(x,d=2):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for k in range(1,d+1) for i in itertools.combinations_with_replacement(range(x.shape[1]),k)])
def ops(s,raw):
 u,v=[s['panel_a'][k] for k in ['u','v']]; m=np.ones(len(u),bool)
 for i in range(8):
  a,b=u[m],v[m]; ac=a-a.mean(0); bc=b-b.mean(0); scale=(ac*bc).sum(0)/(ac**2).sum(0); off=b.mean(0)-scale*a.mean(0); e=np.linalg.norm(u*scale+off-v,axis=1);m=e<max(5*np.median(e),.0006)
 x=(raw-off)/scale
 p=s['panel_b'];b=poly(x,1)@ls(poly(p['u'],1),p['v']);p=s['panel_c'];u=p['u'];c=poly(x,2)@ls(poly(u,2),p['v1']);
 def f(a,w):return np.c_[poly(a,1),np.cos(w*a)]
 def loss(w):
  a=f(u,w);return np.mean((a@ls(a,p['v2'])-p['v2'])**2)
 w=minimize_scalar(loss,bounds=(.6,2),method='bounded').x;d=f(x,w)@ls(f(u,w),p['v2'])
 return {'x':x,'b':b,'c':c,'d':d}

def ridge(F,A,Q,l):
 if l==0:return F@ls(A,Q)
 return (F@A.T)@np.linalg.solve(A@A.T+l*np.eye(len(A)),Q)
if __name__=='__main__':
 np.set_printoptions(precision=4,suppress=True);x=ops(S,T['x']); p=S['panel_c'];print('panel C v1 coefs',ls(poly(p['u'],2),p['v1']))
 vn=np.linalg.svd(T['traj'][:52],full_matrices=False)[2][:2]; q=np.c_[Y[:,0],T['traj']@vn.T];print('targets std',q.std(0),'corr',np.corrcoef(q.T))
 for l in range(1,5):
  for k in itertools.combinations(x,l):
   z=np.column_stack([x[j] for j in k]); out=[]
   for deg in [1,2,3]:
    F=poly(z,deg); sc=np.maximum(.15,F[:52].std(0));sc[0]=1;F=F/sc; a=F[:52];v=F[52:]; res=[]
    for reg in [0,.0001,.01,.1,1]:
     pred=ridge(v,a,q[:52],reg); res.append(np.r_[np.mean(abs(pred[:,0]-q[52:,0])),np.mean(abs(pred[:,1:]@vn-T['traj'][52:]))]);
    res=np.array(res); out.append(np.round(res.min(0),4).tolist())
   print(''.join(k),out)
 for name,z in x.items():
  F=poly(z,2); coef=ls(F[:52],q[:52]); e=F@coef-q; print(name,'coef',np.round(coef,3).T,'train RMSE',np.sqrt((e[:52]**2).mean(0)))
