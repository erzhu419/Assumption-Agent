import pickle,itertools as it
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import qr
from sklearn.linear_model import lars_path
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; q=np.load('/tmp/explore.npz'); X,Y,T=q['x'],q['Y'],q['T']; z,b,d=q['z'],q['b'],q['d']; n=52
u=s['panel_c']['u']; v=s['panel_c']['v1']; np.set_printoptions(precision=5,suppress=True)
def fit(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def stack(*a):return np.column_stack(a)
forms={
 'shared':(lambda x,p:[x,np.tanh(p[0]*x)+p[1]*x/(1+(p[2]*x)**2)],[(a,k,f) for a in (1,2,4,6) for k in (-.4,.4,5) for f in (.6,1.3,2.3)],([.1,-40,.1],[15,40,6])),
 'separate':(lambda x,p:[np.tanh(p[0]*x),x/(1+(p[1]*x)**2)],list(it.product([1,2,4,6],[.6,1.3,2.3])),([.1,.1],[15,6])),
 'separate_lin':(lambda x,p:[x,np.tanh(p[0]*x),x/(1+(p[1]*x)**2)],list(it.product([1,2,4,6],[.6,1.3,2.3])),([.1,.1],[15,6])),
 'sine':(lambda x,p:[x,np.tanh(p[0]*x),np.sin(p[1]*x)],list(it.product([1,2,4],[.8,2,4])),([.1,.1],[15,12])),
 'tan_cub':(lambda x,p:[x,np.tanh(p[0]*x),x**3],[(a,) for a in (1,2,4)],([.1],[15]))}
models=[]
for name,(F,starts,bounds) in forms.items():
 def D(a,p):return stack(np.ones(len(a)),*F(a,p))
 def res(p):
  h=D(u,p);return (h@fit(h,v)-v).ravel()
 r=min((least_squares(res,a,bounds=bounds,max_nfev=140) for a in starts),key=lambda a:a.cost)
 H=D(u,r.x); w=fit(H,v); L=np.sum(H*np.linalg.pinv(H).T,axis=1); print(name,r.x,np.sqrt(np.mean(r.fun**2)),np.sqrt(np.mean((r.fun.reshape(v.shape)/(1-L[:,None]))**2)),w.round(4));models.append((name,D(z,r.x)@w))
models.append(('original',q['c']))
for name,c in models:
 A={'z':z,'b':b,'c':c,'d':d}
 def make(kind):
  F={}
  for k,a in A.items():
   if k=='b':continue
   for i in range(4):F[k+str(i)]=a[:,i]
  for k,l in it.combinations_with_replacement(A,2):
   if kind==0 and k==l=='b':continue
   for off in (range(4) if kind==2 else [0]):
    for i in range(4):
     j=(i+off)%4
     if k==l and i>j:continue
     F[f'{k}{i}*{l}{j}']=A[k][:,i]*A[l][:,j]
    if off==0:F[k+'dot'+l]=np.sum(A[k]*A[l],axis=1)
  for i,j in it.combinations_with_replacement(range(4),2): F[f'z{i}*z{j}']=z[:,i]*z[:,j]
  return F
 for kind in (0,2):
  F=make(kind); names=list(F); D=stack(*F.values()); m=D[:n].mean(0); sd=D[:n].std(0); D=(D-m)/sd
  def path(y):
   _,_,path=lars_path(D[:n],y[:n]-y[:n].mean(),method='lasso',max_iter=140,eps=1e-10)
   return [np.flatnonzero(w) for w in path.T if np.count_nonzero(w)<38]
  for k in (0,8):
   y=Y[:,k]; candidates=[]
   for ids in path(y):
    E=stack(np.ones(len(D)),D[:,ids]); pred=E@fit(E[:n],y[:n]); err=np.abs(pred-y); candidates.append((err[n:].mean(),err[:n].mean(),len(ids),','.join(names[i] for i in ids)))
   print('read',name,kind,k,sorted(candidates)[:2])
