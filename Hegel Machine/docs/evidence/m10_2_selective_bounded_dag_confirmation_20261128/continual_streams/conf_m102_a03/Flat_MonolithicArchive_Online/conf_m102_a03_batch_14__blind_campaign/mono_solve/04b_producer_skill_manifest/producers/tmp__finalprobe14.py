import sys,pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
from batch_solver import _features,_fit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; h,z,b,d,q=_features(s,tr['x']); n=len(y); p=q.sum(1); X=(d[:,:,None]*q[:,None,:]).reshape(n,16); rr=np.random.default_rng(34); ar=np.arange(n); folds=[(ar[:52],ar[52:])]+[(i[:52],i[52:]) for i in [rr.permutation(n) for _ in range(12)]]; folds+= [(np.r_[ar[:j],ar[j+20:]],ar[j:j+20]) for j in [0,13,26,39]]
for a in [0,.003,.01,.03,.1,.3,1,3]:
 A=np.c_[h,X]; es=[]; pred=[]
 for t,v in folds:
  sc=np.sqrt(np.mean(A[t]**2,axis=0)).clip(1e-9); B=A/sc; C=np.diag(np.r_[np.zeros(6),np.full(16,np.sqrt(a))]); c=_fit(np.r_[B[t],C],np.r_[y[t],np.zeros(22)]); es.append(np.mean(abs(B[v]@c-y[v]))); pred.append(B[v]@c)
 print('ridge',a,np.round([es[0],np.mean(es),max(es)],6))
# Profile shared nonlinear scalar modulation using only the fitting portion of each split.
for name,Z in [('z',z),('b',b),('d',d),('z+d',np.c_[z,d])]:
 for family in ['rat','tanh','cos','sq']:
  for mod in ['p','q','b']:
   M={'p':p[:,None],'q':q,'b':b}[mod]; E=[]; F=[]
   for t,v in folds[:8]:
    def des(Z,M,a):
     u=Z@a[:-1]+a[-1]; g={'rat':lambda:1/(1+u*u),'tanh':lambda:np.tanh(u),'cos':lambda:np.cos(u),'sq':lambda:u*u}[family]();return M*g[:,None]
    def r(a):
     A=np.c_[h[t],des(Z[t],M[t],a)];return A@_fit(A,y[t])-y[t]
    rg=np.random.default_rng(49); opt=min([least_squares(r,rg.normal(0,.7,Z.shape[1]+1),bounds=(-5,5),max_nfev=75) for _ in range(6)],key=lambda o:o.cost)
    A=np.c_[h,des(Z,M,opt.x)]; E.append(np.mean(abs(A[v]@_fit(A[t],y[t])-y[v]))); F.append(np.mean(abs(opt.fun)))
   if max(E)<.055 or np.mean(F)<.012:print(name,family,mod,np.round([E[0],np.mean(E),max(E),np.mean(F)],6),flush=True)
