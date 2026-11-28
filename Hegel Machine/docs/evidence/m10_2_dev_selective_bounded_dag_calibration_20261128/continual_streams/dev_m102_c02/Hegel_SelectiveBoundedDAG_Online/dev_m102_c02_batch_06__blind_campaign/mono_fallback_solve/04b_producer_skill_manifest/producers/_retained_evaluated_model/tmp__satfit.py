from graph import *
from scipy.linalg import solve
A,inv=Amap(s['panel_a']); x=inv(s['train']['x']); y=np.c_[s['train']['y'],s['train']['traj']];rng=np.random.default_rng(23); folds=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(72) for _ in range(3)]]; P=poly(x,1)
ix,jx=np.triu_indices(4); i0,j0=np.triu_indices(4,1)
def features(x,q,fn,mode):
 z=fn(x*np.exp(q[0:4])+q[4:8]); gamma=np.roll(x,-1,axis=1)
 F=[np.ones((len(x),1)),z]
 if mode!=5:F+=[x]
 if mode in (0,1,2,3,5,6):F += [z[:,ix]*z[:,jx]]
 if mode==1:F += [x[:,ix]*x[:,jx]]
 if mode in (2,3):F+=[(x[:,:,None]*z[:,None,:]).reshape(len(x),16)]
 if mode==3:F+=[x[:,ix]*x[:,jx]]
 if mode==4:F+=[(x[:,:,None]*z[:,None,:]).reshape(len(x),16)]
 if mode==6:F+=[z*x,z*gamma,z*np.roll(x,1,axis=1)]
 return np.column_stack(F)
forms={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'soft':lambda w:w/(1+abs(w)),'rat':lambda w:w/(1+w*w),'sat':lambda w:w/np.sqrt(1+w*w),'clip':lambda w:np.clip(w,-1,1),'erf':__import__('scipy.special',fromlist=['erf']).erf}
def run(name,mode,reg,shift,target,fold,starts):
 i,j=fold;fn=forms[name]; Y=y[:,target]; best=None
 def mats(q):return features(x,np.r_[q,np.zeros(8-len(q))],fn,mode)
 def calc(q):
  Z=mats(q);mu=Z[i].mean(0); sd=np.maximum(Z[i].std(0),.06); X=(Z-mu)/sd;X[:,0]=1.; W=X[i]; G=W.T@W+np.eye(W.shape[1])*reg; B=solve(G,W.T@Y[i],assume_a='pos');return X@B,B
 def fun(q):
  p,b=calc(q);return np.r_[(p[i]-Y[i]).ravel(),(np.sqrt(reg)*b).ravel(),.008*q]
 for q in starts:
  opt=least_squares(fun,np.r_[np.full(4,q),np.zeros(4)][:4+shift],bounds=(-1.8,1.8),max_nfev=100)
  loss=np.linalg.norm(opt.fun)
  if best is None or loss<best[0]:
   pred,_=calc(opt.x);best=(loss,np.mean(abs(pred[j]-Y[j])),opt.x,np.mean(abs(pred[i]-Y[i])))
 return best
if __name__=='__main__':
 for target in [0,slice(1,None),slice(None)]:
  res=[]
  for name in forms:
   for mode in range(7):
    for shift in [0,4]:
     for reg in [.0001,.035]:
      b=run(name,mode,reg,shift,target,folds[0],[.15,.55]);res.append((b,(name,mode,reg,shift)))
  for b,h in sorted(res,key=lambda w:w[0][1])[:9]:
   print(str(target),h,b,'rnd',[round(run(*h,target,f,[.15,.55])[1],5) for f in folds[1:]],flush=True)
