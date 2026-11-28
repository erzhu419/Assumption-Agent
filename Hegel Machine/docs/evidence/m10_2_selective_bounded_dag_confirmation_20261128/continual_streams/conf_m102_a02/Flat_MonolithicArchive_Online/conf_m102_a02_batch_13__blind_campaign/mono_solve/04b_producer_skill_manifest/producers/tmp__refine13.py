exec(open('/tmp/graph13.py').read().split('h=tr[')[0])
from sklearn.model_selection import KFold
b=bs['cosx'](x); y=tr['y']; A=aff(b); Q=quad(b); cross=Q[:,np.r_[np.arange(5),[6,7,8,10,11,13]]]; diag=Q[:,np.r_[np.arange(5),[5,9,12,14]]]; rng=np.random.default_rng(42)
starts=np.r_[np.eye(5)*.45,np.c_[np.zeros(4),-.45*np.eye(4)],rng.normal(0,.45,(6,5))]; folds=[(np.arange(52),np.arange(52,72)),*KFold(4,shuffle=True,random_state=41).split(b)]
for label,P in [('aff',A),('cross',cross),('diag',diag),('quad',Q)]:
 for kind in ('sin','tanh','rat','split','sinlin','signedrat'):
  out=[]; ps=[]; bb=[]
  for i,j in folds:
   def design(p):
    a=A@p; g=1/(1+a*a)
    f={'sin':np.sin(a),'tanh':np.tanh(a),'rat':g,'split':(g-1)[:,None]*A,'sinlin':np.sin(a)[:,None]*A,'signedrat':a*g}[kind]
    return np.c_[P,f]
   def fit(p):
    Z=design(p)[i]; return Z@ls(Z,y[i])-y[i]
   cand=min((least_squares(fit,p,bounds=(-4,4),max_nfev=130) for p in starts),key=lambda q:q.cost); Z=design(cand.x); c=ls(Z[i],y[i]); out.append([np.mean(abs(Z[j]@c-y[j])),np.sqrt(np.mean(cand.fun**2))]); ps.append(cand.x); bb.append(c)
  print(label,kind,np.round(out,4).T.tolist(),np.round(ps[0],3),np.round(bb[0],3),flush=True)
