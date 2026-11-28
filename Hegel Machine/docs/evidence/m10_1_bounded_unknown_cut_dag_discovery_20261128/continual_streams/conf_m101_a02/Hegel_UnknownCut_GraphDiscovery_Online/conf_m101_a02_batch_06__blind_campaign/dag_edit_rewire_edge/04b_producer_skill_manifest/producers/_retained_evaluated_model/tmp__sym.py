from probe import *
from sklearn.model_selection import KFold
prev=x*np.roll(z,1,axis=1); q=z@[-1,-1,1,1]; w=prev.sum(1)
folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=281).split(x))
base=np.c_[np.ones(len(x)),q,w]; coef=np.linalg.lstsq(base[:52],y[:52],rcond=None)[0]; r=y-base@coef
print('base',coef,np.mean(abs(r[:52])),np.mean(abs(r[52:])),np.quantile(r,[0,.1,.5,.9,1]),'cor T',np.corrcoef(np.c_[r,T].T)[0,1:]); print('time',t)
D={}
for name,a in [('x',x),('z',z),('prev',prev),('diag',x*z),('next',x*np.roll(z,-1,1)),('opp',x*np.roll(z,2,1)),('xpair',x*np.roll(x,1,1)),('zpair',z*np.roll(z,1,1))]:
 D[name]=a
 for lab,f in [('2',a*a),('3',a**3),('sin',np.sin(a)),('tanh',np.tanh(a)),('rat',a/(1+a*a)),('cos',np.cos(a))]:D[name+lab]=f
 for lab,sign in [('sum',[1,1,1,1]),('half',[-1,-1,1,1]),('alt',[-1,1,-1,1]),('out',[1,-1,-1,1])]:
  v=a@sign; D[name+lab]=np.c_[v,v*v,v**3,np.sin(v),np.tanh(v),v/(1+v*v),np.cos(v)]
D['qw']=np.c_[q*w,q*q*w,q*w*w]; D['zopp']=(z*np.roll(z,2,1))[:,:2]
F=np.column_stack([v for v in D.values()]); names=[k+':'+str(i) for k,v in D.items() for i in range(v.shape[1])];
def assess(f):
 E=[]; cs=[]
 for a,b in folds:
  c=np.linalg.lstsq(f[a],y[a],rcond=None)[0]; E.append(np.mean(abs(f[b]@c-y[b])));cs.append(c)
 return np.array(E),cs[0]
for f in [base,np.c_[base,prev,z],np.c_[base,prev,z,T]]:print('baseline',assess(f))
S=[]
for k,v in D.items():
 e,c=assess(np.c_[base,v]);S.append((e[1:].mean(),e,k,c))
for _,e,k,c in sorted(S,key=lambda a:a[0])[:14]:print(k,e.round(4),c.round(4))
# Fit a single projected nonlinear residual alongside the tied cyclic product.
for tag,V in [('x',x),('z',z),('xz',np.c_[x,z]),('prev',prev),('xprev',np.c_[x,prev]),('zprev',np.c_[z,prev])]:
 for fname,func in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('rat',lambda q:q/(1+q*q)),('sq',np.square)]:
  es=[]; pars=[]
  for a,b in folds:
   def solve(p,ind):
    f=np.c_[base,func(V@p[:-1]+p[-1])]; c=np.linalg.lstsq(f[ind],y[ind],rcond=None)[0]; return f@c,c
   rng=np.random.default_rng(806); init=[np.r_[np.zeros(V.shape[1]),.2]+rng.normal(0,.4,V.shape[1]+1) for _ in range(5)]; opts=[least_squares(lambda p:solve(p,a)[0][a]-y[a],p,bounds=(-5,5),max_nfev=100) for p in init]; pp=min(opts,key=lambda r:np.linalg.norm(r.fun)).x; pred,c=solve(pp,a); es.append(np.mean(abs(pred[b]-y[b])));pars.append(np.r_[pp,c])
  print(tag,fname,np.round(es,4),pars[0].round(3),flush=True)
