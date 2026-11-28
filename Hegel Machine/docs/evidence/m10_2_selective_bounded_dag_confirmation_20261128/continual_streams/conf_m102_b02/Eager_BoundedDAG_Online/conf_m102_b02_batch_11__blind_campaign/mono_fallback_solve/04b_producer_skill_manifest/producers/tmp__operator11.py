import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; q=np.load('/tmp/d11.npz'); z=q['z']; b=q['b']; Y=np.c_[q['y'],q['T']]; n=len(Y)
D=lambda u:np.c_[np.ones(len(u)),u]; fit=lambda F,V:np.linalg.lstsq(F,V,rcond=1e-9)[0]
p=s['panel_b']; u=p['u']; v=p['v'];
for name,F in {'aff':D,'cube':lambda u:D(np.c_[u,u**3]),'cyclic':lambda u:D(np.c_[u,u*np.roll(u,-1,1),u**3]),'quadratic':lambda u:D(np.c_[u,u*u,np.column_stack([u[:,i]*u[:,j] for i,j in combinations(range(4),2)])])}.items():
 f=F(u); r=f@fit(f,v)-v; h=np.sum(f*np.linalg.pinv(f).T,1); print('B',name,np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/(1-h[:,None]))**2)))
funcs={'asinh':lambda x,k:np.arcsinh(k*x),'atan':lambda x,k:np.arctan(k*x),'tanh':lambda x,k:np.tanh(k*x),'erf':lambda x,k:erf(k*x),'soft':lambda x,k:x/np.sqrt(1+(k*x)**2),'rat':lambda x,k:x/(1+(k*x)**2),'sin':lambda x,k:np.sin(k*x),'power':lambda x,k:np.sign(x)*np.abs(x)**k,'expsat':lambda x,k:np.sign(x)*(-np.expm1(-k*abs(x)))}
for name,fun in funcs.items():
 def r(k):
  f=D(fun(u,k));return (f@fit(f,v)-v).ravel()
 k=least_squares(r,[1],bounds=(.01,5)).x[0]; print('B nonlinear',name,k,np.sqrt(np.mean(r(k)**2)))
rng=np.random.default_rng(811); ij=[np.arange(n)]+[rng.permutation(n) for _ in range(20)]+[np.argsort(np.linalg.norm(b,axis=1))]; pairs=list(combinations(range(8),2))
def gram(b,fun,k,diag=False):
 x=np.c_[b,fun(b,k)]; return D(np.c_[x,x*x,np.column_stack([x[:,i]*x[:,j] for i,j in pairs if not diag or i%4==j%4 or i<4<=j])])
def stress(F,ridge):
 K=F@F.T; out=[]
 for p in ij:
  i,j=p[:52],p[52:]; e=abs(K[np.ix_(j,i)]@np.linalg.solve(K[np.ix_(i,i)]+ridge*np.eye(len(i)),Y[i])-Y[j]);out.append([e[:,0].mean(),e[:,1:].mean()]);
 E=np.array(out);return np.c_[E[0],E[1:-1].mean(0),E[1:-1].max(0),E[-1]]
for name,fun in funcs.items():
 out=[]
 for k in ([.35,.55,.75,1.4,1.8,2.15,3] if name=='power' else [.22,.45,.7,1,1.4,1.75,2,2.15,2.5,3,4]):
  for diag in (False,True):
   F=gram(b,fun,k,diag)
   for ridge in [1e-7,1e-5,.0003,.003]:
    E=stress(F,ridge);out.append((k,diag,ridge,E))
 for head in (0,1):
  r=min(out,key=lambda r:r[-1][head]@[0,.65,.2,.15]); print('HEAD',head,name,r[:3],r[-1][head].round(5))
