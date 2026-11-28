from ops14 import *
from itertools import combinations_with_replacement
from scipy.linalg import solve
tr=s['train']; x=np.r_[tr['x'],s['test']['x']]; y=tr['y']; T=tr['traj']; Y=np.c_[y,T]; U,S,V=np.linalg.svd(T,full_matrices=False);print('traj sv',S,'range',np.std(Y,axis=0),'summary',tr['summary']);print('modes',V[:4]);
z=cal(s['panel_a'])(x)
a=s['panel_a'];u,v=a['u'],a['v'];k=np.max(abs(u-v),1)<3; h=.18*np.tanh(1.8*u)-.32*np.sin(2.4*np.roll(u,2,axis=1));print('archive cal09',np.std((D(u)@ls(D(u[k]),(v-h)[k])+h-v)[k]))
bp=s['panel_b'];cp=s['panel_c']; B=op(bp['u'],bp['v'],'cos',False)[3]; C=op(cp['u'],cp['v1'],'tanh_sin',False)[3]; E=op(cp['u'],cp['v2'],'exp',True)[3]
F={'z':z,'x':x}; seq=[('z',z)]
for d in range(3):
 seq=[(n+t,fn(a)) for n,a in seq for t,fn in [('B',B),('C',C),('E',E)] if not (d==2 and n[-1]==t)];F.update(seq)
print('z range',np.min(z,axis=0),np.max(z,axis=0))
rows=[]
def sq(a):return np.column_stack([a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(a.shape[1]),2)])
def errors(a,lam=0):
 a=a/(1e-7+a[:52].std(0)); a=np.column_stack([np.ones(len(a)),a]); X=a[:52];c=ls(X,Y[:52]) if not lam else solve(X.T@X+lam*np.eye(X.shape[1]),X.T@Y[:52],assume_a='pos');e=abs(a[52:72]@c-Y[52:]); return np.mean(e[:,0]),np.mean(e[:,1:]),np.max(np.mean(e[:,1:],axis=0))
for n,h in F.items():
 for spec,a in [('direct',h),('linear',np.c_[z,h]),('diag',np.c_[z,h,z*h,np.sum(z*h,axis=1)]),('within',np.c_[z,h,z*z,h*h,z*h]),('quad',np.c_[z,h,sq(z),sq(h),(z[:,:,None]*h[:,None,:]).reshape(len(z),16)]),('smooth',np.c_[z,h,z*h,np.tanh(2.15*h),np.arcsinh(2.15*h),h/(1+.22*h*h)])]:
  rows.append((*errors(a,.01),n,spec))
for n,h in F.items():
 if len(n)>3:continue
 for m,g in F.items():
  if len(m)>3 or m<=n:continue
  for spec,a in [('dual',np.c_[z,h,g]),('dualdiag',np.c_[z,h,g,z*h,z*g,h*g]),('dualplus',np.c_[z,h,g,z*h,z*g,h*g,z*z,h*h,g*g])]:rows.append((*errors(a,.03),n+'-'+m,spec))
for f in [lambda r:r[0],lambda r:r[1],lambda r:max(r[:2])]:
 print('ranking');print(*[(np.round(r[:3],5),*r[3:]) for r in sorted(rows,key=f)[:15]],sep='\n')
