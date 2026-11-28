from ops import *
from scipy.optimize import least_squares
F=features(x);z=F['z'];q=F['zB']; Z=np.c_[y,T]; I=np.arange(52); J=np.arange(52,72)
np.set_printoptions(precision=4,suppress=True,linewidth=150)
# Determine whether panel residual is systematic in the separable warp or in cross-coordinate terms.
for n,p,k,g in [('A','panel_a','v','sin'),('B','panel_b','v','sin'),('C','panel_c','v1','tanh'),('D','panel_c','v2','cos')]:
 u=s[p]['u'];v=s[p][k];ii=np.linalg.norm(v-u,axis=1)<2 if n=='A' else np.ones(len(u),bool);u=u[ii];v=v[ii];e=v-ops[n](u);print('panel residual',n,np.round(e[:7],4),np.std(e,0))
# Identify omitted bases and shared residual against response and trajectory amplitude.
P=list(itertools.combinations_with_replacement(range(4),2)); ts=np.c_[1-np.exp(-times),times];amp=ls(ts,T.T).T; Y=np.c_[y,amp];al=lambda a:np.c_[np.ones(len(a)),a]; b=al(np.c_[q,*[q[:,i]*q[:,j] for i,j in P]]); print('Q quadratic coefficients',ls(b[I],Y[I]));
D={};
for n,a in F.items():
 if '+' in n or '-' in n or '*' in n:continue
 for i in range(4):D[n+str(i)]=a[:,i]
 for i,j in P:
  D[f'{n}{i}{j}']=a[:,i]*a[:,j]
  for sg in [-1,1]:
   if i==j and sg==-1:continue
   h=a[:,i]+sg*a[:,j] if i!=j else a[:,i]
   for w in [.22,.6,1.25,2.15,2.9,4.]:
    for f,g in fn.items():D[f'{n}{i}{j}/{sg}/{f}/{w}']=g(h,w)
 for i,j,k in itertools.combinations_with_replacement(range(4),3):D[f'{n}{i}{j}{k}']=a[:,i]*a[:,j]*a[:,k]
for n in ['z','zB','zC','zD','zBC','zBD']:
 for m in ['z','zB','zC','zD','zBC','zBD']:
  if n>=m:continue
  for i,j in itertools.product(range(4),repeat=2):D[f'{n}{i}*{m}{j}']=F[n][:,i]*F[m][:,j]
keys=list(D); A=np.column_stack(list(D.values()));
for j in [0,1,2]:
 bb=b.copy()
 for it in range(6):
  c=ls(bb[I],Y[I,j]);r=Y[:,j]-bb@c; aa=A-bb@ls(bb[I],A[I]); norm=np.linalg.norm(aa[I],axis=0); cor=abs(r[I]@aa[I])/np.maximum(1e-10,norm);idx=np.argsort(cor)[-6:][::-1];print('OMP',j,it,'errors',np.mean(abs(r[I])),np.mean(abs(r[J])),[(keys[k],round(cor[k],4)) for k in idx]);bb=np.c_[bb,A[:,idx[0]]]
# Stacked coordinate-warp quadratic feature tests with train-only error.
for n in ['z','zB','zBC','zBD']:
 a=F[n]; results=[]
 for g in fn:
  for w in np.linspace(.25,5,39):
   for lam in [-1,-.6,-.4,-.2,.2,.5,1,2]:
    zz=a+lam*fn[g](a,w); d=al(np.c_[zz,*[zz[:,i]*zz[:,j] for i,j in P]]); c=ls(d[I],Y[I]); e=d@c-Y; h=(np.linalg.qr(d[I])[0]**2).sum(1); results.append((np.mean(abs(e[I]/(1-h[:,None])),0),np.mean(abs(e[J]),0),g,w,lam))
 for j in [0,1,2]:print('warp',n,j,*sorted(results,key=lambda v:v[0][j])[:2])
