from ops import *
from scipy.optimize import least_squares
F=features(x);z=F['z'];q=F['zB']; I=np.arange(52);J=np.arange(52,72)
B=np.c_[-np.expm1(-times),times]; A=ls(B,T.T).T; Y=np.c_[y,A[:,0]]
P=list(itertools.combinations_with_replacement(range(4),2)); sq=lambda a:np.column_stack([a[:,j]*a[:,k] for j,k in P]); np.set_printoptions(precision=4,suppress=True)
for c in [1,1.25,1.35,1.5]:
 d=y-c*A[:,0]
 for k in ['z','zB','zC','zBC','zD']:
  a=aff(F[k]);p=a@ls(a[I],d[I]);print('DIFF',c,k,np.mean(abs(p[I]-d[I])),np.mean(abs(p[J]-d[J])))
# Fit compact coordinate mixtures by variable projection, compare independent linear and quadratic paths.
S=[('B',q),('z',z),('C',F['zC']),('D',F['zD']),('BC',F['zBC']),('BD',F['zBD'])]; res=[]
for base,w in S[:2]:
 for nm,h in S:
  if nm==base:continue
  for route in range(4):
   def design(p):
    m=w+p[0]*h
    l=[m,w,np.c_[w,h],h][route]
    return aff(np.c_[l,sq(m)])
   def err(p):
    a=design(p);return (a[I]@ls(a[I],Y[I])-Y[I]).ravel()
   r=min([least_squares(err,[k],bounds=(-4,4),max_nfev=45) for k in [0.,-.9,.9]],key=lambda r:r.cost);a=design(r.x);e=a@ls(a[I],Y[I])-Y
   res.append((np.mean(abs(e[I]),0),np.mean(abs(e[J]),0),base,nm,route,r.x))
for k in range(2):print('MIX',k,*sorted(res,key=lambda r:r[k][0])[:7],sep='\n')
res=[]
for bn,w in S[:2]:
 for ln,l0 in S[:3]:
  for gn,g in fn.items():
   for mode in range(4):
    def design(p):
     if mode<2:u=w+p[0]*g(w,p[1]); l=(u if ln==bn else l0) if mode==0 else l0
     else:
      # panel B's unmixed inputs, or a variable power series in latent coordinates
      u=z+p[0]*g(z,p[1]);l=l0 if mode==2 else u
     return aff(np.c_[l,sq(u)])
    def err(p):
     a=design(p);return (a[I]@ls(a[I],Y[I])-Y[I]).ravel()
    for init in [[-.4,2.9],[.5,2.2],[-.6,1.25],[.8,4]]:
     r=least_squares(err,init,bounds=([-3,.15],[3,8]),max_nfev=45);a=design(r.x);e=a@ls(a[I],Y[I])-Y;res.append((np.mean(abs(e[I]),0),np.mean(abs(e[J]),0),bn,ln,gn,mode,r.x))
for k in range(2):print('WARP',k,*sorted(res,key=lambda r:r[k][0])[:12],sep='\n')
# Higher precision test: fit a shared odd residual on the warped latent axes, with independent linear paths.
a0=aff(np.c_[q,z,sq(q)]);c=ls(a0[I],Y[I]);r=Y-a0@c
print('BOTH',np.mean(abs(r[I]),0),np.mean(abs(r[J]),0),np.corrcoef(r.T))
for i in range(4):print('across',i,np.corrcoef(np.c_[z[:,i],q[:,i],r].T))
