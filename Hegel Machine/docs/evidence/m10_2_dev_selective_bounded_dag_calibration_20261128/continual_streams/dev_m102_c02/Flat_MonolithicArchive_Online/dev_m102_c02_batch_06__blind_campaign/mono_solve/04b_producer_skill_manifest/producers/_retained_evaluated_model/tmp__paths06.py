from graph06 import *
import sys
sys.path.insert(0,'/tmp/hegel_continual_archive')
from dev_m102_c02_batch_04_solver import _calibration
x=np.r_[s['train']['x'],s['test']['x']]; z=_calibration(s['panel_a'],x)
u,v=s['panel_a']['u'],s['panel_a']['v']; r=np.roll(u,-1,axis=1);d=4;j=np.arange(d);h=np.zeros((len(u),d,11));h[:,j,j]=1;h[:,j,d+j]=u;h[:,:,-3:]=np.stack((r,u*r,u**3),-1);keep=np.ones(len(u),bool)
for _ in range(10):
 p=fit(h[keep].reshape(-1,11),v[keep].ravel());e=np.linalg.norm(h@p-v,axis=1);keep=e<.6
print('cal params',p,'errors',np.sort(e),'z range',z.min(0),z.max(0))
B=op(**s['panel_b'],fn=soft,p=[.35,1.5],lim=([0,.1],[.8,8])); c=s['panel_c'];u=c['u'];v=c['v1'];w=c['v2'];C=op(u,v,power,[.9,1,1.25],([0,.1,.1],[8,8,8]))
a,L=terms(u,2); k=fit(a,w);print('C2 terms',list(zip(L,k.round(4).tolist())))
D=lambda x: terms(x,2)[0]@k
Y=np.c_[s['train']['y'],s['train']['traj']];out=[]
def score(n,p):
 for q in ([1,2,3] if p.shape[1]==4 else [1]):
  H,_=terms(p,q); P=H[52:72]@fit(H[:52],Y[:52]); e=np.mean(abs(P-Y[52:]),axis=0);out.append((e[0],e[1:].mean(),n,q))
Z={'x':x,'z':z}; F={'b':B,'c':C,'d':D}
for _ in range(3):
 for n,p in list(Z.items()):
  if len(n)==_+1:
   for f,fn in F.items():Z[f+n]=fn(p)
for n,p in Z.items():score(n,p)
for i,(n,p) in enumerate(Z.items()):
 if len(n)>2:break
 for m,r in list(Z.items())[i+1:]:
  if len(m)>2:break
  score(n+'+'+m,np.c_[p,r]);score(n+'*'+m,np.c_[p,r,p*r]);score(n+'~'+m,p+r);score(n+'-'+m,p-r)
for row in sorted(out,key=lambda e:e[0]+e[1])[:35]:print(row)
print('base',[(n,e,f,q) for e,f,n,q in out if n in ['z','bz','cz','dz','cbz','bcz','dbz','bdz']])
