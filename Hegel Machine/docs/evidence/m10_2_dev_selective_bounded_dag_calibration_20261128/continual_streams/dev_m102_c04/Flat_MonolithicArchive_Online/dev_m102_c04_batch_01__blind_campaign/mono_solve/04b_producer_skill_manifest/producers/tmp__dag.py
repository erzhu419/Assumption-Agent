from bases import *
from itertools import product,combinations
u=s['panel_c']['u']; pairs=[(0,1),(0,3),(1,2),(2,3)]
def qb(z):return np.c_[np.ones(len(z)),*[z[:,i]*z[:,j] for i,j in pairs]]
wq=ls(qb(u),s['panel_c']['v1']); wh=ls(np.c_[np.ones(len(u)),np.cos(2.4*u)],s['panel_c']['v2'])
def Q(z):return qb(z)@wq
def H(z):return np.c_[np.ones(len(z)),np.cos(2.4*z)]@wh
B=lambda z:poly(z,1)@cb
Bi=lambda z:(z-cb[0])@np.linalg.inv(cb[1:])
A=lambda z:z*sc+ca[0]
Ai=lambda z:(z-ca[0])/sc
ops={'B':B,'b':Bi,'Q':Q,'H':H}
def paths(x,depth=3):
 a={'x':x,'A':A(x),'a':Ai(x)}
 for k in range(depth):
  for n,z in list(a.items()):
   if len(n)!=k+1:continue
   for op,fn in ops.items():
    if n[-1]+op not in ('Bb','bB'):a[n+op]=fn(z)
 return a
if __name__=='__main__':
 print('telemetry',s['telemetry_vec'],'summary',s['train']['summary'],'H weights',wh,'H rmse',np.std(H(u)-s['panel_c']['v2']));d=paths(x)
 modes={'U':lambda z:z,'P':lambda z:poly(z,2)[:,5:],'F':lambda z:np.c_[np.sin(2.4*z),np.cos(2.4*z)],'R':lambda z:np.c_[z/(1+.22*z*z),z*z/(1+.22*z*z)],'T':lambda z:np.tanh(1.25*z),'abs':lambda z:abs(z)}
 vs=np.c_[y,t]; out=[]
 for k,z in d.items():
  for m,fn in modes.items():
   for n,f in [('raw',np.c_[poly(x,2),fn(z)]),('local',np.c_[poly(z,1),fn(z)]),('both',np.c_[poly(x,2),z,fn(z)])]:
    w=ls(f[:52],vs[:52]);e=np.abs(f@w-vs);out.append((k,m,n,e[:52,0].mean(),e[52:,0].mean(),e[:52,1:].mean(),e[52:,1:].mean()))
 for col in (3,4,5,6):print('rank',col,*sorted(out,key=lambda a:a[col])[:10],sep='\n')
 d={k:z for k,z in d.items() if len(k)<=3}; out=[]
 for (k,z),(j,zz) in combinations(d.items(),2):
  if k[0]!=j[0]:continue
  for m,f in [('add',np.c_[poly(x,2),z,zz]),('prod',np.c_[poly(x,1),z,zz,z*zz]),('div',np.c_[poly(x,1),z,zz,z/(1+.22*zz**2),zz/(1+.22*z**2)]),('mix',np.c_[poly(x,1),z,zz,np.einsum('ni,nj->nij',z,zz).reshape(-1,16)])]:
   f=f/np.maximum(np.sqrt(np.mean(f[:52]**2,axis=0)),1e-10);w=np.linalg.lstsq(np.r_[f[:52],.05*np.eye(f.shape[1])],np.r_[vs[:52],np.zeros((f.shape[1],9))],rcond=1e-10)[0];e=np.abs(f@w-vs);out.append((k,j,m,e[:52,0].mean(),e[52:,0].mean(),e[:52,1:].mean(),e[52:,1:].mean()))
 for col in (3,4,5,6):print('pair rank',col,*sorted(out,key=lambda a:a[col])[:10],sep='\n')
