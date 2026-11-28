from graph06 import *
import sys
sys.path.insert(0,'/tmp/hegel_continual_archive')
from dev_m102_c02_batch_04_solver import _calibration
x=np.r_[s['train']['x'],s['test']['x']]; z=_calibration(s['panel_a'],x)
bp=s['panel_b'];cp=s['panel_c'];B=op(**bp,fn=soft,p=[.35,1.5],lim=([0,.1],[.8,8]));C=op(cp['u'],cp['v1'],power,[.9,1,1.25],([0,.1,.1],[8,8,8]));
for fn,p,lo,hi in [(lambda x,p:x+p[0]*x*x,[.45],[-2],[2]),(lambda x,p:x+p[0]*x*x/(1+p[1]*x*x),[.45,.1],[-2,0],[2,8]),(lambda x,p:x+p[0]*x*x+p[1]*x**3,[.45,0],[-2,-2],[2,2]),(lambda x,p:(x+p[0]*x*x)/(1+p[1]*x*x),[.45,.1],[-2,0],[2,8])]:
 D=op(cp['u'],cp['v2'],fn,p,(lo,hi))
Y=np.c_[s['train']['y'],s['train']['traj']];print('t',s['eval_times'],'Y corr',np.round(np.corrcoef(Y.T)[0],4),'y range',Y[:,0].min(),Y[:,0].max());
print('B',fit(af(soft(bp['u'],[.35,1.5])),bp['v'])); print('C',fit(af(power(cp['u'],[.912,.966,1.276])),cp['v1']));
b=B(z);c=C(b);d=D(b);q=C(z);r=D(z);a=soft(z,[.35,1.5]);
from scipy.spatial.distance import cdist
from scipy.linalg import solve
res=[]
def sc(n,H,typ='lin'):
 if typ=='lin':
  k=H[:52]; m=k.mean(0); sd=k.std(0)+1e-9; U=af((H-m)/sd); P=U[52:72]@fit(U[:52],Y[:52]); e=np.mean(abs(P-Y[52:]),axis=0);res.append((e[0],e[1:].mean(),n,typ));return
 K=H[:52];sd=K.std(0)+1e-6; ds=cdist(H[:72]/sd,H[:52]/sd,'sqeuclidean')/H.shape[1];
 for l in [.2,.4,.8,1.6,3.2,6.4]:
  E=np.exp(-ds/l); U=np.sqrt(ds)/l; F=(1+U+U*U/3)*np.exp(-U)
  for name,J in [('rbf',E),('mat',F)]:
   for reg in [1e-6,.0001,.003,.03]:
    m=Y[:52].mean(0);P=J[52:]@solve(J[:52]+reg*np.eye(52),Y[:52]-m,assume_a='pos')+m;e=np.mean(abs(P-Y[52:]),axis=0);res.append((e[0],e[1:].mean(),n,(name,l,reg)))
for n,h in dict(x=x,z=z,a=a,b=b,c=c,d=d,q=q,r=r,ab=np.c_[a,b],bc=np.c_[b,c],bd=np.c_[b,d],cd=np.c_[c,d],bcd=np.c_[b,c,d],bq=np.c_[b,q],br=np.c_[b,r],zc=np.c_[z,c],za=np.c_[z,a],abc=np.c_[a,b,c]).items():
 sc(n,h,'kernel');sc(n,h)
 for j in [0,1,2,3]:
  if h.shape[1]!=4:continue
  p=np.roll(h,j,axis=1);o=h*p;sc(n+'pair'+str(j),np.c_[h,o]);sc(n+'triple'+str(j),np.c_[h,o,o*np.roll(p,1,axis=1)])
  sc(n+'quot'+str(j),np.c_[h,h/(1+.22*p*p),h*p/(1+.22*p*p)])
for th in [0,.1,.22,.3,.35,.4,.45,.5,.6,.7]:
 for pp in [1,1.25,1.5,2,3]:
  h=soft(z,[th,pp]);sc(f'warp {th} {pp}',h,'kernel');sc(f'warp {th} {pp}',h);sc(f'warpquad {th} {pp}',terms(h,2)[0]);
print(*sorted(res,key=lambda p:p[0]+p[1])[:24],sep='\n');print('linear',*sorted([r for r in res if r[-1]=='lin'],key=lambda p:p[0]+p[1])[:12],sep='\n')
