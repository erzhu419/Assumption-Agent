exec(open('/tmp/search04.py').read().split('f=np.c_')[0])
from scipy.optimize import least_squares
from itertools import product
u=s['panel_a']['u']; v=s['panel_a']['v']; g=poly(u,1); B=np.linalg.lstsq(g,v,rcond=None)[0]; good=np.argsort(np.linalg.norm(v-g@B,axis=1))[:-3]; F=poly(u,3); A=np.linalg.lstsq(F[good],v[good],rcond=None)[0]; terms=[()]+[j for d in (1,2,3) for j in cwr(range(4),d)]; print('panel A coefficients',[(j,np.round(a,4).tolist()) for j,a in zip(terms,A) if np.max(np.abs(a))>.002]); A[np.abs(A)<.002]=0
for j in range(4):
 mask=A[:,j]!=0; A[mask,j]=np.linalg.lstsq(F[good][:,mask],v[good,j],rcond=None)[0]
B=A[1:5]; base=(x-A[0])@np.linalg.inv(B); w=[]; errs=[]
for p,b in zip(x,base):
 fun=lambda t:poly(t[None,:],3)[0]@A-p
 sol=least_squares(fun,b,gtol=1e-11); w.append(sol.x);errs.append(np.linalg.norm(sol.fun))
w=np.array(w); print('inverse residual',np.max(errs),'coordinate ranges',w.min(0),w.max(0),'sd',w.std(0),'corr',np.corrcoef(w.T,q.T)[:4,4:]); print('time sv',np.linalg.svd(z,compute_uv=False));
folds.extend([np.argsort(np.linalg.norm(x,axis=1))[-12:]]+list(np.array_split(np.random.default_rng(118).permutation(n),4)))
regs=(.000001,.00003,.0003,.0015,.007,.03); ex=[]
for name,hq in [('inverse',w),('frozen',q),('raw',x)]:
 for h,typ in product((0,.22,.55,.85,1.25,1.8),('sqrt','den','tanh','sin')):
  if h==0 and typ!='sqrt':continue
  f={'sqrt':lambda:hq/np.sqrt(1+h*hq*hq),'den':lambda:hq/(1+h*hq*hq),'tanh':lambda:np.tanh(np.sqrt(h)*hq)/np.sqrt(h),'sin':lambda:np.sin(np.sqrt(h)*hq)/np.sqrt(h)}[typ]()
  for d in range(1,7):
   g=poly(f,d); js=[()]+[j for D in range(1,d+1) for j in cwr(range(4),D)];
   for mixed in (.25,.5,1):
    weights=np.array([mixed**max(0,len(set(j))-1)*.5**(.5*max(0,len(j)-2)) for j in js]); k=(g*weights)@(g*weights).T; score=evaluate(k,regs); ex.append((name,h,typ,d,mixed,*score))
for i in (-2,-1):
 print('BEST',i); print(*sorted(ex,key=lambda r:r[i])[:10],sep='\n')
