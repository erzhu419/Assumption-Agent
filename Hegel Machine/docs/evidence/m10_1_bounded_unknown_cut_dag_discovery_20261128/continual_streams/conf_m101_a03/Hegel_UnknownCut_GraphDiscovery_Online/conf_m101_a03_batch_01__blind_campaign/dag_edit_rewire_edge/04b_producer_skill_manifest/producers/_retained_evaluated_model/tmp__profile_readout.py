exec(open('/tmp/readout_probe.py').read().split("print('ranges'")[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
rng=np.random.default_rng(12)
base=affine(z); quad=poly(z,2)
terms={'one':np.ones(len(x)),**{f'x{i}':x[:,i] for i in range(4)},**{f'z{i}':z[:,i] for i in range(4)}}
fs={'tanh':np.tanh,'sin':np.sin,'rat':lambda u:u/(1+u*u),'soft':lambda u:u/np.sqrt(1+u*u)}
xa=affine(x)
seeds=[np.r_[0,v] for v in np.r_[np.eye(4),[[0,1,0,-1],[1,0,0,0],[0,0,1,-1],[0,1,0,1],[1,1,1,1]],rng.normal(size=(8,4))]]
seeds+= [np.linalg.lstsq(xa,d,rcond=None)[0] for d in [z[:,2]-z[:,3],z[:,0]+z[:,1],z[:,0]-z[:,1],z[:,2]+z[:,3],z[:,1]]]
ix=np.arange(52)
def fit(B,spec,pp,ix,maxn=140):
 def make(p):
  F=[B]
  for j,(op,am) in enumerate(spec):
   u=xa@p[5*j:5*j+5]; F.append((fs[op](u)*terms[am])[:,None])
  return np.column_stack(F)
 def res(p):
  F=make(p)[ix]; return y[ix]-F@np.linalg.lstsq(F,y[ix],rcond=None)[0]
 opt=least_squares(res,np.clip(pp,-5,5),bounds=(-6,6),max_nfev=maxn,ftol=2e-8)
 F=make(opt.x); c=np.linalg.lstsq(F[ix],y[ix],rcond=None)[0]; return np.linalg.norm(opt.fun),opt.x,F,c

def report(label,res):
 err,p,F,c=res; print(label,'rmse/official',round(err/np.sqrt(len(ix)),5),round(np.mean(np.abs(y[52:]-F[52:]@c)),5),'p',np.round(p,3),'c',np.round(c,3),'zparams',np.round([np.linalg.lstsq(base,xa@v,rcond=None)[0] for v in p.reshape(-1,5)],3),flush=True)
rank=[]
for a in terms:
 for op in fs:
  res=min((fit(quad,[(op,a)],s,ix,maxn=90) for s in seeds),key=lambda r:r[0]); rank.append((res[0],[(op,a)],res))
rank.sort(key=lambda r:r[0]);
for _,sp,res in rank[:12]: report(sp,res)
for _,sp,r0 in rank[:3]:
 rank2=[]
 for a in terms:
  for op in ('tanh','sin','rat'):
   sp2=sp+[(op,a)]; res=min((fit(quad,sp2,np.r_[r0[1],s],ix,maxn=110) for s in seeds[:9]),key=lambda r:r[0]);rank2.append((res[0],sp2,res))
 for _,sp2,res in sorted(rank2,key=lambda r:r[0])[:3]: report(sp2,res)
