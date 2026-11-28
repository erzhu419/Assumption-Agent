import passed_solver as m
import numpy as np,pickle
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train']; x=np.r_[t['x'],s['test']['x']];z0=m._calibrate(s['panel_a'],x);p=s['panel_a'];u,v=p['u'],p['v'];w=np.roll(u,2,axis=1);A=m._aff(u);e=np.linalg.norm(v-A@m._ls(A,v),axis=1);keep=e<max(3*np.median(e),.2)
def des(x,r,linear=False):
 w=np.roll(x,2,axis=-1);g=np.zeros(x.shape+(11 if linear else 10,)); i=np.arange(4);g[...,i,i]=1;g[...,i,i+4]=x;g[...,8]=x**3;g[...,9]=np.sin(r*w)
 if linear:g[...,10]=w
 return g
zs={'base':z0}
for lin in [False,True]:
 def coeff(r):return m._ls(des(u[keep],r,lin).reshape(-1,10+lin),v[keep].ravel())
 def rr(r):return (des(u[keep],r,lin)@coeff(r)-v[keep]).ravel()
 r=minimize_scalar(lambda r:np.std(rr(r)),bounds=(1.4,3.5),method='bounded').x;c=coeff(r); print('compact cal',lin,r,np.std(rr(r)),c.round(5));zs[str(lin)]=np.array([least_squares(lambda z:des(z,r,lin)@c-y,g).x for y,g in zip(x,z0)])
p=s['panel_c'];u=p['u'];B=s['panel_b']; ls=m._ls; aff=m._aff; cross=m._cross
rng=np.random.default_rng(18);folds=[np.arange(72)]+[rng.permutation(72) for _ in range(12)]
def errors(F):
 total=[]
 for ids in folds:
  a,b=ids[:52],ids[52:];Q=m._basis(s['eval_times'],t['traj'][a]);target=np.c_[t['y'][a],t['traj'][a]@Q];pred=m._readout(F[ids],target,len(a));e=abs(pred[:,1:]@Q.T-t['traj'][b]);total.append([np.mean(abs(pred[:,0]-t['y'][b])),e.mean(),e.mean(0).max()])
 total=np.array(total);return np.r_[total[0],total.mean(0),total.max(0)].round(5)
for zn,z in zs.items():
 c=m._branch(p,z);b=aff(z)@ls(aff(B['u']),B['v']);gs={}
 for typ,func,start in [('cos',np.cos,2.15),('gauss',lambda p:np.exp(-p*p),.88)]:
  for lev in [0,1,2]:
   def d(x,r):return np.c_[aff(x),func(r*x),*( [x*x] if lev>=1 else []),*( [x**3] if lev>=2 else [])]
   def res(r):
    f=d(u,r);return np.std(f@ls(f,p['v2'])-p['v2'])
   r=minimize_scalar(res,bounds=(start*.65,start*1.35),method='bounded').x
   if zn=='base':print('v2',typ,lev,r,res(r))
   gs[typ+str(lev)]=d(z,r)@ls(d(u,r),p['v2'])
 gs['raw_g']=np.exp(-.877**2*z*z);gs['raw_c']=np.cos(2.15*z)
 for tag,g in gs.items():
  for mode in [0,1,2]:
   if mode==0:f=np.c_[c,g,cross(c,g),cross(z,c),cross(z,g),z]
   if mode==1:f=np.c_[c,g,cross(c,g),cross(z,c),cross(z,g),z,cross(z,z)]
   if mode==2:f=np.c_[c,g,cross(c,g),z,cross(z,z),np.sum(c*g,axis=1)]
   print(zn,tag,mode,errors(f))
