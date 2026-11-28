import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations,product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
ls=lambda X,y:np.linalg.lstsq(X,y,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def calibration(p):
 u,v=p['u'],p['v']; q=.18*np.tanh(1.8*u)-.32*np.sin(2.4*np.roll(u,2,axis=1)); Y=v-q; B=np.stack([ls(aff(u[:,j:j+1]),Y[:,j]) for j in range(4)],1)
 for _ in range(5):
  r=Y-B[0]-u*B[1]; m=np.max(abs(r),1)<.5; B=np.stack([ls(aff(u[m,j:j+1]),Y[m,j]) for j in range(4)],1)
 def f(x):return B[0]+B[1]*x+.18*np.tanh(1.8*x)-.32*np.sin(2.4*np.roll(x,2,axis=1))
 def inv(x):
  z=(x-B[0])/B[1]
  for _ in range(180):z=(x-B[0]-.18*np.tanh(1.8*z)+.32*np.sin(2.4*np.roll(z,2,axis=1)))/B[1]
  return z
 return f,inv
f,inv=calibration(s['panel_a'])
funs={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x),'cos':np.cos,'sq':lambda x:x*x,'cube':lambda x:x**3}
ops={}
for pk,vk in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[pk]['u']; v=s[pk][vk]; res=[]
 for nam,g in funs.items():
  for shape in (0,1,2):
   def feat(x,z):
    q=g(z[0]*x+z[1]);return aff(q) if shape==0 else (aff(x+z[2]*q) if shape==1 else np.c_[aff(x),q])
   def calc(z):
    A=feat(u,z); return (A@ls(A,v)-v).ravel()
   for freq,b,c in product([.7,1.5,2.5,3.7],[0.,.6],[-.4,.4] if shape==1 else [0.]):
    opt=least_squares(calc,[freq,b,c],max_nfev=110,bounds=([.08,-3.2,-6],[6,3.2,6]),ftol=1e-10); A=feat(u,opt.x); e=calc(opt.x).reshape(v.shape); h=np.sum(A*np.linalg.pinv(A).T,1); res.append((np.mean(abs(e)/np.maximum(1-h[:,None],1e-7)),np.mean(abs(e)),nam,shape,opt.x,ls(A,v)))
 res.sort(key=lambda a:a[0]); op=pk[-1]+vk; print(op,[(round(r[0],8),round(r[1],8),*r[2:5]) for r in res[:4]],flush=True)
 r=res[0]; g=funs[r[2]]; shape=r[3]; z=r[4]; B=r[5]
 def opfun(x,g=g,shape=shape,z=z,B=B):
  q=g(z[0]*x+z[1]); return (aff(q) if shape==0 else (aff(x+z[2]*q) if shape==1 else np.c_[aff(x),q]))@B
 ops[op]=opfun
 print('coef',B,flush=True)
x=s['train']['x']; V=np.c_[s['train']['y'],s['train']['traj']]; print('telemetry',s['telemetry_vec'],'summary',s['train']['summary'],'x range',x.min(),x.max(), 'inv range',inv(x).min(),inv(x).max(),'inv error',abs(f(inv(x))-x).max(),flush=True)
coords={'raw':x,'cal':inv(x),'fwd':f(x)}
for name in list(coords):
 for op,g in ops.items():coords[name+'/'+op]=g(coords[name])
for name in list(coords):
 if '/' in name:
  for op,g in ops.items():
   if op!=name.split('/')[-1]:coords[name+'/'+op]=g(coords[name])
def emit(coords):
 score=[]
 for name,u in coords.items():
  P=np.stack([u[:,i]*u[:,j] for i,j in combinations(range(4),2)],1); A=aff(u)
  for tag,X in [('lin',A),('quad',np.c_[A,u*u,P]),('pair',np.c_[A,P]),('signed',np.c_[A,np.sum(u[:,:2]*u[:,2:],1)])]:
   for l in [0,.03]:
    B=np.linalg.solve(X[:52].T@X[:52]+(l+1e-10)*np.eye(X.shape[1]),X[:52].T@V[:52]); e=abs(X@B-V); score.append((e[52:,0].mean(),e[52:,1:].mean(),name,tag,l,e[:52,0].mean(),e[:52,1:].mean()))
 print('Y',sorted(score)[:14]);print('T',sorted(score,key=lambda a:a[1])[:14]);
emit(coords)
