import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=s['train']['traj']; Y=s['train']['y']; x=s['train']['x']; t=s['eval_times']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]; ic=lambda a:np.column_stack((np.ones(len(a)),a));
def cald(u,mode):
 if mode==0:return [ic(u)]*4
 if mode==1:return [ic(np.c_[u,u**2,u**3])]*4
 if mode==2:return [ic(np.c_[u[:,i],u[:,(i+1)%4],u[:,i]*u[:,(i+1)%4],u[:,i]**3]) for i in range(4)]
 if mode==3:return [ic(np.c_[u,u**2,u**3,*[u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]])]*4
 if mode==4:return [ic(np.c_[u,u**3])]*4

def calfit(mode):
 u,v=s['panel_a'].values(); d=cald(u,mode); r=np.linalg.norm(v-u,axis=1); keep=r<np.median(r)*3
 for _ in range(7):
  c=[ls(a[keep],b[keep]) for a,b in zip(d,v.T)]; e=np.linalg.norm(np.column_stack([a@b for a,b in zip(d,c)])-v,axis=1); keep=e<max(.0003,3*np.median(e))
 return lambda z:np.column_stack([a@b for a,b in zip(cald(z,mode),c)]), np.sort(e)
fs={'asinh':np.arcsinh,'erf':erf,'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'soft':lambda z:z/np.sqrt(1+z*z),'rat':lambda z:z/(1+z*z),'cos':np.cos,'square':lambda z:z*z,'cubic':lambda z:z**3,'exp':lambda z:np.exp(-z*z)}
def opfit(u,v,fn=None,combined=False,free=False):
 if fn is None:
  return lambda z:ic(z)@ls(ic(u),v),np.sqrt(np.mean((ic(u)@ls(ic(u),v)-v)**2)),[]
 def d(u,q):return ic(np.c_[u,fn(u*np.exp(q))] if combined else fn(u*np.exp(q)))
 def e(q):
  a=d(u,q);return (a@ls(a,v)-v).ravel()
 fit=min([least_squares(e,np.full(4 if free else 1,p),bounds=(-2.5,1.6),max_nfev=90) for p in [-.3,.5,1.15]],key=lambda f:f.cost); q=fit.x; c=ls(d(u,q),v)
 return lambda z:d(z,q)@c,np.sqrt(np.mean(e(q)**2)),np.exp(q)
if __name__=='__main__':
 print('sv',np.linalg.svd(T,compute_uv=False),'times',t,'telemetry',s['telemetry_vec']);
 for mode in range(5):print('cal',mode,np.round(calfit(mode)[1][::3],6))
 for panel,out in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u,v=s[panel]['u'],s[panel][out]; print(panel,out,'affine',opfit(u,v,None)[1]); rows=[]
  for name,fn in fs.items():
   for combined in [False,True]:
    for free in [False,True]:
     _,e,q=opfit(u,v,fn,combined,free);rows.append((e,name,combined,free,np.round(q,4)))
  for row in sorted(rows,key=lambda r:r[0])[:14]:print(row)
