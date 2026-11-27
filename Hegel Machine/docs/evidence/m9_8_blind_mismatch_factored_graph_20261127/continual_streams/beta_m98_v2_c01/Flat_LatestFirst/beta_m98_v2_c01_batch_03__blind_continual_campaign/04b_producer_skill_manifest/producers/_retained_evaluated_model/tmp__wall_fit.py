import pickle, importlib.util, numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('base','/tmp/hegel_continual_archive/beta_m98_v2_c01_batch_02_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; s=p['study']; ds=p['campaign_diagnostics']; Y=np.stack([m._calibrate(dict(s,y_obs_raw=d['y_pulse_raw'])) for d in ds],axis=1); W=np.stack([d['w_pulse'] for d in ds],axis=1)
x,y=s['nx'],s['ny']; I=eye(x*y,format='csc'); ij=s['sensor_ij']; rows=ij[:,0]+x*ij[:,1]; rhs=np.stack([v.ravel(order='F') for v in np.tensordot(W.T,s['source_modes'],axes=1)],axis=1)
ks=[s['kappa_'+a] for a in 'xy']; vs=[s['vel_'+a] for a in 'xy']; hs=[s['Lx']/(x-1),s['Ly']/(y-1)]; rs=[s['robin_'+a] for a in 'xy']; V=diags(s['V_reaction_2d'].ravel(order='F'),format='csc')
def pred(c,l,v=(0,0,0,0), face=False):
 A=[]
 for j,(k,vv,h,r) in enumerate(zip(ks,vs,hs,rs)):
  a=m._axis(k,vv,h,0).tolil()
  for t,i in enumerate((0,len(k)-1)):
   u=i+(1 if t==0 else -1); kk=.5*(k[i]+k[u]) if face else k[i]; e=c[2*j+t]*kk/h**2+v[2*j+t]*abs(vv[i])/h
   a[i,u]=-e; a[i,i]=e+l[2*j+t]*r/h
  A.append(a.tocsc())
 L=kron(eye(y),A[0],format='csc')+kron(A[1],eye(x),format='csc')+V
 return np.vstack([splu(L+mu*I).solve(rhs)[rows] for mu in s['mu_shifts']])
def err(a): return np.sqrt(np.mean(np.square(a-Y),axis=0))
for label,c,l,v,face in [('reflect',1,0,(0,)*4,False),('twice',2,0,(0,)*4,False),('half',.5,0,(0,)*4,False),('face',1,0,(0,)*4,True),('twice-face',2,0,(0,)*4,True),('reflect-velocity',1,0,(1,)*4,False),('open',1,1,(0,)*4,False)]:
 a=pred([c]*4,[l]*4,v,face);print(label,err(a),'shift RMS',np.sqrt(np.mean((a-Y).reshape(3,32,2)**2,axis=1)).round(6).tolist())
models=[('shared-wall', [1.,0.],lambda b:([b[0]]*4,[b[1]]*4)),('axis-wall',[1.,1.,0.,0.],lambda b:(np.repeat(b[:2],2),np.repeat(b[2:],2))),('four-wall',[1.]*4+[0.]*4,lambda b:(b[:4],b[4:]))]
for label,start,par in models:
 for tr in ([0],[1],[0,1]):
  fit=least_squares(lambda b:(pred(*par(b))-Y)[:,tr].ravel(),start,bounds=([.05]*(len(start)//2)+[-.2]*(len(start)//2),[8.]*(len(start)//2)+[2.]*(len(start)//2)),max_nfev=100)
  print(label,tr,np.round(fit.x,6),err(pred(*par(fit.x))))
q=np.load('/dev/null') if False else None
