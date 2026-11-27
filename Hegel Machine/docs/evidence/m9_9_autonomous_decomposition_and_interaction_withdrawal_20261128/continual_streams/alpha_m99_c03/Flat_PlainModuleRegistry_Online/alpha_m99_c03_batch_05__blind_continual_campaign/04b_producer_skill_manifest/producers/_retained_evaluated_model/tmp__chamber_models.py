from chamber_probe import *
from scipy.optimize import least_squares
A,G=m._design(tau,inp_d,dt,t)
past,hold,F=inp_d
z=hold[:,None]-G[:,:2]*tau
r0=F@th[2:]
Y=np.c_[yd-A@th,rd-G@th]; ts=np.r_[t,0.]
B=np.vstack((np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1.]))
u,v,w=B.T; bend=v-w/tau[1]; one=F[:,0]; a,b=z.T
P=np.linalg.lstsq(B,Y.T,rcond=None)[0].T
for label,target,M in [('fast',P[:,0],np.c_[one,a,b,hold,r0]),('slow',P[:,1],np.c_[one,a,b,hold,r0]),('drift',P[:,2],np.c_[one,a,b,hold,r0]),('rate after full withdrawal',Y[:,-1]-a*th[0]/tau[0],np.c_[one,a,b,hold,r0])]:
 c=np.linalg.lstsq(M,target,rcond=None)[0]; print(label,'[1,z0,z1,hold,r]',c.round(5),score(target-M@c))
def fit(M, mask):
 return np.linalg.lstsq(M[mask].reshape(-1,M.shape[-1]),Y[mask].ravel(),rcond=None)[0]
def make(terms):
 return np.stack([f[:,None]*g for f,g in terms],axis=-1)
models={
 'withdrawal':[(a,u),(r0,bend)],
 'withdrawal + constant':[(a,u),(one,u),(r0,bend),(one,bend)],
 'withdrawal + memory mixing':[(a,u),(one,u),(b,u),(r0,bend),(one,bend),(b,bend)],
 'withdrawal + linear drift':[(a,u),(one,u),(r0,bend),(one,bend),(r0,w),(one,w)],
 'withdrawal + feature drift':[(a,u),(one,u),*[(f,bend) for f in F.T],*[(f,w) for f in F.T]],
 'memory + feature drift':[(a,u),(one,u),(b,u),(hold,u),*[(f,bend) for f in F.T],*[(f,w) for f in F.T]],
 'separate amplitudes':[(a,u),(one,u),(b,u),(hold,u),*[(f,bend) for f in np.c_[F,a,b,hold].T],*[(f,w) for f in np.c_[F,a,b,hold].T]],
 'fast rate + slow bend':[(a,u),(one,u),(b,u),*[(f,bend) for f in F.T]],
}
for label,terms in models.items():
 M=make(terms); c=fit(M,np.ones(len(a),bool)); e=np.empty_like(Y)
 for g in np.unique(d['lab_id']):
  mask=d['lab_id']==g; e[mask]=Y[mask]-M[mask]@fit(M,~mask)
 E=Y-M@c
 print(label,'coefs',c.round(4),'in sample',score(E[:,:-1]),score(E[:,-1]),'LOLO',score(e[:,:-1]),score(e[:,-1]),'worst lab RMSE',np.max([np.sqrt(np.mean(e[d['lab_id']==g]**2,axis=0)) for g in np.unique(d['lab_id'])],axis=0).round(5))
