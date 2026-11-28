import pickle, numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; Y=np.c_[y,z]; n=len(x)
def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def affine(q): return np.c_[np.ones(len(q)),q]
def cross(q): return q[:,[0,0,1,2]]*q[:,[1,3,2,3]]
def basis1(q): return np.c_[np.ones(len(q)),cross(q)]
def basis2(q): return np.c_[affine(q),q*q]
p=s['panel_c']; u=p['u']; v=p['v2']; b1=fit(basis1(u),p['v1']); g=basis1(x)@b1
for name, fn, init in [('quadratic',lambda q,k:q+k[0]*q*q,[.45]),('cubic',lambda q,k:q+k[0]*q*q+k[1]*q**3,[.45,0]),('exp',lambda q,k:np.expm1(k[0]*q),[.8]),('rational',lambda q,k:(q+k[0]*q*q)/(1+k[1]*q*q),[.45,0]),('sin',lambda q,k:np.sin(k[0]*q)+k[1]*q*q,[.4,.2]),('even sine',lambda q,k:q+k[0]*(1-np.cos(k[1]*q)),[1.,1.]),('tanh+quad',lambda q,k:np.tanh(k[0]*q)+k[1]*q*q,[.4,.2])]:
 def res(k):
  M=affine(fn(u,k)); return (M@fit(M,v)-v).ravel()
 opt=least_squares(res,init,max_nfev=300); print('v2',name,opt.x.round(5),np.std(opt.fun).round(7))
b2=fit(basis2(u),v); h=basis2(x)@b2
p=s['panel_b']; f=affine(x)@fit(affine(p['u']),p['v'])
def pair(a,b): return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def cv(A):
 errors=[]
 for te in [np.arange(52,72),*np.array_split(np.random.default_rng(42).permutation(n),5)]:
  tr=np.setdiff1d(np.arange(n),te); r=abs(A[te]@fit(A[tr],Y[tr])-Y[te]);errors.append([r[:,0].mean(),r[:,1:].mean(),r[:,1:].mean(0).max()])
 print(A.shape[1],np.linalg.matrix_rank(A),np.std(A@fit(A,Y)-Y,axis=0)[[0,1,-1]].round(5),np.array([errors[0],np.max(errors[1:],axis=0)]).round(5).tolist())
P=np.c_[np.ones(n),f,g,h]; Q=np.c_[affine(x),pair(x,x)[:,[0,1,2,3,5,6,7,10,11,15]]]; C=pair(f,g); D=pair(f,h)
for name,A in [('P+diag both',np.c_[P,f*g,f*h]),('Q+diag both',np.c_[Q,f*g,f*h]),('Q+v1 pair',np.c_[Q,C]),('Q+v2 pair',np.c_[Q,D]),('Q+both',np.c_[Q,C,D]),('Q+v1 pair+v2 diag',np.c_[Q,C,f*h]),('Q+v2 pair+v1 diag',np.c_[Q,D,f*g]),('P+same f times both',np.c_[P,pair(f,g+h)]),('P+f products',np.c_[P,pair(f,f),C,D])]: print(name,end=' ');cv(A)
M=lambda a,b:-np.expm1(-t[:,None]*[a,b]); amp=fit(M(.41225926,1.79590825),z.T).T
# Are the two empirical modes supplied by distinct stages?
for name,A in [('P',P),('Q',Q),('Q+C',np.c_[Q,C]),('Q+D',np.c_[Q,D]),('Q+C+D',np.c_[Q,C,D]),('Q+diag',np.c_[Q,f*g,f*h])]:
 r=np.c_[y,amp]-A@fit(A,np.c_[y,amp]); print('latent',name,r.std(0).round(6))
