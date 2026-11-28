import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import _coords,_panel_b,_poly,_ls
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; z=_coords(s,x); h=_panel_b(s,z)
T=Y[:,1:]; _,_,vh=np.linalg.svd(T[:52],full_matrices=False); target=np.c_[Y[:,0], T@vh[:3].T]; ix=np.arange(52); te=np.arange(52,72)

def scores(P):return np.mean(abs(P[:,0]-Y[te,0])),np.mean(abs(P[:,1:]@vh[:3]-T[te]))

def ridge(F, a, b, r):
 norm=np.maximum(np.linalg.norm(F[a],axis=0),1e-8); H=F/norm; return H[b]@np.linalg.solve(H[a].T@H[a]+r*np.diag(np.r_[0,np.ones(H.shape[1]-1)]),H[a].T@target[a])
rows=[]
for tag,q in [('h',h),('z',z),('hz',np.c_[h,z])]:
 for d in (2,3,4,5):
  if d>3 and q.shape[1]>4:continue
  A=_poly(q,d); N=A.shape[1]
  for r in [1e-8,1e-6,1e-4,.001,.01,.05,.2]:
   p=ridge(A,ix,te,r); rows.append((*scores(p),tag,d,r))
print('poly y',sorted(rows)[:12]);print('poly t',sorted(rows,key=lambda r:r[1])[:12])
forms={
 'rquad':lambda q,w,A: A/(1+(q@w[:4]+w[4])[:,None]**2),
 'rsplit':lambda q,w,A:np.c_[A[:,:5],A[:,5:]/(1+(q@w[:4]+w[4])[:,None]**2)],
 'rlinear':lambda q,w,A:np.c_[A,A[:,:5]/(1+(q@w[:4]+w[4])[:,None]**2)],
 'tanh':lambda q,w,A:np.c_[A,np.tanh(q@w[:4]+w[4])],
 'sin':lambda q,w,A:np.c_[A,np.sin(q@w[:4]+w[4])],
 'sincube':lambda q,w,A:np.c_[A,np.sin(q@w[:4]+w[4]),(q@w[:4]+w[4])**3],
 'cube':lambda q,w,A:np.c_[A,(q@w[:4])**3],
 'prod':lambda q,w,A:np.c_[A,(q@w[:4])[:,None]*A[:,5:]],
 'isotropic':lambda q,w,A:np.c_[A/(1+(q*q)@np.exp(np.clip(w[:4],-9,3)))[:,None]],
 'denquad':lambda q,w,A:A/(1+np.sum((q*w[:4])**2,axis=1)[:,None]),
}
for name,f in forms.items():
 out=[]
 for tag,q in [('h',h),('z',z)]:
  A=_poly(q,2); pp=[]; pars=[]
  for k in range(4):
   yy=target[:,k]; sc=max(.005,np.std(yy[ix])); rr=np.random.default_rng(44); starts=np.r_[np.eye(5),rr.normal(size=(9,5))]; cand=[]
   for start in starts:
    def err(w):
     F=f(q[ix],w,A[ix]); return np.r_[(F@_ls(F,yy[ix])-yy[ix])/sc,.002*w]
    r=least_squares(err,start,bounds=(-9,9),max_nfev=100); cand.append((np.linalg.norm(r.fun),r.x))
   e,w=min(cand,key=lambda v:v[0]); F=f(q,w,A); pp.append(F[te]@_ls(F[ix],yy[ix])); pars.append((round(e*sc,4),np.round(w,3).tolist()))
  out.append((tag,scores(np.array(pp).T),pars[:2]))
 print(name,out,flush=True)
