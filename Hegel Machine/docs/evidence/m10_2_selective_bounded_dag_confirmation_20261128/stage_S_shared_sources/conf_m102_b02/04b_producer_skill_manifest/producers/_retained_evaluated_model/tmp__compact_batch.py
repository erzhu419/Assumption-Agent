import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
p='/holdout_task/'; spec=importlib.util.spec_from_file_location('candidate',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; ix=np.arange(len(x)); a=m._affine(s['panel_a'],True); b=m._affine(s['panel_b']); z=np.linalg.solve(a[1:].T,(x-a[0]).T).T@b[1:]+b[0]; Y=np.c_[t['y'],t['traj']]; i,j=np.triu_indices(4); k,l=np.triu_indices(4,1)
forms={'quadratic':lambda z,h:np.c_[h,h[:,i]*h[:,j]],'hybrid':lambda z,h:np.c_[z,h,h[:,i]*h[:,j],z*h],'factor':lambda z,h:np.c_[z,h,h*h,z*h,h[:,k]*h[:,l]],'shared':lambda z,h:np.c_[h,z,h*h,z*h,h[:,k]*h[:,l],z[:,k]*z[:,l]]}
blocks=[ix[52:],ix[:20],ix[20:40]]; edges=[q for j in range(4) for q in [np.argsort(z[:,j])[:10],np.argsort(z[:,j])[-10:]]]; rng=np.random.default_rng(729); random=[rng.choice(len(ix),20,replace=False) for _ in range(32)]
for name,f in forms.items():
 for bias in [False,True]:
  vals=[]; pars=[]
  for te in blocks+edges+random:
   tr=np.setdiff1d(ix,te); modes=np.linalg.svd(Y[tr,1:],full_matrices=False)[2][:2]; G=np.c_[Y[tr,0],Y[tr,1:]@modes.T]; scale=G.std(0); G=G/scale
   def design(zz,p):
    h=np.arcsinh(np.exp(p[0])*(zz+p[1])); F=f(zz,h); return np.c_[np.ones(len(F)),F]
   def fitted(p):
    F=design(z[tr],p); r=np.eye(F.shape[1])*.002; r[0,0]=0; H=np.r_[F,r]; T=np.r_[G,np.zeros((len(r),3))]; c=np.linalg.lstsq(H,T,rcond=None)[0]; return c,(H@c-T).ravel()
   def fun(q):return fitted(np.r_[q,0.][:2])[1]
   opt=least_squares(fun,[np.log(3.5),0.] if bias else [np.log(3.5)],bounds=([np.log(.8),-.35],[np.log(12),.35]) if bias else ([np.log(.8)],[np.log(12)]),max_nfev=50); p2=np.r_[opt.x,0.][:2]; pars.append([np.exp(p2[0]),p2[1]])
   pred=(design(z[te],p2)@fitted(p2)[0])*scale; e=np.abs(np.c_[pred[:,0],pred[:,1:]@modes]-Y[te]); vals.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()])
  E=np.array(vals); print(name,bias,'block',np.round(E[:3],5).tolist(),'edge worst',E[3:11].max(0),'random worst',E[11:].max(0),'parameters',np.round(pars[:3],4).tolist())
