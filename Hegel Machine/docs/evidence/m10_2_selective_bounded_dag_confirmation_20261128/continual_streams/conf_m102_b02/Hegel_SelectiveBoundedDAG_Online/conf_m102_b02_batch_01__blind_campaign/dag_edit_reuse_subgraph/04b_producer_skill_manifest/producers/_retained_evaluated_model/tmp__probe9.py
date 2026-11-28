import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/compare_maps.py').read())
from scipy.optimize import minimize_scalar
u=base['deA']; y=np.column_stack([train['y'],train['traj']@np.linalg.svd(train['traj'],full_matrices=False)[2][:2].T]); mats={}; basis={}
for name,P,key in [('B',b,'v'),('C1',c,'v1'),('C2',c,'v2')]:
 U,V=P['u'],P[key]
 for typ,f in [('rat',lambda t,k:t/(1+k*t*t)),('tanh',lambda t,k:np.tanh(k*t)),('sin',lambda t,k:np.sin(k*t))]:
  def fit(k):
   F=np.c_[np.ones(len(U)),U,f(U,k)]; C=np.linalg.lstsq(F,V,rcond=None)[0];return np.mean((F@C-V)**2),C
  ks=np.linspace(.1,6,58);i=np.argmin([fit(k)[0] for k in ks]);k=minimize_scalar(lambda k:fit(k)[0],bounds=(ks[max(0,i-1)],ks[min(i+1,len(ks)-1)]),method='bounded').x
  rm,C=fit(k); l,sv,r=np.linalg.svd(C[1:5]); M=l@r; print(name,typ,k,rm**.5,sv)
  mats[name]=M; basis[name+typ]=(k,C,f)
results=[]
for name,M in {'I':np.eye(4),**mats,**{k+'T':m.T for k,m in mats.items()}}.items():
 v=u@M
 for a1 in [.45,.8,1.2,1.8,2.5,3.5,5,7]:
  for a2 in [-1.1,-.85,-.6,-.35,.0,.5,1.]:
   for desc,q in [('pre-tanh',(u+a2*np.tanh(a1*u))@M),('post-tanh',v+a2*np.tanh(a1*v)),('pre-sin',(u+a2*np.sin(a1*u))@M),('post-sin',v+a2*np.sin(a1*v)),('pre-rat',(u+a2*u/(1+a1*u*u))@M),('post-rat',v+a2*v/(1+a1*v*v))]:
    for sc in [.15,.35,.7,1.2]:
     e=score(q*sc);results.append((max(e),e.tolist(),name,desc,a1,a2,sc))
print('sweep',*sorted(results)[:12],sep='\n')
# Identify unusually predictive univariate/pairwise terms on held-out training rows.
tr=np.arange(52);te=np.arange(52,72)
for name,M in {'I':np.eye(4),**mats,**{k+'T':m.T for k,m in mats.items()}}.items():
 v=u@M; F=np.c_[np.ones(len(u)),v]; R=y-F@np.linalg.lstsq(F[tr],y[tr],rcond=None)[0]; D=[]; names=[]
 for i in range(4):
  for j in range(i,4):
   for nm,t in [('prod',v[:,i]*v[:,j]),('sum',v[:,i]+v[:,j]),('diff',v[:,i]-v[:,j])]:
    for nm2,t2 in [('raw',t),*[(f'{fn}{k}',f(k*t)) for fn,f in [('sin',np.sin),('cos',np.cos),('asinh',np.arcsinh),('tanh',np.tanh)] for k in [1,2,3.5,5,7]]]:
     S=t2-F@np.linalg.lstsq(F[tr],t2[tr],rcond=None)[0]; D.append(S); names.append((i,j,nm,nm2))
 D=np.array(D).T; C=np.abs(D[tr].T@R[tr])/np.maximum(np.linalg.norm(D[tr],axis=0)[:,None],1e-10)/np.linalg.norm(R[tr],axis=0)
 print('corr',name, [[(names[j],round(C[j,i],3)) for j in np.argsort(C[:,i])[-3:][::-1]] for i in range(3)])
