exec(open('/tmp/corrected_probe.py').read().split("print('diagnostic reference")[0])
_,xs=m._features(g,g); ds=th[2]+xs@th[3:]; Y=np.c_[y,r]; E=np.vstack([-np.expm1(-t[:,None]/tau),1/tau]); L=np.r_[np.log1p(t),1.]; n=len(h); ids=np.arange(n)

def rm(e):return np.array([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))])
def branch(v,j):return v[:,None,None]*E[None,:,j:j+1]
def shared(v,f):return v[:,None,None]*f[None,:,None]
def eval_(name, base, F):
 es=[]
 for k in ids:
  keep=ids!=k; w=np.linalg.lstsq(F[keep].reshape(-1,F.shape[-1]),(Y-base)[keep].ravel(),rcond=None)[0]; es.append(base[k]+F[k]@w-Y[k])
 w=np.linalg.lstsq(F.reshape(-1,F.shape[-1]),(Y-base).ravel(),rcond=None)[0]; es=np.array(es); print(name,'LOO',rm(es).round(5),'max abs',np.max(abs(es),axis=0)[[0,-1]].round(5),'weights',w.round(5))
 return w
for dn,dv in [('self',ds),('pooled',dr)]:
 Z=Y-dv[:,None]*L; A=np.linalg.lstsq(E,Z.T,rcond=None)[0].T; print(dn,'fixed-log free-amplitude error',rm(A@E.T-Z).round(5)); print(dn,'amplitudes divided by intrinsic weights minus hold', (A/th[:2]-h[:,None]).round(4).T)
 for mode in ('hold','last','all'):
  inputs={'hold':np.c_[h], 'last':np.c_[h,p[:,-1]], 'all':np.c_[h,p]}[mode]; F=np.concatenate([branch(inputs[:,q],j) for j in range(2) for q in range(inputs.shape[1])] + [shared(dv,L),shared(np.ones(n),L)],axis=-1); eval_(dn+'/'+mode,np.zeros_like(Y),F)
# Use the full chronological design to reveal branch history kernels without assuming exponential memory.
F=np.concatenate([branch(v,j) for j in range(2) for v in (h,*p.T)] + [shared(ds,L),shared(np.ones(n),L),shared(dr,L)],axis=-1);eval_('combined',np.zeros_like(Y),F)
