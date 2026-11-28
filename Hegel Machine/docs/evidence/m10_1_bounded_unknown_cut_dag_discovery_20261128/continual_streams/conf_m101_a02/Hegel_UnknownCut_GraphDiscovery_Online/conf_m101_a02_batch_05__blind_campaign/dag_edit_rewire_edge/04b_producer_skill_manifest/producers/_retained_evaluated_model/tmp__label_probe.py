exec(open('/tmp/sink_probe.py').read().split('for label,A in')[0])
y=Y[:,0]
B=poly(z,1)
def cv(D):
 return np.array([np.mean(abs(D[j]@np.linalg.lstsq(D[i],y[i],rcond=1e-9)[0]-y[j])) for i,j in folds])
F={}
for key,A in [('x',x),('z',z)]:
 for i,u in enumerate(A.T):
  for lab,v in [('sq',u*u),('cub',u**3),('sin',np.sin(2*u)),('cos',np.cos(2*u)),('tanh',np.tanh(2*u)),('sat',u/(1+u*u)),('abs',abs(u))]:F[f'{lab}({key}{i})']=v
for i,u in enumerate(x.T):
 for j,v in enumerate(z.T):
  for lab,f in [('prod',u*v),('sinprod',np.sin(2*u*v)),('tanhprod',np.tanh(2*u*v)),('u*v2',u*v*v),('u2*v',u*u*v),('u*sinv',u*np.sin(2*v)),('v*sinu',v*np.sin(2*u)),('u*cosv',u*np.cos(2*v)),('v*cosu',v*np.cos(2*u))]:F[f'{lab}(x{i},z{j})']=f
for key,A in [('x',x),('z',z)]:
 for i,j in it.combinations(range(4),2):
  u,v=A[:,i],A[:,j]
  for lab,f in [('prod',u*v),('u*v2',u*v*v),('u2*v',u*u*v),('sin+',np.sin(2*(u+v))),('sin-',np.sin(2*(u-v))),('cos+',np.cos(2*(u+v))),('cos-',np.cos(2*(u-v)))]:F[f'{lab}({key}{i},{key}{j})']=f
M=np.column_stack(list(F.values())); names=list(F)
for base in [B,np.c_[B,x],poly(z,2)]:
 D=base.copy(); print('BASE',D.shape,np.round(cv(D),4));
 for step in range(6):
  # Select only on the official training prefix by partial regression.
  i,j=folds[0]; Q=np.linalg.svd(D[i],full_matrices=False)[0]; R=M[i]-Q@(Q.T@M[i]); score=abs(R.T@y[i])/np.maximum(np.linalg.norm(R,axis=0),1e-12); order=np.argsort(score)[::-1]; k=order[0]; D=np.c_[D,M[:,k]]; print(step,[(names[k],round(score[k],4)) for k in order[:3]],np.round(cv(D),4))
print('single candidates'); scored=[(cv(np.c_[B,v]),k) for k,v in F.items()]; print([(k,np.round(e,4)) for e,k in sorted(scored,key=lambda q:max(q[0]))[:12]])
# Time-shape diagnostic: a common single time curve versus a low-rank two-curve trajectory.
u,sv,v=np.linalg.svd(Y[:,1:],full_matrices=False); print('time rank-1 reconstruction MAE',np.mean(abs(Y[:,1:]-np.outer(u[:,0]*sv[0],v[0])),axis=0)); print('time shape (unit terminal)',v[0]/v[0,-1])
