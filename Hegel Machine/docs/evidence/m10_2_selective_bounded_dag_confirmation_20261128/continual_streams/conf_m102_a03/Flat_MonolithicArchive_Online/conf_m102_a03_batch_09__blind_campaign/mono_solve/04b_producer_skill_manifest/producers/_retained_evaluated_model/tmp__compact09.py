import runpy,numpy as np
q=runpy.run_path('/tmp/verify_batch09.py'); m,s,d,z=q['m'],q['s'],q['d'],q['z']; b=s['panel_b']; B=m._fit(m._affine(b['u']),b['v']); w=m._affine(z)@B
F=m._affine(w); G=np.c_[F,w[:,0]*w[:,2]+w[:,1]*w[:,3]]; H=m._quadratic(w); A=d['traj']; y=d['y']; print('B residual RMS',np.mean((m._affine(b['u'])@B-b['v'])**2)**.5); print('times',s['eval_times']); print('trajectory slopes by time\n',m._fit(F,A).T)
def compare(i,j):
 out=[]
 for f in [H,G]:out.append(np.mean(abs(f[j]@m._fit(f[i],y[i])-y[j])))
 for r in [None,2]:
  coef=m._fit(F[i],A[i]); pred=F[j]@coef
  if r:
   _,_,v=np.linalg.svd(F[i]@coef,full_matrices=False); pred=(pred@v[:r].T)@v[:r]
  e=abs(pred-A[j]); out.extend([e.mean(),e.mean(0).max()])
 return out
idx=np.arange(len(w)); print('holdout (quadratic y, symmetric y, affine trajectory mean/worst time, rank2 trajectory mean/worst time)',compare(idx[:52],idx[52:])); vals=[]
for v in [*z.T,*w.T,np.linalg.norm(z,axis=1)]:
 j=np.argsort(v); vals.extend([compare(j[20:],j[:20]),compare(j[:-20],j[-20:])])
print('edge worst',np.max(vals,axis=0))
