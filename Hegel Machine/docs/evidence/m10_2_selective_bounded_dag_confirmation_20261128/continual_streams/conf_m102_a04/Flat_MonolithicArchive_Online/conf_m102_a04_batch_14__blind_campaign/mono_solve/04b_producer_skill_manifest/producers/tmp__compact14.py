exec(open('/tmp/stress14.py').read().split('tr=np.arange(52)')[0])
z=m._calibration(s['panel_a'], a['x']); idxs=[np.arange(52,72),np.argsort(abs(yy))[-20:],*[np.argsort(z[:,i])[-20:] for i in range(4)],*[np.argsort(z[:,i])[:20] for i in range(4)],*[np.random.default_rng(i).permutation(72)[:20] for i in range(16)]]
D=np.c_[simple,full[:,10]]
def basis(r):return -np.expm1(-t[:,None]*r)
for mode in range(4):
 metrics=[]
 for va in idxs:
  tr=np.setdiff1d(np.arange(72),va)
  if mode==0: c=m._ls(D[tr],tt[tr]); st=D[va]@c
  else:
   F=D if mode==1 else full
   c=m._ls(F[tr],tt[tr]); T=(F[tr]@c).T if mode<3 else tt[tr].T
   r=least_squares(lambda r:(basis(r)@m._ls(basis(r),T)-T).ravel(),[.4,1.8],bounds=([.05,.9],[.85,4.]),gtol=1e-11).x
   st=(F[va]@c)@np.linalg.pinv(basis(r)).T@basis(r).T
  e=abs(st-tt[va]); metrics.append([e.mean(),e.mean(0).max(),e.max()])
 print(mode,'fixed',metrics[0],'worst',np.max(metrics,0))
print('compact coeffs',np.round(m._ls(D[:52],tt[:52]),4));print('times',t)
