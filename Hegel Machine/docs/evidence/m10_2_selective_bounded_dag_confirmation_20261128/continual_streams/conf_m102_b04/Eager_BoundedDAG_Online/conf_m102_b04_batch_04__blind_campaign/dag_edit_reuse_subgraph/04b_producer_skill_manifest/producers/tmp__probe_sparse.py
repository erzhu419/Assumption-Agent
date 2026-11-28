exec(open('/tmp/profile_heads.py').read().split('res=[]')[0])
from scipy.optimize import least_squares
T=z@np.linalg.svd(z,full_matrices=False)[2][:2].T
for j in range(2):
 print('scalar/state corr',j,np.corrcoef(y,T[:,j])[0,1])
R=np.c_[y,T/T.std(0)*y.std()]
for name,k in [('sin',.85),('soft',.4),('rat',.18)]:
 h=G(xc,name,k); C=np.linalg.lstsq(h[:n],R,rcond=None)[0]; print(name,[(t,np.round(w,3)) for t,w in zip([()]+terms,C) if np.max(abs(w))>.07])
from scipy.linalg import lstsq
res=[]
for name in links:
 for shrink in [0,.008,.018,.035,.06]:
  for separate in [False,True]:
   out=[]; pars=[]
   for i,j in ss:
    def estimate(k):
     h=G(xc,name,k); R0=R[i,:1] if separate else R[i]; w=lstsq(h[i],R0)[0]; keep=np.ones(len(w),bool)
     if shrink:
      for _ in range(6):
       keep &= np.linalg.norm(w,axis=1)>shrink; keep[:5]=True; w[~keep]=0; w[keep]=lstsq(h[i][:,keep],R0)[0]
     return h,w,np.mean((h[i]@w-R0)**2)+.00001*keep.sum()
    grid=np.geomspace(.02,2,60); k=min(grid,key=lambda k:estimate(k)[2]); h,w,_=estimate(k); out.append(np.mean(abs(h[j]@w[:,0]-y[j]))); pars.append(k)
   res.append((max(out),out[0],np.mean(out),name,shrink,separate,np.round(pars,3)))
print('sparse best'); print(*sorted(res)[:14],sep='\n')
