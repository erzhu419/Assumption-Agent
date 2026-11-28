exec(open('/tmp/operators.py').read().split('ops={}')[0])
from itertools import combinations_with_replacement
p=s['panel_b']; o=identify(p['u'],p['v'],common); e,n,par,f,c=o[0]; B=lambda z:af(f(z,*par))@c
w=B(z); h=f(z,*par); Y=np.c_[y,T]; ix=np.arange(52); val=np.arange(52,72)
print('B',e,n,par,c,'range',w.min(0),w.max(0))
def polys(w,d):return np.column_stack([np.prod(w[:,ij],axis=1) for k in range(1,d+1) for ij in combinations_with_replacement(range(w.shape[1]),k)])
def assess(name,Q,regs=(0,)):
 Q=np.asarray(Q); u=Q[ix].mean(0); d=Q[ix].std(0).clip(1e-9); q=af((Q-u)/d); U,S,V=np.linalg.svd(q[ix],full_matrices=False); out=[]
 for reg in regs:
  co=V.T@((S/(S*S+reg+1e-18))[:,None]*(U.T@Y[ix])); er=np.mean(abs(q[val]@co-Y[val]),0); out.append((round(er[0],5),round(er[1:].mean(),5),reg))
 print(name,Q.shape[1],out)
for nam,a0 in [('w',w),('h',h),('z',z)]:
 for d in (2,3,4,5):assess((nam,d),polys(a0,d),(0,.001,.03,.3,2,8))
# Profile a shared quadratic normalization of a polynomial numerator.
for nam,a0 in [('w',w),('h',h),('z',z)]:
 for k in (1,2,3):
  q=af(polys(a0,k));
  for j,b in [('norm',(a0*a0).sum(1,keepdims=True)),('squares',a0*a0),('pairs',polys(a0,2)[:,4:])]:
   for iy in (0,8):
    def calc(p):return q/(1+b@p).clip(.07)[:,None]
    r=least_squares(lambda p:(lambda L:L@fit(L,Y[ix,iy])-Y[ix,iy])(calc(p)[ix]),np.zeros(b.shape[1])+.05,bounds=(-.4,8),max_nfev=65); Q=calc(r.x); pr=Q@fit(Q[ix],Y[ix,iy]); print('rational',nam,k,j,iy,np.mean(abs(pr[val]-Y[val,iy])),np.mean(abs(r.fun)),r.x)
