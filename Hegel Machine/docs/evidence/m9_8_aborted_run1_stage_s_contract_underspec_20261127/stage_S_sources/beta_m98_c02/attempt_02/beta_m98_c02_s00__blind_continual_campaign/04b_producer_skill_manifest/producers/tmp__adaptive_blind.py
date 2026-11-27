exec(open('/tmp/blind_slices.py').read().split('for label,basis')[0])
from scipy.optimize import minimize
for constrained in [False,True]:
 for ratio in [1.,1.8,2.5]:
  vals=[]
  for tag,d,v in cases:
   def profile(S):
    D=np.c_[S@Q,Z]; r=v-S@w0
    a=np.linalg.lstsq(D,r,rcond=None)[0]
    if not constrained: return r-D@a
    P=np.linalg.pinv(Z); A=S-Z@(P@S); t=v-Z@(P@v)
    w=b._quadratic(A,t,1e-9,s['total_budget'],s['u_box'])['weights']
    return t-A@w
   F=np.c_[E,c.slopes]; r0=profile(c.response(base,c.rhs)[0]); scale=max(3e-5,ratio*rms(r0)); penalties=scale*np.r_[np.ones(4),np.full(4,2.5)]
   def fun(p):
    S,_=c.response(base+F@p,c.rhs); return np.r_[profile(S),penalties*p]
   p=least_squares(fun,np.zeros(F.shape[1]),bounds=(-2.5,3),gtol=1e-10).x
   S,U=c.response(base+F@p,c.rhs); St,Ut=c.response(d,c.rhs); w,wt=fit(S,v),fit(St,v)
   vals.append((np.linalg.norm(U@w-Ut@wt)/np.linalg.norm(Ut@wt),np.linalg.norm(w-wt),tag))
  print(constrained,ratio,'public',vals[0], 'worst clean',max(vals[1::2]),'worst noisy',max(vals[2::2]))
