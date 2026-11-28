exec(open('/tmp/state_probe.py').read().split('results=[]')[0].replace('for name,f in funcs.items():','for name,f in []:'))
from scipy.optimize import minimize_scalar
P=np.c_[one,a,b,pair]; times=s['eval_times'];
def evaluate(fn):
 e=[]
 for j,k,h,y in folds:
  F,lam=fn(j,y); d=F[j]; coef=np.linalg.lstsq(d,y,rcond=None)[0] if lam==0 else np.linalg.solve(d.T@d+lam*np.eye(d.shape[1]),d.T@y)
  r=abs(F[k]@coef@h.T-T[k]); e.append([r.mean(),r.mean(0).max()]);
 return np.asarray(e)
res=[]
for label,w in [('b',b),('a',a),('z',z)]:
 fns={'sin':np.sin,'tanh':np.tanh,'soft':lambda x:x/np.sqrt(1+x*x),'cos':np.cos,'exp':lambda x:np.exp(np.clip(x,-20,20)),'rat':lambda x:1/(1+x*x)}
 for name,f in fns.items():
  for quadratic in [False,True]:
   for reg in [0,.0003,.003]:
    def fn(j,y):
     D=P if not quadratic else np.c_[P,w*w];
     def design(c): return np.c_[D,f(c*w)]
     def loss(c):
      d=design(c)[j]; r=y[:,0]-d@np.linalg.lstsq(d,y[:,0],rcond=None)[0]; return r@r
     ws=np.linspace(.3,4,26); i=np.argmin([loss(c) for c in ws]); c=minimize_scalar(loss,bounds=(ws[max(i-1,0)],ws[min(i+1,len(ws)-1)]),method='bounded').x
     return design(c),reg
    E=evaluate(fn); res.append((E[:,1].max(),E[:,0].max(),E[0,0],E[:,0].mean(),label,name,quadratic,reg))
for d in sorted(res)[:15]: print('structured',d)
for name,w in [('a',a),('b',b),('z',z)]:
 A=np.c_[P,w*w,w**3,w**4,w**5]; coef=np.linalg.lstsq(A,Y,rcond=None)[0];print('poly',name,np.round(coef.T,3),np.mean(abs(A@coef-Y),0))
p=s['panel_a'];u,v=p['u'],p['v']; i=np.arange(4);j=(i+1)%4; L=np.c_[one[:len(u)],u]; mask=np.ones(len(u),bool)
for _ in range(8):
 e=np.max(abs(L@np.linalg.lstsq(L[mask],v[mask],rcond=None)[0]-v),axis=1); mask=e<max(.001,3*np.median(e))
for lab in ['affine','cyclic']:
 errors=[]
 for c in i:
  D=L if lab=='affine' else np.c_[np.ones(len(u)),u[:,c],u[:,j[c]],u[:,c]*u[:,j[c]],u[:,c]**3]; D=D[mask]; Q,_=np.linalg.qr(D); yy=v[mask,c];errors.extend(abs((yy-Q@(Q.T@yy))/(1-(Q*Q).sum(1))))
 print('calibration inlier LOO',lab,len(errors)//4,np.mean(errors),np.max(errors))
