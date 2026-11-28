exec(open('/tmp/structure.py').read().split('Y=[]; T=[]')[0])
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
np.set_printoptions(precision=6,suppress=True)
for key,val in s.items():
 if not isinstance(val,dict): print('study',key,val)
q=lambda zz: np.c_[np.ones(len(zz)),zz[:,0]*zz[:,1],zz[:,0]*zz[:,3],zz[:,1]*zz[:,2],zz[:,2]*zz[:,3]]
qc=ls(q(u),p['v1']); print('sparse c RMSE',np.mean((q(u)@qc-p['v1'])**2)**.5)
def dmat(w,zz): return np.c_[ones(zz), np.cos(w[0]*zz+w[1:])]
def dr(w):
 d=dmat(w,u);return (d@ls(d,p['v2'])-p['v2']).ravel()
w=least_squares(dr,[1.27,0,0,0,0]).x; dc=ls(dmat(w,u),p['v2']); print('cos',w,dc)
coords['AinvBlin']=z@B[1:];coords['AinvBt']=z@B[1:].T+B[0];coords['AinvBinvlin']=z@np.linalg.inv(B[1:]);coords['splitAinv']=(x-a*b)/b
res=[[],[]]
for k,zz in coords.items():
 c=q(zz)@qc;d=dmat(w,zz)@dc
 cross=(c[:,:,None]*d[:,None,:]).reshape(n,-1)
 dcos=np.cos(w[0]*zz+w[1:]); bzz=ones(zz)@B
 fs={'z':ones(zz),'poly2':PolynomialFeatures(2).fit_transform(zz),'poly3':PolynomialFeatures(3).fit_transform(zz),'zcd':ones(np.c_[zz,c,d]),'zcddot':ones(np.c_[zz,c,d,(c*d).sum(1)]),'zcdcross':ones(np.c_[zz,c,d,cross]),'poly2cd':np.c_[PolynomialFeatures(2).fit_transform(zz),d], 'poly3cd':np.c_[PolynomialFeatures(3).fit_transform(zz),d], 'poly2cdcross':np.c_[PolynomialFeatures(2).fit_transform(zz),d,cross], 'cosproducts':ones(np.c_[zz,c,dcos,(zz[:,:,None]*dcos[:,None,:]).reshape(n,-1)]),'b+c+d+products':ones(np.c_[zz,c,d,bzz*c,bzz*d,(c*d).sum(1)]),'c+d+z interactions':ones(np.c_[zz,c,d,(zz[:,:,None]*np.c_[c,d][:,None,:]).reshape(n,-1)])}
 for kf,f in fs.items():
  U,S,V=np.linalg.svd(f,full_matrices=False); f=U[:,S>S[0]*1e-10]*np.sqrt(n)
  for jt,target in enumerate((y,t)):
   for lam in [0.,.03,.3,1.,3.]:
    scores=[]
    for r,v in cuts:
     h=f[r]; coeff=ls(h,target[r]) if not lam else np.linalg.solve(h.T@h+lam*np.eye(h.shape[1]),h.T@target[r]); scores.append(np.mean(abs(f[v]@coeff-target[v])))
    res[jt].append((max(scores),np.mean(scores),k,kf,lam,np.round(scores,5)))
for rr in res:
 for val in sorted(rr,key=lambda x:x[:2])[:15]: print(val)
# Check distributions and signatures of unmodeled regimes.
for k,a in [('x',x),('z',z),('y',y),('traj',t)]: print(k,np.quantile(a,[0,.05,.25,.5,.75,.95,1],axis=0))
print('largest absolute y and x', np.c_[x,y,t[:,-1]][np.argsort(abs(y))[-15:]])
