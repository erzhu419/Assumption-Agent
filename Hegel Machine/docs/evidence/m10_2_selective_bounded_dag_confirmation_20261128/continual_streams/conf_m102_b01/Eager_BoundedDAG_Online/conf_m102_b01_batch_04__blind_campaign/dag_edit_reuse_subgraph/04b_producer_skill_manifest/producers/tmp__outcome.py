exec(open('/tmp/fit.py').read().split("print('B linear'")[0])
import glob
from scipy.optimize import least_squares
u,v=c['u'],c['v1']; cf=lambda z:np.c_[np.ones(len(z)),z[:,0]*z[:,1],z[:,0]*z[:,3],z[:,1]*z[:,2],z[:,2]*z[:,3]]
C=cf(x)@ls(cf(u),v)
u,v=c['u'],c['v2']; cf2=lambda z,p:aff(z+p[0]*z*z/(1+p[1]*z*z)); fun=lambda p:(cf2(u,p)@ls(cf2(u,p),v)-v).ravel(); p=least_squares(fun,[.55,.18]).x; D=cf2(x,p)@ls(cf2(u,p),v)
Y=np.c_[tr['y'],tr['traj']]; n=len(Y); ids=np.arange(n); folds=[(ids[:52],ids[52:])]+[(a[:52],a[52:]) for a in [np.random.default_rng(j).permutation(ids) for j in range(8)]]
def cv(f,lam=.02):
 f=f[:n]; ret=[]
 for ti,vi in folds:
  F=aff((f-f[ti].mean(0))/np.maximum(f[ti].std(0),1e-8)); z=F[ti]; pred=F[vi]@np.linalg.solve(z.T@z+lam*np.eye(z.shape[1]),z.T@Y[ti]); r=abs(pred-Y[vi]); ret.append([r[:,0].mean(),r[:,1:].mean()])
 return np.r_[ret[0],np.max(ret,0),np.mean(ret,0)]
u,v=b['u'],b['v']; Bs={'cubic':aff(x**3)@ls(aff(u**3),v),'lcubic':np.c_[aff(x),x**3]@ls(np.c_[aff(u),u**3],v)}; Bs.update({a.split('/tmp/cal_B_')[-1][:-4]:np.load(a)['bx'] for a in glob.glob('/tmp/cal_B_*.npz')})
# Fit shared separable rational cubics, allowing input saturation or enhancement.
for name,g in [('cuberat',lambda z,p:(z**3+p[0]*z)/(1+p[1]*z*z)),('cubetan',lambda z,p:np.tanh(p[0]*z**3)+p[1]*z),('cubeatan',lambda z,p:np.arctan(p[0]*z**3)+p[1]*z),('rat1',lambda z,p:z/(1+p[0]*z*z)+p[1]*z**3),('odd',lambda z,p:z+p[0]*z**3+p[1]*z**5),('cubeout',lambda z,p:z)] :
 def err(p):
  Z=aff(g(u,p)); return (Z@ls(Z,v)-v).ravel()
 fit=min((least_squares(err,p0,bounds=([-20,-.40],[40,30]),max_nfev=120) for p0 in ([.22,.22],[1.25,1.25],[4.,.2],[.1,-.2])),key=lambda fit:fit.cost); p=fit.x; print('B warp',name,p,np.mean(abs(err(p)))); Bs[name]=aff(g(x,p))@ls(aff(g(u,p)),v)
for name,M in Bs.items():
 print('B',name,'C2 p',p if False else '')
 results=[]
 for an,L in [('B',M),('BC',np.c_[M,C]),('BD',np.c_[M,D]),('BCD',np.c_[M,C,D]),('BC+D',np.c_[M+C,D]),('BC-D',np.c_[M-C,D]),('B+C+D',M+C+D),('CX',np.c_[C,D,x])]:
  def nlf(z,k):return [z,np.tanh(1.25*z),np.arctan(1.25*z),np.arcsinh(1.25*z),np.sinh(np.arcsinh(4*z)/3),np.sin(1.25*z),z+z**3,np.c_[z,np.tanh(1.25*z)]][k]
  for k in range(8):
   f=nlf(L,k); results.append((cv(f),an,k));
   if k in [0,4] and an in ['BCD','B','BC','BD']: results.append((cv(np.c_[f,q(f[:,:4])]),an+'+Q',k))
 for j in [0,1]:
  for err,an,k in sorted(results,key=lambda a:a[0][j+2])[:4]: print(j,an,k,err.round(5))
