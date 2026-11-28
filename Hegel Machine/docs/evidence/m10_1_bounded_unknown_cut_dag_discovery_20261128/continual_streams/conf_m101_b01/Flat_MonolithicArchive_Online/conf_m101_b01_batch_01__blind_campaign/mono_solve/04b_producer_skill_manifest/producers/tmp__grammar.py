from operators import *
from scipy.special import erf,hyp2f1
from scipy.optimize import least_squares
T=S['train']; X=cal(T['x']); Y=T['y']; Q=T['traj']; times=S['eval_times']
a=least_squares(lambda p:(Q@np.linalg.pinv(-np.expm1(-np.exp(p[:,None])*times))@(-np.expm1(-np.exp(p[:,None])*times))-Q).ravel(),[-1,1]); H=-np.expm1(-np.exp(a.x[:,None])*times); A=Q@np.linalg.pinv(H)
forms={
 'tanh':(lambda x,k:np.tanh(k*x),[1.5]),'erf':(lambda x,k:erf(k*x),[1.5]),'atan':(lambda x,k:np.arctan(k*x),[2]),
 'sat':(lambda x,k,p:x/(1+(k*x)**2)**p,[1,.5]), 'integral':(lambda x,k,p:x*hyp2f1(.5,p,1.5,-(k*x)**2),[1,1]),
 'cuberat':(lambda x,k,p:x**3/(1+(k*x)**2)**p,[1,1]), 'sinh':(lambda x,k:np.arcsinh(k*x),[2]), 'sin':(lambda x,k:np.sin(k*x),[2]),
 'poly5':(lambda x,k:x**3+k*x**5,[.1]),'exp':(lambda x,k:x*np.exp(-(k*x)**2),[1])}
def panel_fit(u,v):
 results=[]
 for name,(f,p0) in forms.items():
  for kind in ['mix','shared','standalone']:
   def design(x,p,f=f,p0=p0,kind=kind):
    u=f(x,*p[:len(p0)]); return np.c_[aff(x),u] if kind=='mix' else np.c_[np.ones(len(x)),u+(p[-1]*x if kind=='shared' else 0)]
   def resid(p):
    d=design(u,p); return (d@fit(d,v)-v).ravel()
   lo=[.02]+([.03] if len(p0)==2 else []); hi=[8]+([5] if len(p0)==2 else [])
   if name=='poly5':lo[0]=-5
   init=list(p0)
   if kind=='shared':lo+=[-20];hi+=[20];init += [.25]
   z=least_squares(resid,init,bounds=(lo,hi),max_nfev=100)
   p=z.x; d=design(u,p); w=fit(d,v)
   results.append((np.std(z.fun),name,kind,p,lambda x,d=design,p=p,w=w:d(x,p)@w))
 return sorted(results,key=lambda a:a[0])
if __name__=='__main__':
 bs=panel_fit(S['panel_b']['u'],S['panel_b']['v']); cs=panel_fit(S['panel_c']['u'],S['panel_c']['v1'])
 for L in [bs,cs]:
  for r in L: print(r[:4])
 u=S['panel_c']['u']; v=S['panel_c']['v2']; w=fit(np.c_[aff(u),u*u],v); print('v2 square ratio',w[5:]/w[1:5]); D=lambda x:np.c_[aff(x),x*x]@w
 from sklearn.model_selection import KFold
 sp=[(np.arange(52),np.arange(52,72)),*KFold(4,shuffle=True,random_state=91).split(Y)]
 ans=[]
 for b,c in itertools.product(bs[:10],cs[:10]):
  z=b[-1](X); q=c[-1](z); d=D(z)
  bases={'quad':(poly(q,2),aff(q)), 'dual':(np.c_[poly(q,2),d],aff(q)), 'diagdual':(np.c_[aff(q),q*q,d],aff(q)), 'dualboth':(np.c_[poly(q,2),d],aff(np.c_[q,d])), 'diags':(np.c_[aff(q),q*q],aff(q))}
  for k,(sy,st) in bases.items():
   ee=[]
   for i,j in sp:
    pr=sy[j]@fit(sy[i],Y[i]); tr=st[j]@fit(st[i],A[i])@H
    ee.append([np.mean(abs(pr-Y[j])),np.mean(abs(tr-Q[j]))])
   e=np.array(ee); ans.append((e[1:].mean()+.25*e[1:].max(),b[1:4],c[1:4],k,e[0],e[1:].mean(0),e[1:].max(0)))
 print('best joint predictions'); [print(r) for r in sorted(ans,key=lambda r:r[0])[:16]]
