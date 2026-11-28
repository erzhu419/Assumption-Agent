from operators import *
from itertools import combinations,product,combinations_with_replacement
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
x=s['train']['x']; z=(x-a)/b; y=s['train']['y']; tr=s['train']['traj']; times=s['eval_times']; ht=-np.expm1(-times[:,None]*np.array([.38,1.8,5.4])); T=fit(ht,tr.T).T
p=PolynomialFeatures(3); P=p.fit_transform(z); R=np.c_[y,T]-P@fit(P,np.c_[y,T]); print('cubic residual sv',np.linalg.svd(R,compute_uv=False)); print('cubic coeffs',list(zip(p.get_feature_names_out(),fit(P,np.c_[y,T]).round(3))));print('pair quotient',abs(T[:,2]-(P@fit(P,T))[:,2]).mean())
c=s['panel_c']; u=c['u']; ij=[(0,1),(0,3),(1,2),(2,3)]; J=lambda w:F(np.column_stack([w[:,i]*w[:,j] for i,j in ij])); A1=fit(J(u),c['v1']); G=lambda w:F(np.cos(2.4*w)); A2=fit(G(u),c['v2']); f=lambda w:J(w)@A1; h=lambda w:G(w)@A2
V=np.c_[y,tr]; ans=[]
def out(u,v):return (u[:,:,None]*v[:,None,:]).reshape(len(u),-1)
def check(M,key):
 if M.shape[1]>47:return
 coef=fit(M[:52],V[:52]); e=abs(M@coef-V); z0=e[:52].mean(0);z1=e[52:].mean(0); ans.append((z0[0],z0[1:].mean(),z1[0],z1[1:].mean(),key))
for wname,w in [('z',z),('Bz',Bf(z)),('BL',z@B[1:]),('Bt',Binv(z)),('x',x),('Btz',(z-B[0])@B[1:].T),('zBT',z@B[1:].T),('zB2', Bf(Bf(z)))]:
 for g in [.5,.8,1,1.15,1.25,1.5,1.8,2,2.4,3,4]:
  for sh in [0,.22,-.22]:
   d=h(w*g+sh); cc=f(w*g+sh)
   for lname,l in [('z',z),('w',w),('none',np.empty((len(w),0)))]:
    for ca in [f(z),f(w),cc]:
     for pn,p2 in [('p2',PolynomialFeatures(2,include_bias=False).fit_transform(l) if l.shape[1] else l),('p1',l)]:
      M=F(np.c_[p2,ca,d]); dot=np.sum(ca*d,axis=1)
      for nm,U in [('linear',d[:,:0]),('outer',out(ca,d)),('dot',dot),('crossZ',out(w,d)),('diag',ca*d),('rat',ca/(1+.22*d*d)),('ratc',d/(1+.22*ca*ca)),('mix',(ca+d)/(1+.22*(ca-d)**2))]: check(np.c_[M,U],(wname,g,sh,lname,pn,nm))
 for fun in [np.sin,np.cos,np.tanh]:
  for g in np.linspace(.5,6,23):
   ww=g*w; v=fun(ww); check(np.c_[P,v],(wname,fun.__name__,g,'p3'));check(np.c_[P[:,:15],v,out(w,v)],(wname,fun.__name__,g,'p2cross'))
for k in [0,1,2,3]:print('best',k,*sorted(ans,key=lambda r:r[k])[:8],sep='\n')
