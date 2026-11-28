from explore import *
np.set_printoptions(precision=5,suppress=True)
u=s['panel_c']['u']; v=s['panel_c']; bp=lin(one(pb['u']),pb['v']); print('B',bp)
def maker(y,g,init):
 def fun(q):
  f=g(u,q[0]);return (f@lin(f,y)-y).ravel()
 q=least_squares(fun,[init],bounds=(.04,3),gtol=1e-13).x[0]; F=g(u,q); co=lin(F,y); print('panel q, co, rmse',q,co,np.std(fun([q])));return lambda x:g(x,q)@co
c=maker(v['v1'],lambda x,q:one(np.c_[x,x/(1+q*abs(x))]),.85)
e=maker(v['v2'],lambda x,q:one(np.c_[x,x*x,x*x/(1+q*x*x)]),.3)
B=lambda x:one(x)@bp
pa=s['panel_a']; ua,va=pa['u'],pa['v']; R=va-(o+k*ua); print('A residual',np.round(np.sort(abs(R),axis=0)[::4],5)); print('A rows',np.round(np.c_[ua,va,R][:13],5));
for i in range(4):
 m=abs(R[:,i])<max(.03,5*np.median(abs(R[:,i]))); g=one(ua[m]); coef=lin(g,va[m,i]); print('A full affine',i,m.sum(),coef,np.std(g@coef-va[m,i]));
 for label,f in [('diag2',one(np.c_[ua[:,i],ua[:,i]**2])),('diag3',one(np.c_[ua[:,i],ua[:,i]**2,ua[:,i]**3])),('full2',PolynomialFeatures(2).fit_transform(ua))]:
  f=f[m]; coef=lin(f,va[m,i]); err=f@coef-va[m,i]; h=np.sum(f*np.linalg.pinv(f).T,axis=1); print(label, np.std(err),np.std(err/(1-h)),coef[:5]);
for tag,panel,targets in [('B',pb,['v']),('C',v,['v1','v2'])]:
 for ky in targets:
  uu=panel['u']; vv=panel[ky]; f=one(uu); rr=vv-f@lin(f,vv); print(tag,ky,'u ranges',uu.min(0),uu.max(0),'lin rms by output',rr.std(0));
print('B gram',bp[1:]@bp[1:].T)
