exec(open('/tmp/readout_probe.py').read().split("for kind in ['diag'")[0]); from scipy.optimize import least_squares
z=cal(x,'cyc'); u,v=(s['panel_b'][k] for k in ('u','v')); b=np.linalg.lstsq(np.c_[np.ones(len(u)),u],v,rcond=None)[0]; print('b',b,'sv',np.linalg.svd(b[1:])[1]); w=np.c_[np.ones(len(z)),z]@b; wi=(z-b[0])@np.linalg.inv(b[1:]); u=s['panel_c']['u']; ix=[0,2,3,5]; b1=np.linalg.lstsq(np.c_[np.ones(len(u)),pp(u)[:,ix]],s['panel_c']['v1'],rcond=None)[0]; b2=np.linalg.lstsq(np.c_[np.ones(len(u)),np.cos(2.4*u)],s['panel_c']['v2'],rcond=None)[0];
for zn,h in [('z',z),('w',w),('wi',wi),('x',x)]:
 p=pp(h); c=np.cos(2.4*h); P=np.c_[np.ones(len(h)),p[:,ix]]@b1; C=np.c_[np.ones(len(h)),c]@b2; base=np.c_[p,c];
 for n,g in [('base',base),('restrict',np.c_[p[:,ix],c]),('panel',np.c_[P,C]),('panel_lins',np.c_[P,C,h]),('panel_tanh',np.c_[P,C,np.tanh(P),np.tanh(C)]),('panel_cubic',np.c_[P,C,P**2,C**2,P**3,C**3]),('pair_sat',np.c_[p,c,np.tanh(p),np.tanh(c)]),('panel_cross',np.c_[P,C,P*C]),('xprod',np.c_[P,C,*[(P[:,i]*C[:,j]) for i in range(4) for j in range(4)]]),('panel_sin',np.c_[P,C,np.sin(P),np.sin(C),np.cos(P),np.cos(C)]),('pair_h',np.c_[p,c,p/(1+.5*np.column_stack([h[:,i]**2+h[:,j]**2 for i,j in pairs]))])]:
  print(zn,n,scores(g,.03))
# calibration panel residual: test candidate nonlinear corrections with leave-one-out error
u,v=(s['panel_a'][k] for k in ('u','v')); h=np.c_[np.ones(len(u)),u]; keep=np.linalg.norm(v-h@np.linalg.lstsq(h,v,rcond=None)[0],axis=1)<3; u=u[keep];v=v[keep];
for freq in [.5,1,1.5,2,2.4,3,4,5]:
 for func in (np.sin,np.cos,np.tanh):
  m=[]
  for i in range(4):
   a,b=u[:,i],u[:,(i+1)%4]; H=np.c_[np.ones(len(a)),a,b,a*b];
   for tag,g in enumerate([a,b,a+b,a-b,a*b,a*a+b*b,u[:,(i+2)%4],u[:,(i+3)%4]]):
    F=np.c_[H,func(freq*g)]; Q=np.linalg.qr(F)[0]; e=(v[:,i]-Q@(Q.T@v[:,i]))/(1-(Q**2).sum(1)); m.append((tag,np.mean(e*e)))
  print('cal',freq,func.__name__,np.round(np.sqrt(np.array(m)[:,1].reshape(4,8).mean(0)),5))
