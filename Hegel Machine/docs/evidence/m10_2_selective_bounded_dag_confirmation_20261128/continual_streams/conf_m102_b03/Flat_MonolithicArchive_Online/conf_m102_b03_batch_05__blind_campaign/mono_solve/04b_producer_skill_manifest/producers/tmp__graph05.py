exec(open('/tmp/operators05.py').read().split('p=s[')[0])
from scipy.optimize import least_squares
x=np.vstack((s['train']['x'],s['test']['x'])); z=_calibrate(s['panel_a'],x)
u=s['panel_b']['u']; B=ls(one(u),s['panel_b']['v']); w=one(z)@B
u=s['panel_c']['u']; C=lambda q:one(cyc(q)); D=lambda q:one(np.cos(2.4*q)); c=ls(C(u),s['panel_c']['v1']); d=ls(D(u),s['panel_c']['v2']); print('panel RMS',np.std(C(u)@c-s['panel_c']['v1']),np.std(D(u)@d-s['panel_c']['v2']))
Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; T=Y[:,1:]; H=lambda p:-np.expm1(-t[:,None]*p); rate=least_squares(lambda p:(H(p)@ls(H(p),T[:52].T)-T[:52].T).ravel(),[1.8,.4]).x; h=H(rate); A=ls(h,T.T).T; print('amplitude std',A.std(0),rate)
for xn,U in [('z',z),('b',w)]:
 for yn,V in [('z',z),('b',w)]:
  p,q=C(U)@c,D(V)@d
  dot=np.sum(p*q,axis=1); cross=(p[:,:,None]*q[:,None,:]).reshape(len(p),16)
  for name,F in [('branch',np.c_[p,q]),('dot',np.c_[p,q,dot]),('cross',np.c_[p,q,cross]),('quad',poly(np.c_[p,q][:,:4],2)),('expanded',np.c_[p,q,poly(p,2)[:,4:],poly(q,2)[:,4:],cross]),('aug',np.c_[p,q,cross,z,np.cos(2.4*z),cyc(z)]),('rational',np.c_[p,q,dot,p/(1+q*q),q/(1+p*p)])]:
   F=one(F); P=F[52:72]@ls(F[:52],np.c_[Y[:52,0],A[:52]]); e=P-np.c_[Y[52:72,0],A[52:]]; print(xn,yn,name,np.round(abs(e).mean(0),5),np.mean(abs(P[:,1:]@h.T-T[52:])))
