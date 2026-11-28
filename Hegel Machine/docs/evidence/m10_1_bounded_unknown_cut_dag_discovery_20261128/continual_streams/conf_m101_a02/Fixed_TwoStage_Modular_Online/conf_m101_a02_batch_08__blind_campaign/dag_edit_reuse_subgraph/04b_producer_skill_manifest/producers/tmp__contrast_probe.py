exec(open('/tmp/readout_probe.py').read().split('np.set_printoptions')[0])
np.set_printoptions(precision=3,suppress=True)
C=np.linalg.lstsq(Y[:,1:-1],Y[:,-1],rcond=None)[0]; print('time fit',C)
A=poly(z,1); D=poly(z,2); Q=np.linalg.qr(A)[0]; E=Y-Q@(Q.T@Y); alpha=(E[:,0]@E[:,-1])/(E[:,-1]@E[:,-1]); T=np.column_stack((Y[:,0],Y[:,-1],Y[:,0]-alpha*Y[:,-1])); print('alpha',alpha)
for qn,q in [('z',z),('u',u),('x',x)]:
 for d in [1,2,3]:
  X=poly(q,d); c=np.linalg.lstsq(X,T,rcond=None)[0]; print(qn,d,np.std(T-X@c,0),c[:5].T)
 c=np.linalg.lstsq(poly(q,2),T,rcond=None)[0]; print('quadratic',qn,c[5:].T)
xr=T-D@np.linalg.lstsq(D,T,rcond=None)[0]; I=np.argsort(-np.abs(xr[:,0]))[:20];print('largest residuals row, z, u, y, t, residual:',np.c_[I,z[I],u[I],T[I,:2],xr[I,:2]])
B=s['panel_b']; kb,gb=m._fit_b(B['u'],B['v']);
for qn,q in [('inv-z-linear', (z-gb[0])@np.linalg.pinv(gb[5:])),('B nonlinear only',m._b_design(u,kb)[:,5:]@gb[5:]),('B whitened',z@np.linalg.inv(np.linalg.cholesky(np.cov(z.T))).T)]:
 print(qn,check(poly(q,2))); print('quad coefficients',np.linalg.lstsq(poly(q,2),T,rcond=None)[0].T)
