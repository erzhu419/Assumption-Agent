from probe import *
u=s['panel_c']['u']; v=s['panel_c']['v1']; q=s['panel_c']['v2']
def f(x,p):return np.sign(x)*abs(x)**p[0]/(1+p[1]*abs(x)**p[2])
def resid(p):
 h=aff(f(u,p));return (h@fit(h,v)-v).ravel()
p=least_squares(resid,[.97,.94,1.26],gtol=1e-13).x; C=fit(aff(f(u,p)),v)
print('C1',p,C,'rmse',np.std(resid(p)))
w=aff(f(u,p))@C
D={'u':u,'f':f(u,p),'w':w}
for j in ['u','f','w']:
 for k in ['u','f','w']:
  for kind,extra in [('square',D[k]**2),('cube',D[k]**3),('quad',a._quad(D[k])),('signedsq',D[k]*abs(D[k])),('cyc',D[k]*np.roll(D[k],-1,axis=1)),('diag',D[k][:,:2]*D[k][:,2:])]:
   A=aff(np.c_[D[j],extra]); rr=np.std(A@fit(A,q)-q);print('C2',j,k,kind,round(rr,7)) if rr<.012 else None
B=fit(aff(s['panel_b']['u']),s['panel_b']['v']); b=aff(z)@B
F={'z':z,'b':b,'Cz':aff(f(z,p))@C,'Cb':aff(f(b,p))@C}; F['BCz']=aff(F['Cz'])@B
for name,x in F.items():
 for kind,xx in [('lin',x),('pair',np.c_[x,x[:,:2]*x[:,2:]]),('quad',np.c_[x,a._quad(x)]),('cube',np.c_[x,a._quad(x),x**3])]:
  H=aff(xx); y=H[52:]@fit(H[:52],np.c_[tr['y'],tr['traj']][:52]); e=np.mean(abs(y-np.c_[tr['y'],tr['traj']][52:]),axis=0); print(name,kind,np.round([e[0],e[1:].mean()],5))
T=tr['traj']; print('rates')
def basis(p):return -np.expm1(-np.exp(p[:,None])*t)
for start in [[-.5,1],[-2,1],[-5,1],[0,.3]]:
 def err(p):
  H=basis(p); return (T-fit(H.T,T.T).T@H).ravel()
 opt=least_squares(err,start,gtol=1e-12,max_nfev=400); print(np.exp(opt.x),np.std(opt.fun))
np.set_printoptions(precision=7,suppress=True)
print('calib coefficients')
sp=importlib.util.spec_from_file_location('a3','/tmp/hegel_continual_archive/dev_m102_c02_batch_03_solver.py'); a3=importlib.util.module_from_spec(sp);sp.loader.exec_module(a3); print(a3._calibration(s['panel_a']))
