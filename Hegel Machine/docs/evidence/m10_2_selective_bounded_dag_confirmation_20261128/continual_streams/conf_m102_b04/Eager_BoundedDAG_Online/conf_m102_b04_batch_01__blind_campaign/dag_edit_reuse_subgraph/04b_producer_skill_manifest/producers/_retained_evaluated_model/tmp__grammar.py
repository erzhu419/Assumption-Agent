exec(open('/tmp/probe.py').read().split('Q={};')[0])
from scipy.optimize import least_squares
p=s['panel_c']; u=p['u']; v1=p['v1'];v2=p['v2'];
for V in (v1,v2):
 for name,f in [('diag2',lambda u:np.c_[affine(u),u*u]),('all2',lambda u:basis(u,2)),('diag23',lambda u:np.c_[affine(u),u*u,u**3]),('diag234',lambda u:np.c_[affine(u),u*u,u**3,u**4]),('trig',lambda u:np.c_[affine(u),np.sin(u),np.cos(u)])]:
  D=f(u); w=np.linalg.lstsq(D,V,rcond=None)[0]; r=V-D@w; h=np.sum(D*np.linalg.pinv(D).T,axis=1); print('panel', name,'loo rmse',np.sqrt(np.mean((r/(1-h[:,None]))**2,axis=0)))
 for j in range(4):
  Y=V[:,j]; D=np.c_[affine(u),u*u]; w=np.linalg.lstsq(D,Y,rcond=None)[0]; print('coef',j,w)
print('scalar endpoints',np.sort(y))
