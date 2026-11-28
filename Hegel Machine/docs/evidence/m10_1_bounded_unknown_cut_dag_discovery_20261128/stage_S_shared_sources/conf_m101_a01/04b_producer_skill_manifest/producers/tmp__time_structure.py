from contextlib import redirect_stdout
from io import StringIO
with redirect_stdout(StringIO()):
 from cvprobe import *
from scipy.optimize import least_squares
for name in ('exp','log','rational'):
 def h(rate):
  z=rate*t
  return {'exp':lambda:-np.expm1(-z),'log':lambda:np.log1p(z),'rational':lambda:z/(1+z)}[name]()
 params=[]; residual=[]
 for row in j:
  def fun(r):
   a=h(r[0]);return row-a*(a@row)/(a@a)
  r=least_squares(fun,[1.],bounds=(.005,20.)).x[0]; a=h(r)
  params.append([a@row/(a@a),r]);residual.append(fun([r]))
 p=np.array(params); w=abs(j[:,-1])>.35
 print(name,'temporal RMSE',np.mean(np.array(residual)**2)**.5,'rates on strong rows',np.quantile(p[w,1],[0,.1,.5,.9,1]))
 print('rate r2 strong rows',[(i, round(1-np.sum((poly(x[w],i)@ls(poly(x[w],i),p[w,1])-p[w,1])**2)/np.sum((p[w,1]-p[w,1].mean())**2),4)) for i in [1,2]])
print('times',t)
U,S,vt=np.linalg.svd(j,full_matrices=False)
for label,T in [('dominant',j@vt[0]),('secondary',j@vt[1])]:
 print(label,'MAE per mode')
 for key,z in Z.items():
  for deg in (2,3):
   F=poly(z,deg); a=[]
   for it,iv in folds:
    a.append(np.mean(abs(F[iv]@ls(F[it],T[it])-T[iv])))
   if key=='x': print(deg,np.round(a,4))
