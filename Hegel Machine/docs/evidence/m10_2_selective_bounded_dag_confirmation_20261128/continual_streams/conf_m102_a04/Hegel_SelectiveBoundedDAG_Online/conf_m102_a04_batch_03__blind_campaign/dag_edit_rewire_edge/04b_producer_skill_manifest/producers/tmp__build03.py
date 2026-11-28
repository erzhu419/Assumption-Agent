exec(open('/tmp/inspect03.py').read().split("print('times'")[0])
from scipy.optimize import least_squares
p=s['panel_a']; u,v=p['u'],p['v']; keep=np.max(abs(v-u),axis=1)<3
j=np.roll(np.arange(4),-1); cal=[]
for k,l in enumerate(j):
 f=np.c_[np.ones(len(u)),u[:,k],u[:,l],u[:,k]*u[:,l],u[:,k]**3]
 cal.append(np.linalg.lstsq(f[keep],v[keep,k],rcond=None)[0])
p=s['panel_c']; u=p['u']; maps=[]
for key, init, bounds in [('v1',[.85,-.01],([.02,-1],[10,1])),('v2',[.25],(.08,.65))]:
 def des(x,q):
  if key=='v1': return a.one(x/(1+q[0]*abs(x))+q[1]*x)
  h=x*x; return np.c_[a.one(x),h,h/(1+q[0]*h)]
 def res(q):
  f=des(u,q); return (f@a.lin(f,p[key])-p[key]).ravel()
 q=least_squares(res,init,bounds=bounds,**({'gtol':1e-13} if key=='v2' else {})).x
 maps.extend([q.tolist(),a.lin(des(u,q),p[key]).tolist()])
off,sc=a._align(s['panel_a']); np.set_printoptions(precision=10); print('calibration',cal,'panel map params',maps[::2],'axiswise',off,sc)
with open('/holdout_task/node_03.py','w') as f:
 f.write('import numpy as np\nfrom scipy.optimize import least_squares\n\n# Calibration coefficients estimated from the public reference panels.\n')
 for name,val in zip(('CAL','Q1','C1','Q2','C2','OFFSET','SCALE'),(np.array(cal).T,*maps,off,sc)):
  f.write(f'{name} = np.array({np.asarray(val).tolist()!r})\n')
