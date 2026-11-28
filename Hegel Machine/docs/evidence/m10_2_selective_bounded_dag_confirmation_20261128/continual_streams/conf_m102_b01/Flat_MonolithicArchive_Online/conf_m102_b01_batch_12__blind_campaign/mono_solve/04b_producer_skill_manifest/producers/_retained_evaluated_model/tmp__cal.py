import numpy as np,pickle,glob,importlib.util
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aff(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def trim(d,v,k=40):
 ids=np.arange(len(v))
 for _ in range(15):
  a=ls(d[ids],v[ids]); e=v-d@a; ids=np.argsort(abs(e))[:k]
 return np.median(abs(e)),a,e
u,v=[s['panel_a'][k] for k in ['u','v']]
for j in range(4):
 x=u[:,j]; vv=v[:,j]; best=[]
 for j2 in range(4):
  if j2==j:continue
  z=u[:,j2]
  for desc,d in [('cyclic',aff(np.c_[x,z,x*z,x**3])),('sat',aff(np.c_[x,np.tanh(1.8*x),np.sin(2.4*z)])),('poly',aff(np.c_[x,z,x*z,x*x,z*z,x**3,z**3,x*x*z,x*z*z])),('mix',aff(np.c_[u,x*z,x*x,z*z,x**3,z**3])),('trig',aff(np.c_[x,z,x*z,x*x,z*z,np.sin(2.4*x),np.sin(2.4*z),np.tanh(1.8*x)]))]:
   er,b,e=trim(d,vv);best.append((er,desc,j2,b))
 print('A',j,sorted(best,key=lambda h:h[0])[:6])
for f in glob.glob('/tmp/hegel_continual_archive/*solver.py'):
 spec=importlib.util.spec_from_file_location('m',f);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if hasattr(m,'_cal_basis'):
  d=m._cal_basis(u); print(f[-12:], [trim(d[:,j],v[:,j])[0] for j in range(4)])
