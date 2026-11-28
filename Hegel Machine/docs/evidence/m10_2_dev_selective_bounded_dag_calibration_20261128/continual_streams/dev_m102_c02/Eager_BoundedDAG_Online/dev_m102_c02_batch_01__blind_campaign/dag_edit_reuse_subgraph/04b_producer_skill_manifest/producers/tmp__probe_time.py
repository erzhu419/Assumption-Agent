import numpy as np, pickle, runpy
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True)
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']
a=runpy.run_path(p+'node_00.py')['run_node']({'__input__':s}); st=runpy.run_path(p+'_frozen_nodes/node_01.py')['run_node']({'__input__':s,'node_00':a})
t=st['t']; _,sig,vt=np.linalg.svd(st['fitted'],full_matrices=False); z=vt[:2].T*sig[:2]; print('time',t,'modes',z.T)
fs={'rise':lambda r:-np.expm1(-r*t),'exp':lambda r:np.exp(-r*t),'pulse':lambda r:t*np.exp(-r*t),'sin':lambda r:np.sin(r*t),'cos':lambda r:np.cos(r*t),'t':lambda r:t,'one':lambda r:np.ones_like(t)}
fits=[]
for i,(n,f) in enumerate(fs.items()):
 for m,g in list(fs.items())[i:]:
  if n==m and n in ('one','t'): continue
  def res(rs):
   d=np.column_stack((f(rs[0]),g(rs[1]))); return (z-d@np.linalg.lstsq(d,z,rcond=1e-10)[0]).ravel()
  opts=[least_squares(res, r, bounds=(0.015,40), max_nfev=140) for r in [(0.6,3),(2,7),(6,1),(6,12),(12,4),(12,22)]]
  o=min(opts,key=lambda o:o.cost)
  fits.append((np.sqrt(2*o.cost/z.size),n,m,o.x))
print('best time families',sorted(fits,key=lambda a:a[0])[:12])
