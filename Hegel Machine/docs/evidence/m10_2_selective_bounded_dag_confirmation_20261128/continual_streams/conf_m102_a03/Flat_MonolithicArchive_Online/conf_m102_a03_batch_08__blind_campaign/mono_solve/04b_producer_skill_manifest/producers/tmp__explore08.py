import sys,pickle,numpy as np
sys.path.insert(0,'/tmp/hegel_continual_archive')
import conf_m102_a03_batch_06_solver as m
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; z=m._decode(s['train']['x'],m._calibration(s['panel_a'])); p,b=m._operator(s['panel_b']); print('B warp',p,'matrix',b)
def aff(x):return np.column_stack([np.ones(len(x)),x])
def pol(x,d):return np.column_stack([np.prod(x[:,js],axis=1) for k in range(1,d+1) for js in combinations_with_replacement(range(x.shape[1]),k)])
def probe(a,y=y):
 c=np.linalg.lstsq(a[:52],y[:52],rcond=None)[0]; e=a@c-y
 return np.mean(np.abs(e[:52])),np.mean(np.abs(e[52:]))
for name in ('panel_b','panel_c'):
 panel=s[name]; u=panel['u']
 for key,v in panel.items():
  if key=='u':continue
  for typ in ('linear','quad','warp'):
   a=aff(u) if typ=='linear' else aff(pol(u,2))
   if typ=='warp':
    q,c=m._operator({'u':u,'v':v}); a=aff(m._warp(u,q));print(name,key,'warp q',q)
   print(name,key,typ, np.sqrt(np.mean((a@np.linalg.lstsq(a,v,rcond=None)[0]-v)**2)))
for j in (1,2,3): print('z poly',j,probe(aff(pol(z,j))))
for th in (0,.15,.22,.3,.4,.5,.6):
 for ex in (1,1.25,1.5,2):
  h=m._warp(z,[th,ex]);print('warp',th,ex,'h/z',probe(aff(np.c_[z,h])),'h+quad',probe(aff(np.c_[z,pol(h,2)])),'mixed',probe(m._mixed(z,h)))
