from panel12 import *
from scipy.optimize import least_squares
np.set_printoptions(precision=6,suppress=True,linewidth=150)
a=s['panel_a']; u,v=a['u'],a['v']; keep=np.linalg.norm(v-u,axis=1)<4; u,v=u[keep],v[keep]; print('A survivors',len(u));
for d in [1,2,3,4]:
 f=poly(u,d); w=ls(f,v); print('A degree',d,'RMSE',np.sqrt(np.mean((f@w-v)**2))); print(w if d==2 else '')
for p,key in [('b','v'),('c','v1'),('c','v2')]:
 q=s['panel_'+p]; r=fitpanel(q['u'],q[key]); print(p,key,[(i[:4]) for i in r if 'cube' not in i[2]][:12]);
 for i in [i for i in r if 'cube' not in i[2]][:2]: print(i[2],i[-1]);
 print('urange',q['u'].min(0),q['u'].max(0))
print('t',s['eval_times']); print('x range',tr['x'].min(0),tr['x'].max(0)); print('y vs traj',np.corrcoef(np.c_[tr['y'],tr['traj']].T)[0]);
