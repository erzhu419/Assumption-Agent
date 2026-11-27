from chamber_probe import *
import json
from itertools import combinations_with_replacement
past,hold,drift=inp_d
A,G=m._design(tau,inp_d,dt,t)
y=np.c_[yd-A@th,rd-G@th]
mem=hold[:,None]-G[:,:2]*tau
b=np.vstack((np.c_[-np.expm1(-t[:,None]/tau),t],np.r_[1/tau,1.]))
q=np.linalg.lstsq(b,y.T,rcond=None)[0].T
print('tau, theta',tau,th)
print('amplitude correlations',np.corrcoef(np.c_[mem,q,q[:,2]+q[:,1]/tau[1]].T).round(3))
for label,fast in [('theta0*z0',th[0]*mem[:,0]),('fitted z0',mem[:,0]*np.linalg.lstsq(mem[:,:1],q[:,0],rcond=None)[0][0]),('fitted z0,z1',mem@np.linalg.lstsq(mem,q[:,0],rcond=None)[0])]:
 print(label,score(q[:,0]-fast),'rate residual',score(y[:,-1]-fast/tau[0]))
features=np.c_[np.ones(len(hold)),hold,past,mem,drift[:,1:]]
names=['1','hold',*[f'p{i}' for i in range(past.shape[1])],'z0','z1',*[f'x{i}' for i in range(drift.shape[1]-1)]]
terms=[j for k in (1,2,3) for j in combinations_with_replacement(range(1,len(names)),k)]
v=np.column_stack([np.prod(features[:,j],axis=1) for j in terms]); v-=v.mean(axis=0); v/=np.linalg.norm(v,axis=0)
for name,g in [('fast minus 0.6*z0',q[:,0]-.6*mem[:,0]),('slow',q[:,1]),('net slow rate',q[:,2]+q[:,1]/tau[1]),('initial slope minus fast',y[:,-1]-q[:,0]/tau[0])]:
 c=v.T@(g-g.mean())/np.linalg.norm(g-g.mean()); ii=np.argsort(-abs(c))[:10]; print(name,'std',g.std(),'top',[(tuple(names[j] for j in terms[i]),round(c[i],3)) for i in ii]); print('linear x fit',np.linalg.lstsq(drift,g,rcond=None)[0].round(4),'RMSE',np.mean((g-drift@np.linalg.lstsq(drift,g,rcond=None)[0])**2)**.5)
print('inferred displacement: z0 z1 fast slow net_slow_rate x0..3'); print(np.c_[mem,q[:,:2],q[:,2]+q[:,1]/tau[1],drift[:,1:]].round(4))
log=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
print('log root',list(log) if isinstance(log,dict) else len(log))
def scan(v,path=''):
 if isinstance(v,dict):
  for k,w in v.items(): scan(w,f'{path}/{k}')
 elif isinstance(v,list):
  if not v: return
  if isinstance(v[0],dict):
   for i,w in enumerate(v): scan(w,f'{path}/{i}')
  else: print(path, len(v), str(v[:2])[:180],str(v[-2:])[:180])
 elif isinstance(v,(int,float,str)): print(path,str(v)[:160])
scan(log)
