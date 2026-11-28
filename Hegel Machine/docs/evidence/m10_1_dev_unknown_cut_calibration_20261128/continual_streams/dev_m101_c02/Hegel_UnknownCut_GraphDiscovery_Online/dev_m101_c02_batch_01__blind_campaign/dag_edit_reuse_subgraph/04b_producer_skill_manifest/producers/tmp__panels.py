import pickle,numpy as np
from itertools import combinations
np.set_printoptions(precision=5,suppress=True,linewidth=150)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def library(u):
 d={'1':np.ones(len(u))}
 for i,x in enumerate(u.T): d[f'x{i}']=x
 for i,x in enumerate(u.T): d[f'x{i}²']=x*x
 for i,j in combinations(range(4),2): d[f'x{i}x{j}']=u[:,i]*u[:,j]
 for w in [0.5,0.8,1,1.2,1.4,1.5,1.6,1.8,2,2.2,2.5,3,np.pi,4,5]:
  for i,x in enumerate(u.T):
   d[f'sin({w:.3g}x{i})']=np.sin(w*x);d[f'cos({w:.3g}x{i})']=np.cos(w*x)
 return d
for p in ['panel_b','panel_c']:
 u=s[p]['u'];d=library(u); names=list(d); X=np.array(list(d.values())).T
 L=X[:,:5]; H=X-L@np.linalg.lstsq(L,X,rcond=None)[0]; H=H[:,5:]; H/=np.linalg.norm(H,axis=0)
 for v in set(s[p])-{'u'}:
  Y=s[p][v]; B=np.linalg.lstsq(L,Y,rcond=None)[0]; R=Y-L@B
  print(p,v,'linear coefficients',B,'resid std',R.std(0),'quad std',(Y-X[:,:15]@np.linalg.lstsq(X[:,:15],Y,rcond=None)[0]).std(0))
  for j,y in enumerate(R.T):
   C=H.T@y; ids=np.argsort(abs(C))[-12:][::-1];print(j,[(names[i+5],round(C[i]/np.linalg.norm(y),3)) for i in ids])
print('times',s['eval_times'],'telemetry',s['telemetry_vec'],'summary',s['train']['summary'])
T=s['train']['traj'];print('Y range',s['train']['y'].min(),s['train']['y'].max(),'T range',T.min(),T.max(),'T svd',np.linalg.svd(T,compute_uv=False));print(T[:8])
