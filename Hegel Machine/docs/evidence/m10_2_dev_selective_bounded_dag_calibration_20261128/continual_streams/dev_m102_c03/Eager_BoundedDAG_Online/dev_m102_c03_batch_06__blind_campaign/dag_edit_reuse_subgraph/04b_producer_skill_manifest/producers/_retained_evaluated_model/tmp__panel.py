import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
for pn,key in [('panel_a','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[pn]['u'];v=s[pn][key]; ok=np.linalg.norm(v,axis=1)<7;u=u[ok];v=v[ok];print(pn,key,u.shape)
 for d in [1,2,3]:
  ids=[ix for k in range(d+1) for ix in it.combinations_with_replacement(range(4),k)]; f=np.column_stack([np.prod(u[:,ix],axis=1) for ix in ids]); w=np.linalg.lstsq(f,v,rcond=None)[0];e=f@w-v;h=np.sum(f*np.linalg.pinv(f).T,axis=1);print(d,np.sqrt(np.mean(e**2)),np.sqrt(np.mean((e/np.maximum(1-h[:,None],1e-9))**2))); 
  if d==2:
   for j in range(4):print(j,sorted(zip(ids,w[:,j].round(4)),key=lambda z:-abs(z[1]))[:12]);
 if pn!='panel_a':continue
 lin=np.c_[np.ones(len(u)),u];
 for j in range(4):
  y=v[:,j]; a=np.linalg.lstsq(lin,y,rcond=None)[0]; print('A residual',j);r=y-lin@a
  args=[(str(k),u[:,k]) for k in range(4)]+[(f'{i}{sig}{k}',u[:,i]+sig*u[:,k]) for i,k in it.combinations(range(4),2) for sig in [-1,1]]
  results=[]
  for nm,z in args:
   for typ in ['power','cos','exp','rat','expabs','sin']:
    def f(k):return {'power':lambda:abs(z)**k,'cos':lambda:np.cos(k*z),'exp':lambda:np.exp(-k*z*z),'rat':lambda:1/(1+k*z*z),'expabs':lambda:np.exp(-k*abs(z)),'sin':lambda:np.sin(k*z)}[typ]()
    def residual(k):
     m=np.c_[lin,f(k[0])];return m@np.linalg.lstsq(m,y,rcond=None)[0]-y
    o=least_squares(residual,[1.],bounds=(.01,8),max_nfev=40);results.append((np.std(o.fun),nm,typ,o.x[0]))
  print(sorted(results)[:10])
