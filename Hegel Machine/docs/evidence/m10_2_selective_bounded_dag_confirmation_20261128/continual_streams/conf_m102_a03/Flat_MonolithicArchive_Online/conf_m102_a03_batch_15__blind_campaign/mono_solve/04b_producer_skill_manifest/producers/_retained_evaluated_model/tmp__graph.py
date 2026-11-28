import pickle,runpy,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
from pathlib import Path
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; arc='/tmp/hegel_continual_archive/conf_m102_a03_batch_'
m=runpy.run_path(arc+'14_solver.py'); z=m['_decode'](tr['x'],m['_calibration'](s['panel_a']))
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
aff=lambda a:np.c_[np.ones(len(a)),a]
def panel(p,k,d):
 u=p['u']; v=p[k]; a=aff(d(u)); c=fit(a,v); return aff(d(z))@c,np.sqrt(np.mean((a@c-v)**2))
b,eb=panel(s['panel_b'],'v',lambda x:np.cos(2.6*x));
for ii in [2,6,8,10,12,14]:
 print(ii,Path(arc+f'{ii:02d}_solver.py').read_text().split('def _calibration')[1].split('def solve_batch')[0].split('def _operator')[-1][-1750:]) if False else None
u=s['panel_c']['u']; v=s['panel_c']['v1']; w=s['panel_c']['v2'];
c,ec=panel(s['panel_c'],'v1',lambda x:np.c_[x,np.arctan(4.5*x),np.sin(2.56*x)]); d,ed=panel(s['panel_c'],'v2',lambda x:x+.2*np.cos(1.3*x)); print('panel errors',eb,ec,ed)
F=-np.expm1(-np.outer(t,[.41255,1.81505])); A=fit(F,tr['traj'].T).T; Y=np.c_[tr['y'],A];
if __name__=='__main__':
 print('y/amps scale',Y.std(0));
 X={'z':z,'b':b,'c':c,'d':d}; pairs={}
 for k,p in X.items():
  for l,q in X.items():
   if k>l:continue
   pairs[k+l]=(p[:,:,None]*q[:,None,:]).reshape(len(z),16) if k!=l else np.column_stack([p[:,i]*p[:,j] for i,j in combinations_with_replacement(range(4),2)])
 def report(name,G):
  G=aff(G); coef=fit(G[:52],Y[:52]); e=G@coef-Y; return (np.round(np.mean(abs(e[52:]),0),4),np.round(np.std(e[:52],0),4),name, G.shape[1])
 print('single and paired main effects')
 for k in X:
  for l in X:
   if k>l:continue
   print(report(k+l,np.c_[X[k],X[l]]))
 print('quadratic candidates')
 L=[]
 for k in X:
  for l in X:
   if k>l:continue
   for r,P in pairs.items():L.append(report(k+l+'+'+r,np.c_[X[k],X[l],P]))
 for i in range(3):print('top',i,sorted(L,key=lambda q:q[0][i])[:7])
 print('all linear + each product')
 for r,P in pairs.items():print(report(r,np.column_stack(list(X.values())+[P])))
