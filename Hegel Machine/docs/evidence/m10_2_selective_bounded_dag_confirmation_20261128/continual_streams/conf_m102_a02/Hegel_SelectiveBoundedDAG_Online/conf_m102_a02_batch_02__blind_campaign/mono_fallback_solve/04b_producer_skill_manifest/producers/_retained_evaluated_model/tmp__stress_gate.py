import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.model_selection import KFold,ShuffleSplit
sp=importlib.util.spec_from_file_location('m','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; off,scale=m._calibration(s['panel_a']); x=(t['x']-off)/scale; q=m._quadratic(x); ids=np.arange(len(x));
splits=[('mandatory',ids[:52],ids[52:])]+[(f'fold{i}',a,b) for i,(a,b) in enumerate(KFold(4,shuffle=True,random_state=12).split(x))]+[(f'rand{i}',a,b) for i,(a,b) in enumerate(ShuffleSplit(96,test_size=20,random_state=1199).split(x))];
for label,r in [('boundary',abs(x[:,2])),('norm',-np.linalg.norm(x,axis=1))]:
 b=np.argsort(r)[:20]; splits.append((label,np.setdiff1d(ids,b),b))
def fit(a,b):
 qa,xa,ya=q[a],x[a],y[a]; qb,xb=q[b],x[b];
 def fit_one(start):
  def resid(p):
   f=m._gated(xa,qa,p); return f@np.linalg.lstsq(f,ya,rcond=None)[0]-ya
  o=least_squares(resid,start,bounds=([1,-.35,-3,-3,-3],[35,.35,3,3,3]),max_nfev=100)
  f=m._gated(xa,qa,o.x); pred=m._gated(xb,qb,o.x)@np.linalg.lstsq(f,ya,rcond=None)[0]; return (o.cost,o.x,pred)
 starts=[[8,0,0,0,0],[3,0,0,0,0],[14,0,0,0,0],[8,-.12,0,0,0],[8,.12,0,0,0],[8,0,1,0,0],[8,0,-1,0,0]]
 sol=[fit_one(st) for st in starts]; best=min(sol,key=lambda z:z[0]); fa=np.c_[qa,qa*np.tanh(8*xa[:,2:3])]; fb=np.c_[qb,qb*np.tanh(8*xb[:,2:3])]; base=fb@np.linalg.lstsq(fa,ya,rcond=None)[0];
 def mae(p):return np.mean(abs(p-y[b]));
 return [mae(sol[0][2]),mae(best[2]),mae(.8*best[2]+.2*base),mae(.5*best[2]+.5*base),mae(base)],sol[0][:2],best[:2]
rows=[]
for name,a,b in splits:
 e,p1,p2=fit(a,b);rows.append(e)
 if name in ('mandatory','boundary','norm') or e[0]>.028 or e[1]>.028: print(name,np.round(e,6),'local',p1,'multi',p2,flush=True)
print('random mean / worst',np.mean(rows[5:-2],axis=0),np.max(rows[5:-2],axis=0)); print('full fit:',fit(ids,ids)[1:])
