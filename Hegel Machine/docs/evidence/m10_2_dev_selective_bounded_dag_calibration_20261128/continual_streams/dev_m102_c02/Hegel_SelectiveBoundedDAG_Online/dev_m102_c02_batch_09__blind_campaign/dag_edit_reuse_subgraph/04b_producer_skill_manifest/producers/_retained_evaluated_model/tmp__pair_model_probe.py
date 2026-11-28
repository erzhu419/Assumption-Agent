import pickle,numpy as np
from scipy.optimize import least_squares
ss=pickle.load(open('/holdout_task/public_batch.pkl','rb')); print('batches',len(ss))
s=ss[0]['study']; u,v=s['panel_a']['u'],s['panel_a']['v']; m=np.linalg.norm(v-u,axis=1)<5; u,v=u[m],v[m]; w=np.roll(u,2,axis=1)
fs={'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'rat':lambda a:a/(1+a*a),'soft':lambda a:a/np.sqrt(1+a*a),'erf':__import__('scipy.special',fromlist=['erf']).erf}
def fit(f,g,oplinear=False):
 def prof(p,ret=False):
  co=[]; rr=[]
  for j in range(4):
   a,b=u[:,j],w[:,j]; d=np.c_[np.ones(len(u)),a,f(p[0]*a),g(p[1]*b)];
   if oplinear: d=np.c_[d,b]
   c=np.linalg.lstsq(d,v[:,j],rcond=None)[0]; co.append(c);rr.append(d@c-v[:,j]);
  return np.array(co) if ret else np.array(rr).ravel()
 opt=min((least_squares(prof,[k,l],bounds=(.1,5.),max_nfev=90) for k,l in [(1.,2.4),(2.15,2.4),(3.6,3.5)]),key=lambda r:r.cost)
 return np.std(opt.fun),opt.x,prof(opt.x,True)
for n,f in fs.items():
 for o in [False,True]:
  e,p,c=fit(f,np.sin,o); print(n,o,e,p,np.round(c,5).tolist())
print('telemetry',s['telemetry_vec']); print('rows',[len(s['train']['x']),len(s['test']['x'])]); print('v-u median/max',np.quantile(np.linalg.norm(v-u,axis=1),[.5,1]))
