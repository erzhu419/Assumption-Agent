from pathlib import Path
import pickle,itertools,numpy as np
src=Path('/holdout_task/batch_solver.py').read_text()
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; sw=s['sensor_calibration_sweeps']
for selfkind,cross in itertools.product(('exp','quartic'),('base','q4','v3','v3q4','v3u5','v3abs','v3rat')):
    text=src
    tails={'base':'[]','q4':'[v**4/(1+b[-1]*v*v)]','v3':'[v**3]','v3q4':'[v**3,v**4/(1+b[-1]*v*v)]','v3u5':'[v**3,u**5]','v3abs':'[v**3,v*np.abs(v)]','v3rat':'[v**3/(1+b[-1]*v*v)]'}
    text=text.replace('if extra: cols.extend',f'cols.extend({tails[cross]})\n        if extra: cols.extend')
    ns={}; exec(text,ns); estimate=ns['_estimate']; f,err,b,c=estimate(sw,selfkind)
    oof=[estimate(sw,selfkind,j)[1] for j in range(4)]; q=np.array([np.quantile(abs(np.concatenate([x[k] for x in oof])),[.5,.9,1]) for k in range(4)])
    raw=s['y_obs_raw'].reshape(-1,4); from scipy.optimize import least_squares
    obs=np.array([least_squares(lambda u:f(u[None,:])[0]-y,np.zeros(4),gtol=1e-12).x for y in raw]).ravel(); S,_=ns['_responses'](s,ns['_operator'](s)); Z=s['Z_nuisance']; r=[]
    for i in range(0,len(obs),4):
      hold=slice(i,i+4); tr=np.ones(len(obs),bool); tr[hold]=False; z=Z[tr]; d=np.c_[S[tr],obs[tr]]; p=np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]),z.T@d); D=d-z@p; A,y=D[:,:-1],D[:,-1]; w,_=ns['_box_simplex'](A.T@A+s['source_ridge']*np.eye(A.shape[1]),-A.T@y,s['u_box'],s['total_budget']); r.append(np.sqrt(np.mean((S[hold]@w+Z[hold]@(p[:,-1]-p[:,:-1]@w)-obs[hold])**2)))
    print(selfkind,cross,'sweep worst med/p90/max',np.round(q.max(0),6),'chamber rms/max',np.round([np.sqrt(np.mean(np.square(r))),max(r)],6),'tails',np.round(np.asarray(c)[:,6:],5).tolist(),flush=True)
