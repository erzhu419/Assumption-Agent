import importlib.util as iu, itertools, json, pickle
import numpy as np
s=iu.spec_from_file_location('prior','/tmp/hegel_continual_archive/alpha_w01_batch_06_solver.py'); p=iu.module_from_spec(s); s.loader.exec_module(p)
d=pickle.load(open('/holdout_task/public_alpha_w01_batch_07.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; dt,t,sy=[d[k] for k in ['past_dt','eval_times','yield_stress']]; x,_,complete,_=p._align(tr,te); lab=tr['lab_id']; panel=d['calibration_load_cells']; h=json.loads(p._ARCHIVE.read_text()); maps=p._calibrate(panel)
choices=[]
for l in [2,3]:
 a,b=p._checks(panel,l); candidates={}
 for i,j in itertools.combinations(range(len(a)),2):
  line=p._line(a[[i,j]],b[[i,j]])
  if line[0]>0: candidates[str(i)+str(j)]=line
 for name,k in [('last',4),('identity',0)]: candidates[name]=p._line(*p._checks(h[k],l))
 candidates['pooled']=p._line(a,b); choices.append(candidates)
print('Wing A',maps[:2]); print('Candidates',choices)
rng=np.random.default_rng(7); val=np.zeros(len(lab),bool)
for l in range(4):
 ids=np.flatnonzero(lab==l); val[rng.permutation(ids)[:len(ids)//3]]=True
fit={k:v[~val] for k,v in tr.items()}
for a,b in itertools.product(*choices):
 maps[2:]=[choices[0][a],choices[1][b]]
 tau,th=p._fit(fit,x[~val],complete[~val],maps,dt,t,sy)
 hist,hold=p._stress(tr,maps); A,B=p._design(tau,hist,hold,x,dt,t,*p._plastic(hist,hold,dt,sy))
 ec=A@th+tr['strain_0'][:,None]-tr['strain_curves_obs']; er=B@th-tr['instant_rate_obs']; diff=ec-ec[:,[-1]]*t/t[-1]
 print(a,b,'tau',tau.round(3),'plastic',th[7:].round(5),'val RMSE curve/rate/curvature',[[round(np.sqrt(np.mean(z[val&(lab==l)]**2)),6) for l in range(4)] for z in (ec,er,diff)])
