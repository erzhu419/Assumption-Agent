import importlib.util,pickle,numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('b06',p+'alpha_w01_batch_06_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_w01_batch_06.pkl','rb'))[0]['dataset']; tr=d['train']; t=d['eval_times']; dt=d['past_dt']; sy=d['yield_stress']; lab=tr['lab_id']; maps=m._calibrate(d['calibration_load_cells']); h,u=m._stress(tr,maps); ov,w=m._plastic(h,u,dt,sy)
x,_,complete,_=m._align(tr,d['test'])
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; levels=np.array([0,.125,.25,.5,.75,1.]); caps=.12+.012*t

def errors(fit, held):
 a={k:v[fit] for k,v in tr.items()}; tau,theta=m._fit(a,x[fit],complete[fit],maps,dt,t,sy); A,B=m._design(tau,h[held],u[held],x[held],dt,t,ov[held],w[held]); return A@theta-y[held],B@theta-rate[held]

def loss(e):
 return np.mean(np.minimum(abs(e),caps)**2)

all_scores=[]
for l in range(4):
 inner=[]; inner_complete=[]
 for j in range(4):
  if j==l: continue
  z=lab==j; e,r=errors(~z&(lab!=l),z); inner.append(e[None]-levels[:,None,None]*r[None,:,None]*t); inner_complete.extend(complete[z])
 inner=np.concatenate(inner,axis=1); c=np.asarray(inner_complete); chosen=[levels[np.argmin([loss(v[c==q]) for v in inner])] for q in range(2)]
 z=lab==l; e,r=errors(~z,z); aug=np.where(complete[z],chosen[1],chosen[0]); base=np.where(complete[z],0.,1.); print('outer lab',l,'inner-selected missing/complete',chosen)
 for above in (False,True):
  i=(abs(u[z])>sy)==above
  rec=[]
  for a in (base,aug):
   ec=(e-a[:,None]*r[:,None]*t)[i]; rr=((1-a)*r)[i]; rec.append([np.mean(ec**2)**.5,loss(ec)**.5,np.mean(rr**2)**.5,np.sqrt(np.max(np.mean(np.minimum(abs(ec),caps)**2,axis=0)))])
  all_scores.append(rec); print('  above',above,'baseline/challenger [raw curve, robust curve, rate, robust worst horizon]',np.round(rec,6))
print('worst by metric, baseline/challenger:',np.round(np.max(all_scores,axis=0),6)); print('mean over slices:',np.round(np.mean(all_scores,axis=0),6))
