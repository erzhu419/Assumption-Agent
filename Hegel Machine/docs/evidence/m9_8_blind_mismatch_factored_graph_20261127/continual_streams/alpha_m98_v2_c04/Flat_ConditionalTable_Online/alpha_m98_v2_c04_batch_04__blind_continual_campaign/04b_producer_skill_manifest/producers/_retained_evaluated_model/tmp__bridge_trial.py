import importlib.util,pickle,numpy as np
from itertools import combinations_with_replacement
p='/holdout_task/batch_solver.py'; s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr,te=d['train'],d['test']; c=d['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; dt,t=d['past_dt'],d['eval_times']
full=[()] + [j for n in range(1,4) for j in combinations_with_replacement(range(4),n)]
aidx=np.arange(4); prev=(aidx-1)%4

def basis(z):
 return np.c_[np.ones(len(z)),z,z**3,z*z[:,prev]]

def fit(xx,yy):
 a=np.array([np.prod(xx[:,j],axis=1) for j in full]).T
 e=yy-a@np.linalg.lstsq(a,yy,rcond=None)[0]; _,_,v=np.linalg.svd(e,full_matrices=False); n=v[0]; proj=np.eye(4)-np.outer(n,n)
 b=basis(xx); delta=yy-b@np.linalg.lstsq(b,yy@proj,rcond=None)[0]; score=delta@n
 # Seed the physical fit with the compact nonlinear model on the majority of rows.
 keep=np.ones(len(xx),bool)
 for _ in range(5):
  k=np.linalg.lstsq(b[keep],score[keep],rcond=None)[0]; r=score-b@k
  idx=np.argsort(abs(r))[:int(.9*len(r))]; keep[:]=False; keep[idx]=True
 k=np.linalg.lstsq(b[keep],yy[keep],rcond=None)[0]; e=yy-b@k
 keep=np.linalg.norm(e,axis=1)<max(.002,8*np.median(np.linalg.norm(e,axis=1)))
 return np.linalg.lstsq(b[keep],yy[keep],rcond=None)[0],keep

def invert(raw,k):
 shape=raw.shape; y=raw.reshape(-1,4); z=np.linalg.solve(k[1:5].T,(y-k[0]).T).T
 for _ in range(20):
  r=basis(z)@k-y
  J=np.broadcast_to(k[1:5].T,(len(z),4,4)).copy()
  for j in range(4):
   J[:,:,j]+=3*z[:,j,None]**2*k[5+j]+z[:,prev[j],None]*k[9+j]
   J[:,:,prev[j]]+=z[:,j,None]*k[9+j]
  step=np.linalg.solve(J,r[...,None])[...,0]; z-=step/np.maximum(1,np.max(abs(step),axis=1,keepdims=True))
  if np.max(abs(r))<1e-10: break
 return z.reshape(shape),np.linalg.norm(basis(z)@k-y,axis=1)

if __name__=='__main__':
 k,good=fit(x,y); print('excluded',np.flatnonzero(~good),'coefficients',k.round(5)); print('clean forward p90',np.quantile(abs(basis(x[good])@k-y[good]),.9));
 for split in (np.arange(len(x))%5, np.digitize(np.linalg.norm(x,axis=1),np.quantile(np.linalg.norm(x,axis=1),[.2,.4,.6,.8]))):
  for g in np.unique(split):
   i=split==g; kf,ok=fit(x[~i],y[~i]); z,_=invert(y[i & good],kf); e=z-x[i & good]; print('cal fold',g,'clean load/sum max',np.max(abs(e)),np.max(abs(e.sum(1))),'retained',ok.sum())
 loads=[]
 for b in (tr,te):
  for key in ('past_channel_raw','hold_channel_raw'):
   z,r=invert(b[key],k); print(key,'max reconstruction',max(r),'load range',z.min(),z.max()); loads.append(z.sum(-1))
 past,hold=loads[:2]; xx=m._features(tr,te)[:len(hold)]; yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; labs=tr['lab_id'];
 for lab in [*np.unique(labs),None]:
  i=labs==lab if lab is not None else np.ones(len(labs),bool); j=~i if lab is not None else i; tau,th=m._fit(xx[j],past[j],hold[j],dt,t,yc[j],yr[j]); A,B=m._design(xx[i],past[i],hold[i],dt,t,tau); print('lab',lab,'tau',tau,'theta',th.round(4),'curve/rate p90',np.quantile(abs(A@th-yc[i].ravel()),.9),np.quantile(abs(B@th-yr[i]),.9))
