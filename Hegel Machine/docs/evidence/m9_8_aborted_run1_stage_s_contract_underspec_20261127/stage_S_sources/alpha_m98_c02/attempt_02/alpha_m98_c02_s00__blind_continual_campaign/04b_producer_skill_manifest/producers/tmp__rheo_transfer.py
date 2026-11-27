import sys, pickle, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import batch_solver as m
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr=d['train']; ch=d['chamber_verification_trials']; te=d['test']; ca=d['calibration_load_cells']; t=d['eval_times']; dt=d['past_dt']; b=m._load_bridge(ca)
s=ca['single_channel_sweeps']; print('sweeps:',{k:(np.shape(v),np.asarray(v).ravel()[:8]) for k,v in s.items()}); print('bridge total map:',b)
for name,g in [('bench',tr),('pilot',ch),('test',te)]:
 x=m._prepare(g,b)[0][:,1:]; print(name,'feature medians/std',np.median(x,axis=0),x.std(0),'telemetry',[(round(float(a),5),round(float(z),5)) for a,z in zip(np.mean(g.get('aux_chamber_telemetry', np.empty((len(x),0))),0),np.std(g.get('aux_chamber_telemetry', np.empty((len(x),0))),0))]);
 for l in np.unique(g['lab_id']):
  mask=g['lab_id']==l; rc=g['ref_coupons'][mask]; weights=np.array([.174,-.146,.118,-.149]); print(l,mask.sum(),np.round([np.median(rc.mean(0)),np.median(rc.std(0)),np.mean(x[mask]@weights),np.std(x[mask]@weights)],4))
pb=m._prepare(tr,b); pc=m._prepare(ch,b); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; yc=ch['strain_curves_obs']-ch['strain_0'][:,None]; rc=ch['instant_rate_obs'];
def stat(e,r,keep):return np.round([np.quantile(abs(e[keep]),.9),np.quantile(abs(r[keep]),.9)],5)
def solve(rows, mode, scale=.012):
 q=tuple(a[rows] for a in pb); lab=tr['lab_id'][rows]; ones=lab[:,None]==np.unique(lab)[None,:]; tel=tr['aux_chamber_telemetry'][rows]; ref=None
 if mode in ('lab','labstress'): ref=np.c_[ones[:,1:],ones[:,1:]*q[0][:,1:]@np.zeros((4,4))] if False else np.c_[ones[:,1:],ones[:,1:]*(q[0][:,1:]@np.array([.174,-.146,.118,-.149]))[:,None]]
 if mode=='labstress': ref=np.c_[ref,q[2]-3.35]
 if mode=='stress': ref=(q[2]-3.35)[:,None]
 if mode=='telemetry': ref=np.c_[tel,tel**2-1]
 def design(z):
  A,B=m._design(q,t,dt,np.exp(z));
  if ref is not None: A=np.concatenate([A,t[None,:,None]*ref[:,None,:]],axis=2);B=np.c_[B,ref]
  return np.r_[A.reshape(-1,A.shape[-1]),.7*B]
 yy=np.r_[y[rows].ravel(),.7*r[rows]]; start=np.log([.68,4]); w=np.linalg.lstsq(design(start),yy,rcond=None)[0]; fit=least_squares(lambda z:design(z[:2])@z[2:]-yy,np.r_[start,w],loss='soft_l1',f_scale=scale,max_nfev=250,bounds=([np.log(.05001)]*2+[-np.inf]*len(w),[np.log(24.999)]*2+[np.inf]*len(w))); return np.exp(fit.x[:2]),fit.x[2:]
for mode in ['base','lab','stress','labstress','telemetry']:
 taus,w=solve(pb[-1],mode); A,B=m._design(pc,t,dt,taus); e=A@w[:7]-yc; er=B@w[:7]-rc; pf=pc[-1]; print(mode,'fit',np.round(taus,4),np.round(w,5),'zero-shot pilot p90 complete/missing',stat(e,er,pf),stat(e,er,~pf));
 for l in np.unique(ch['lab_id']): print('  pilot lab',l,stat(e,er,ch['lab_id']==l))
