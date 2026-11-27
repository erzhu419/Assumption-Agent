import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr=d['train']; ch=d['chamber_verification_trials']; times=d['eval_times']; dt=d['past_dt']; b=m._load_bridge(d['calibration_load_cells']); pb=m._prepare(tr,b); pc=m._prepare(ch,b)

def fit(rows,mode):
 q=tuple(a[rows] for a in pb); lab=np.asarray(tr['lab_id'])[rows]; labs,li=np.unique(lab,return_inverse=True); y=tr['strain_curves_obs'][rows]-tr['strain_0'][rows,None]; r=tr['instant_rate_obs'][rows]
 base=m._fit(tr,pb,times,dt,mask=rows)
 if mode=='base': return base
 hot=np.eye(len(labs))[li]; contrast=hot[:,:-1]-hot[:,[-1]]
 def fun(z):
  A,B=m._design(q,times,dt,np.exp(z[:2])); drift=(q[0][:,1:]@z[5:9])*(contrast@z[9:]); return np.r_[(A@z[2:9]+times*drift[:,None]-y).ravel(),.7*(B@z[2:9]+drift-r)]
 o=least_squares(fun,np.r_[np.log(base[0]),base[1],np.zeros(contrast.shape[1])],loss='soft_l1',f_scale=.004,max_nfev=160,x_scale='jac',bounds=([np.log(.050001)]*2+[-np.inf]*(7+contrast.shape[1]),[np.log(24.9999)]*2+[np.inf]*(7+contrast.shape[1]))); order=np.argsort(o.x[:2]); return np.exp(o.x[:2][order]),np.r_[o.x[2:4][order],o.x[4:9]]

def errors(g,q,f):
 A,B=m._design(q,times,dt,f[0]); return A@f[1]-(g['strain_curves_obs']-g['strain_0'][:,None]),B@f[1]-g['instant_rate_obs']
def report(e,r,sel): return np.round([np.quantile(abs(e[sel]),.9),np.quantile(abs(r[sel]),.9),np.quantile(abs(e[sel,-3:]),.9)],5)
for mode in ['base','centered_gain']:
 f=fit(pb[-1],mode); e,r=errors(ch,pc,f); print(mode,'transfer fit',np.round(f[0],4),np.round(f[1],5),'complete',report(e,r,pc[-1]),'missing',report(e,r,~pc[-1]),'worst_lab',np.max([report(e,r,ch['lab_id']==l) for l in np.unique(ch['lab_id'])],axis=0)); e,r=errors(tr,pb,f); print(' bench train complete',report(e,r,pb[-1])); out=[]
 for l in np.unique(tr['lab_id']):
  sel=tr['lab_id']==l; f=fit(pb[-1]&~sel,mode); e,r=errors(tr,pb,f); out.append(report(e,r,sel&pb[-1])); print(' bench holdout',l,out[-1],np.round(f[0],3));
 print(' bench worst complete',np.max(out,axis=0))
