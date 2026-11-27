import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from route_m99 import *
from scipy.optimize import least_squares
x=X['diag']; D=th[2]+x@th[3:]; F=np.c_[np.ones(n),D]; truth=Y+(h[:,None]*th[:2])@T[:,:2].T
print('intrinsic times/weights',ta,th)
print('route full intercept and gain',np.linalg.lstsq(F,Z,rcond=None)[0]); print('expected tau-scaled gains',np.linalg.lstsq(F,Z,rcond=None)[0][1]/np.r_[ta,1.])
for labx in [None,*np.unique(lab)]:
 keep=np.ones(n,bool) if labx is None else lab!=labx; chk=keep if labx is None else ~keep
 for kind in ('route','intercept','tied','omit drift'):
  if kind=='route': A=D[:,None,None]*T[None,:,:]
  elif kind=='intercept': A=F[:,None,:,None]*T[None,:,None,:]; A=A.reshape(n,len(T),-1)
  elif kind=='tied': A=D[:,None,None]*np.column_stack((T[:,:2]@ta,T[:,2]))[None,:,:]
  else: A=D[:,None,None]*T[None,:,:2]
  opt=least_squares(lambda v:(A[keep]@v-Y[keep]).ravel(),np.zeros(A.shape[-1]),loss='soft_l1',f_scale=.004)
  e=(A@opt.x-Y)[chk]; print(labx,kind,np.round(opt.x,6),np.round(np.sqrt([np.mean(e[:,:-1]**2),np.mean(e[:,-1]**2)]),6))
print('batch entries',[(list(item), item['campaign_diagnostics'] is None) for item in pickle.load(open('/holdout_task/public_batch.pkl','rb'))]); print('top-level dataset keys',list(d)); print('archive metadata',[(r['completed_campaign_short_name'],list(r)) for r in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))])
