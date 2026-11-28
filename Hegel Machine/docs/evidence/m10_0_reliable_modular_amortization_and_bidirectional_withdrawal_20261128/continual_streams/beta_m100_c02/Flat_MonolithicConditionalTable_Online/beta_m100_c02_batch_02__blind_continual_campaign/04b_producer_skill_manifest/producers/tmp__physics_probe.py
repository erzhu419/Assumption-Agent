import pickle,numpy as np,itertools
from scipy.linalg import null_space
from scipy.optimize import least_squares
import model
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=model.calibrate(s); z=s['Z_nuisance']; n=len(s['source_modes']); w0=np.ones(n)*s['total_budget']/n; U=null_space(np.ones((1,n))); rob=np.repeat([s['robin_x'],s['robin_y']],2)
def fit(q,tr=None):
 a,_=model.chamber_response(s,np.asarray(q)*rob); v=np.c_[a@U,z]; rhs=y-a@w0
 if tr is None: tr=np.ones(len(y),bool)
 p=np.linalg.lstsq(v[tr],rhs[tr],rcond=None)[0]
 return v@p-rhs, w0+U@p[:n-1]
for q in [(1,1,1,1),(0,0,0,0),(.5,)*4,(1.5,)*4]:
 e,w=fit(q); print('wall',q,'unregularized RMSE',np.sqrt(np.mean(e**2)),'w',w.round(5),'per shift',np.sqrt(np.mean(e.reshape(3,-1)**2,axis=1)))
folds=np.tile(np.repeat(np.arange(8)%4,4),3)
for fold in [-1,0,1,2,3]:
 tr=folds!=fold; p=least_squares(lambda q:fit(q,tr)[0][tr],np.ones(4),bounds=(0,4),gtol=1e-12)
 e,w=fit(p.x,tr); print('free',fold,p.x.round(5),'train/test',np.sqrt(np.mean(e[tr]**2)),np.sqrt(np.mean(e[~tr]**2)) if fold>=0 else None,'baseline test', np.sqrt(np.mean(fit(np.ones(4),tr)[0][~tr]**2)) if fold>=0 else None)
print('gate',open('/tmp/hegel_continual_archive/pre_llm_gate_diagnostic.json').read())
