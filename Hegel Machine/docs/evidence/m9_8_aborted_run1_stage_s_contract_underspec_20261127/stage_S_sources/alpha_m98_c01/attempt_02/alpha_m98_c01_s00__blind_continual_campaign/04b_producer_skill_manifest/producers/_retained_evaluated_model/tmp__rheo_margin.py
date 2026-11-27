import importlib.util,pickle
import numpy as np
p=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); s=importlib.util.module_from_spec(p); p.loader.exec_module(s)
d=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr,te,ch=(d[k] for k in ('train','test','chamber_verification_trials')); t=d['eval_times']; z=np.r_[t,1.]; age=np.r_[np.cumsum(d['past_dt'][::-1])[::-1],0.]; bridge=s._bridge(d['calibration_load_cells']); ref=s._reference([tr,te,ch]); x,fill=s._standardize(tr,ref); tau,b=s._fit(s._loads(tr,bridge),x,s._target(tr),age,t); X,_=s._standardize(ch,ref,fill); g=ch['lab_id']; E=s._target(ch)-s._design(s._loads(ch,bridge),X,tau,age,t)@b; y=s._drift_residual(E,t); score=X[:,1:]@b[3:]; spread=np.array([ref[k][1] for k in g]); hold=s._loads(ch,bridge)[1]; L=np.unique(g); I=(g[:,None]==L).astype(float)
print('lab pilot counts and reference spreads:',[(int(l),int(sum(g==l)),round(ref[l][1],4)) for l in L]); print('calibration sweep fields:', list(d['calibration_load_cells']['single_channel_sweeps'])); print('largest median time/rate residual after independent slope removal:',np.median(abs(E-y[:,None]*z),axis=0).max())
local=np.c_[I,I*score[:,None]]
def ridge(F,train,test,pen):
    return F[test]@np.linalg.lstsq(np.r_[F[train],np.diag(np.broadcast_to(pen,F.shape[1]))],np.r_[y[train],np.zeros(F.shape[1])],rcond=None)[0]
models={'baseline':(local,.08),'gain only':(I*score[:,None],.08),'offset with stronger gain shrinkage':(local,np.r_[np.full(len(L),.08),np.full(len(L),.35)]),'local gain plus shared saturation':(np.c_[local,score**3/(1+score**2)],np.r_[np.full(2*len(L),.08),.35]),'local gain plus pairwise contrast':(np.c_[local,X[:,1]-X[:,3],X[:,2]-X[:,4]],np.r_[np.full(2*len(L),.08),np.full(2,.35)]),'reference-spread transfer':(np.c_[np.ones(len(g)),score,score*(1/spread-1),local],np.r_[.20,.30,.30,np.full(2*len(L),.22)]),'shared + local gain':(np.c_[np.ones(len(g)),score,local],np.r_[.20,.30,np.full(2*len(L),.22)]),'local gain plus hold load':(np.c_[local,hold-hold.mean()],np.r_[np.full(2*len(L),.08),.35])}
def measure(F,pen,group):
    er=E.copy()
    for k in np.unique(group):
        j=group==k; er[j]-=ridge(F,~j,j,pen)[:,None]*z
    v=np.array([[np.mean(abs(er[m,:-1])),np.mean(abs(er[m,-1])),np.mean(abs(er[m,-2]))] for m in [np.ones(len(g),bool),*[g==l for l in L]]]); return np.r_[v[0], v[1:].max(axis=0)].round(4).tolist(),v[1:,:2].round(4).tolist()
for k,(F,pen) in models.items():
    print(k,[(n,measure(F,pen,c)[0]) for n,c in [('coupon',np.arange(len(g))),('stratum',ch['stratum']),('lab',g)]], 'by-lab stratum',measure(F,pen,ch['stratum'])[1])
