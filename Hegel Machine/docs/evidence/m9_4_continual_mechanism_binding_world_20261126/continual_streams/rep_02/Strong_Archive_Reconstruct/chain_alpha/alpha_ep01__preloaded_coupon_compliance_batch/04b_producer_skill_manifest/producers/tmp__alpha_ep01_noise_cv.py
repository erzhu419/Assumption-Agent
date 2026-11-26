import sys,pickle,inspect,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_ep01_solver as s
src=inspect.getsource(s._fit)
needle='return np.r_[np.full(r.shape[1] - 1, _mad(r[:, :-1].ravel())), _mad(r[:, -1])]'
assert needle in src
fits={'pooled':s._fit}
for name,expr in [('horizon','_mad(r)'),('shrink','np.sqrt(_mad(r) * np.r_[np.full(r.shape[1] - 1, _mad(r[:, :-1].ravel())), _mad(r[:, -1])])')]:
    ns=dict(vars(s)); exec(src.replace(needle,'return '+expr),ns); fits[name]=ns['_fit']
d=pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb'))[0]['dataset']; a,b,y,complete=s._prepare(d); c=complete[:len(y)]; t=d['eval_times']; tr=d['train']; l=tr['lab_id']; g=np.where(c,.25,1.); denom=np.maximum(np.abs(a.hold),.5)
pre=np.linalg.norm(tr['initial_memory_states'],axis=1)
slices=[l==j for j in np.unique(l)]+[tr['stratum']==j for j in np.unique(tr['stratum'])]+[c,~c,np.abs(a.hold)<.5,pre>=np.quantile(pre,.9)]
clean=None
for name,fit in fits.items():
    tau_all=[]; residual=np.empty_like(y)
    for j in np.unique(l):
        tau,w=fit(a,y,(l!=j)&c); tau_all.append(tau); residual[l==j]=(y-a.matrix(np.log(tau))@w)[l==j]
    e=residual[:,:-1]-g[:,None]*residual[:,-1,None]*t
    if clean is None:
        # The same isolated gross-error flags apply to every candidate.
        clean=np.abs(e)<.4
        print('gross errors by horizon', (~clean).sum(axis=0))
    def score(m):
        ae=np.abs(e[m]); ok=clean[m]; sec=ae[:,-1]/denom[m]; return [np.sqrt(np.mean(ae**2)),np.median(ae),np.quantile(ae,.9),np.sqrt(np.mean(ae[ok]**2)),np.quantile(sec[ok[:,-1]],.9)]
    print(name,'worst [allRMSE,medAE,p90,nonspikeRMSE,nonspikeSecantP90]',np.round(np.max([score(m) for m in slices],axis=0),5),'complete',np.round(score(c),5),'missing',np.round(score(~c),5),'tau range',np.round(np.ptp(tau_all,axis=0),5))
