import pickle,sys
import numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_ep04_solver as s
d=pickle.load(open('/holdout_task/public_alpha_ep04.pkl','rb'))[0]['dataset']
tr,te,y,seen=s._prepare(d); seen=seen[:len(y)].all(axis=1)
lab=d['train']['lab_id']; stratum=d['train']['stratum']; p=np.empty_like(y)
for l in np.unique(lab):
    held=lab==l
    tau,w=s._fit(tr,y,~held & seen)
    p[held]=tr.matrix(np.log(tau))[held] @ w
    print('LOLO',l,'taus',np.round(tau,5),'w',np.round(w,5))
innovation=y[:,-1]-p[:,-1]; base=y[:,:-1]-p[:,:-1]
slope=(base@tr.times)/(tr.times@tr.times)
for flag in [True,False]:
    m=seen==flag
    print('complete',flag,'n',m.sum(),'innovation abs q50/90/max',np.round(np.quantile(abs(innovation[m]),[.5,.9,1]),6),'curve slope gap q50/90',np.round(np.quantile(abs(slope[m]),[.5,.9]),6))
    for a in (0,.1,.25,.5,.75,1):
        e=abs(base-a*innovation[:,None]*tr.times)
        slices=[e[m&(lab==l)&(stratum==st)] for l in range(4) for st in range(3) if np.sum(m&(lab==l)&(stratum==st))>=3]
        print(a,'MAE',round(e[m].mean(),6),'median',round(np.median(e[m]),6),'worst cell MAE',round(max(v.mean() for v in slices),6),'worst lab MAE',round(max(e[m&(lab==l)].mean() for l in range(4)),6))
