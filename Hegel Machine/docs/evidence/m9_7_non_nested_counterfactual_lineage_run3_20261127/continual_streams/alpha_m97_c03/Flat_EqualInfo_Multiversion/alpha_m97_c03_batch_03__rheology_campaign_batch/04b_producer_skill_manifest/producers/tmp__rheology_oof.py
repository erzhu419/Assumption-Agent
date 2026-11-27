import importlib.util, pickle
import numpy as np
p='/holdout_task/alpha_m97_c03_batch_03_solver.py'
s=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_alpha_m97_c03_batch_03.pkl','rb'))[0]
d=item['dataset']; tr=d['train']; lab=tr['lab_id']; n=len(lab)
p,h,x,o,dt,t=m._prepare(d)
y=np.column_stack((tr['strain_curves_obs']-o[:n,None],tr['instant_rate_obs']))
def args(mask): return p[:n][mask],h[:n][mask],x[:n][mask],dt,t
for mode in ('cauchy','soft_l1'):
    folds=[]
    for l in np.unique(lab):
        tau,w=m._fit(args(lab!=l),y[lab!=l],mode)
        a,_,_=m._basis(np.log(tau),*args(lab==l)); e=abs(a@w-y[lab==l]); folds.append(e)
        print(mode,'lab',l,'taus',np.round(tau,4),'rw',np.round(w[7:],4),'abs median/q90 strain',np.round(np.quantile(e[:,:-1],[.5,.9]),4),'rate',np.round(np.quantile(e[:,-1],[.5,.9]),4),'worst horizon median',round(np.median(e[:,:-1],axis=0).max(),4))
    print(mode,'worst lab medians',max(np.median(e[:,:-1]) for e in folds),max(np.median(e[:,-1]) for e in folds))
tau,w=m._fit(args(np.ones(n,bool)),y); print('full fit taus/weights',tau,w)
a,da,ds=m._basis(np.log(tau),*args(np.ones(n,bool))); logs=np.log(tau); eps=1e-6
for j in range(2):
    delta=np.eye(2)[j]*eps
    fd=((m._basis(logs+delta,*args(np.ones(n,bool)))[0]-m._basis(logs-delta,*args(np.ones(n,bool)))[0])@w)/(2*eps)
    err=np.max(abs(fd-da[:,:,j]*w[j]-ds[:,:,j]*w[-1])); print('logtau derivative max error',j,err); assert err<1e-7
print('public truth field names',list(item['gt']) if isinstance(item['gt'],dict) else type(item['gt']).__name__)
