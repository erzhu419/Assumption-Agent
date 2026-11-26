import importlib.util, pickle, numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('solver',p+'alpha_ep01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_ep01.pkl','rb'))[0]['dataset']; a,_,y,c=m._prepare(d); t=d['eval_times']; labs=d['train']['lab_id'];
for lab in np.unique(labs):
    hold=labs==lab; fit=~hold
    tau,b=m._fit(a,y,fit&c); pred=a.matrix(np.log(tau))@b
    weights=m._rate_weights((y-pred)[fit],c[fit],t)
    print('fold',lab,'tau',tau,'weights',np.round(weights,4))
    for scheme,w in [('none',(0,0)),('fixed',(.25,1)),('adaptive',weights)]:
        r=(m._update_drift(pred,y[:,-1],c,t,w)-y)[:,:-1]; vals=[]
        for sel in (hold&c,hold&~c):
            e=abs(r[sel]); vals.append(np.round([np.sqrt(np.mean(e**2)),np.median(e),np.median(e[:,-1]),np.mean(np.minimum(e,.2))],5))
        print(scheme,vals)
