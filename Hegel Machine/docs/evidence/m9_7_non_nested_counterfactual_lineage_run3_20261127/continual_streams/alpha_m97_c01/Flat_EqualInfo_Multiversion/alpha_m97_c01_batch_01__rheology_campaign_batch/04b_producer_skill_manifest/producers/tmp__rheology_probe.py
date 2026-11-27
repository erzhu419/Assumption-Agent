import importlib.util, pickle
import numpy as np
from rheology_fit import basis, fit
p='/tmp/hegel_continual_archive/alpha_m97_c01_s00_solver.py'; s=importlib.util.spec_from_file_location('previous',p); old=importlib.util.module_from_spec(s); s.loader.exec_module(old)
d=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]['dataset']
past,hold,x=old._prepare(d); x=x[:,1:]; tr=d['train']; n=len(tr['lab_id']); dt=d['past_dt']; t=d['eval_times']; yc=tr['strain_curves_obs']-tr['strain_0'][:,None]; yr=tr['instant_rate_obs']; past,hold,x=past[:n],hold[:n],x[:n]; D=(np.maximum(abs(past)-d['crit_stress'],0)**2)@dt; act=np.tanh(D); l=tr['lab_id']
for name,mode in [('active-only',D>0),('all',np.ones(n,bool)),('linear',np.zeros(n,bool))]:
    errs=[]; ps=[]
    for i in range(4):
        m=l!=i; q=~m
        p=fit(dt,t,past[m],hold[m],x[m],act[m],mode[m],yc[m],yr[m]); ps.append(p)
        c,r,*_=basis(p[:2],dt,t,past[q],hold[q],x[q],act[q],mode[q]); ec=c@p[2:]-yc[q]; er=r@p[2:]-yr[q]
        for a in (False,True):
            b=(D[q]>0)==a
            errs.append([i,int(a),round(np.sqrt(np.mean(ec[b]**2)),5),round(np.median(abs(ec[b])),5),round(np.sqrt(np.mean(er[b]**2)),5),round(np.median(abs(er[b])),5)])
    print(name,'lab active curveRMSE curveMedAE rateRMSE rateMedAE:',errs,'tau:',np.round(np.exp(np.array(ps)[:,:2]),4).tolist(),flush=True)
