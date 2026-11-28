import importlib.util,pickle,numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
c=d['calibration_load_cells']; raw=c['bridge_raw']; loads=c['true_channel_loads']; sweeps=c['single_channel_sweeps']
# Held-out calibration compares omitted spatial interactions and rational response with the deployed basis.
from scipy.optimize import least_squares
for name in ('archived','interactions','rational'):
    errors=[]
    for fold in range(5):
        v=np.arange(len(raw))%5==fold
        if name=='archived':
            p=m._bridge(dict(c,bridge_raw=raw[~v],true_channel_loads=loads[~v])); pred=m._forward(loads[v],p)
        else:
            pred=np.empty_like(raw[v])
            for j,k in enumerate([2,3,0,1]):
                a=loads[~v]; y=raw[~v,j]; keep=abs(y-np.median(y))<6; a,y=a[keep],y[keep]
                sv=sweeps['lab_id']==j; isolated=np.zeros((sum(sv),4)); isolated[:,j]=sweeps['true_stress'][sv]; a=np.r_[a,isolated]; y=np.r_[y,sweeps['cmd_stress'][sv]]
                def basis(x,q):
                    z,w=x[:,j],x[:,k]; f,g=q[:2]
                    b=np.column_stack([np.ones(len(x)),z,np.sin(f*z),w,-2*np.sin(g*w/2)**2]); others=[i for i in range(4) if i not in (j,k)]
                    if name=='rational': return b/(1+q[2]*(z-w)**2)[:,None]
                    return np.column_stack([b,x[:,others],z*w,(z-w)**2,(z+w)**2,z*w*x[:,others[0]],z*w*x[:,others[1]]])
                q0=[2.5,2.]+([.01] if name=='rational' else []); lo=[1.,.7]+([0.] if name=='rational' else []); hi=[4.,3.5]+([10.] if name=='rational' else [])
                def lin(q):
                    b=basis(a,q); return b,np.linalg.lstsq(b,y,rcond=None)[0]
                q=least_squares(lambda q: lin(q)[0]@lin(q)[1]-y,q0,bounds=(lo,hi),gtol=1e-11).x
                pred[:,j]=basis(loads[v],q)@lin(q)[1]
        errors.append(pred-raw[v])
    r=np.concatenate(errors); print('held-out bridge',name,'worst channel inlier RMS',max(np.sqrt(np.mean(z[abs(z)<1]**2)) for z in r.T),'gross faults',np.sum(abs(r)>1,axis=0))
a,b=d['train'],d['test']; n=len(a['lab_id']); x=m._features(a,b); dt,t=d['past_dt'],d['eval_times']; bridge=m._bridge(c); past,hold=m._loads(a,bridge); y=a['strain_curves_obs']-a['strain_0'][:,None]; r=a['instant_rate_obs']; labels=np.c_[a['lab_id'],a['stratum']]
for lab in np.unique(labels[:,0]):
    v=labels[:,0]==lab; z=~v; tau,theta=m._fit(past[z],hold[z],x[:n][z],dt,t,y[z],r[z]); cv,cr=m._design(tau,past[v],hold[v],x[:n][v],dt,t); ec=cv@theta-y[v]; er=cr@theta-r[v]; print('held-out lab / stratum signed median residuals (strain, rate)',lab,[(int(k),round(np.median(ec[labels[v,1]==k]),6),round(np.median(er[labels[v,1]==k]),6)) for k in np.unique(labels[v,1])]); print('  held-out lab signed median (strain, rate)',np.median(ec),np.median(er))
