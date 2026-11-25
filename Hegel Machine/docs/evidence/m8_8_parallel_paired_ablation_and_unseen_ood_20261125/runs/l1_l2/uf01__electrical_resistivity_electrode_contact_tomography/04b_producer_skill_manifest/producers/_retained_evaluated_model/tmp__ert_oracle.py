import importlib.util, pickle
import numpy as np
from scipy.linalg import helmert

def load(path):
    spec=importlib.util.spec_from_file_location('candidate',path+'/ert_solver.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.invert_ert_anomaly

names=['reject','oracle']
fit=load('/holdout_task')
items=pickle.load(open('/holdout_task/public_ert_surveys.pkl','rb'))
rng=np.random.default_rng(380491)
rows={key:[] for key in ('middle','interior','edge','variable_jump','different_geometry')}
for q,d in enumerate(items):
    s,g=d['survey'],d['gt']; L,D,a=s['L_m'],s['D_m'],s['a0_m']; tt,rr=s['z_tx_m'],s['z_rx_m']; m,n=s['V_obs'].shape; ref=np.asarray(g['rx_contact_offset_v']); dz=rr[None,:]-tt[:,None]; R=np.hypot(L,dz)
    def ker(p,t,r,l,b):
        x,z=p
        return 1/np.sqrt((x*x+(z-t[:,None])**2+b*b)*((l-x)**2+(z-r[None,:])**2+b*b))
    A=np.column_stack((1/R.ravel(), (dz**2/R**3).ravel(), np.repeat(helmert(m).T,n,axis=0),ker(g['anomaly_xz_m'],tt,rr,L,a).ravel()))
    yy=(s['V_obs']-ref).ravel(); mask=np.ones(len(yy),bool)
    for _ in range(3):
        c=np.linalg.lstsq(A[mask],yy[mask],rcond=None)[0]; res=yy-A@c; mask=np.abs(res)<.03
    sig=np.std(res[mask])*np.sqrt(mask.sum()/(mask.sum()-A.shape[1])); txref=helmert(m).T@c[2:-1]
    for mode in rows:
        for rep in range(24):
            u,v=rng.uniform(.15,.85,2); l,d0,b,t,r=L,D,a,tt,rr
            if mode=='middle': u,v=rng.uniform(.43,.57),rng.uniform(.28,.72)
            if mode=='edge':
                axis=rep%2; f=rng.uniform(0,.06); f=f if (rep//2)%2 else 1-f; u,v=(f,v) if axis==0 else (u,f)
            if mode=='different_geometry':
                l=L*rng.uniform(.85,1.15); d0=D*rng.uniform(.85,1.15); b=a*rng.uniform(.75,1.25); t=tt*(d0/D)+rng.uniform(-1,1,m); r=rr*(d0/D)+rng.uniform(-1,1,n)
            p=np.array([u*l,v*d0]); dz=r[None,:]-t[:,None]; R=np.hypot(l,dz)
            y=c[0]/R+c[1]*dz**2/R**3+txref[:,None]+ref+c[-1]*ker(p,t,r,l,b)+rng.normal(0,sig,(m,n))
            clean=y.copy(); ix=rng.choice(y.size,10,replace=False); y.flat[ix]+=rng.choice([-1,1],10)*(.06+rng.normal(0,.003 if mode=='variable_jump' else 0,10))
            s0=dict(s,L_m=l,D_m=d0,a0_m=b,z_tx_m=t,z_rx_m=r,V_obs=y)
            sample=[]
            for survey in (s0, dict(s0,V_obs=clean)):
                ans=fit(survey); sample.append([np.linalg.norm(np.asarray(ans['anomaly_xz_m'])-p), 1000*np.sqrt(np.mean((np.asarray(ans['rx_contact_offset_v'])-ref)**2))])
            rows[mode].append(sample)
for mode,ss in rows.items():
    mat=np.array(ss)
    print(mode)
    for i,name in enumerate(names):
        p,r=mat[:,i].T
        print(name,'pos mean/p90/max, pos misses, rx max mV:',np.round([p.mean(),np.quantile(p,.9),p.max(),(p>2).sum(),r.max()],4)); print('  same-draw misses after exact spike removal:',np.sum((mat[:,0,0]>2)&(p>2)))
