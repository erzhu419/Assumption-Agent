import importlib.util,pickle,numpy as np
P='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',P+'alpha_m97_c03_batch_04_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
d=pickle.load(open(P+'public_alpha_m97_c03_batch_04.pkl','rb'))[0]['dataset']; tr=d['train']; lab=tr['lab_id']; st=tr['stratum']; n=len(lab)
p,h,x,o,dt,t=m._prepare(d); D=m._damage(d,p,dt)[:n]; p,h,x=p[:n],h[:n],x[:n]
y=np.column_stack((tr['strain_curves_obs']-o[:n,None], tr['instant_rate_obs']))
def args(i): return p[i],h[i],x[i],dt,t,D[i]
def metrics(e):
 return np.array([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),np.median(np.abs(e[:,:-1])),np.median(np.abs(e[:,-1])),np.max(np.abs(e[:,:-1])),np.max(np.abs(e[:,-1]))])

def scale(e):
    return np.maximum(1e-6, 1.4826*np.median(np.abs(e-np.median(e))))

def fit(i, mode):
    a=args(i); yy=y[i]
    pp,w=m._fit(a,yy,mode)
    def resid(): return m._basis(pp,*a,terminal_mode=mode)@w-yy
    e=resid(); sc=np.array([scale(e[:,:-1]),scale(e[:,-1])]); sc=np.r_[np.repeat(sc[0],len(t)),sc[1]]
    for k in range(12):
        mult=1/(sc*np.sqrt(1+(e/(sc*(1.5 if k<3 else 2.385)))**2))
        pp,w=m._fit(a,yy,mode,mult); e=resid()
        if k in (2,5,8):
            sc=np.r_[np.repeat(scale(e[:,:-1]),len(t)),scale(e[:,-1])]
    return pp,w,sc

for mode in ('active','all_log'):
    pp,w,sc=fit(slice(None),mode); print(mode,'taus',np.exp(pp),'weights',w,'scales',sc[[0,-1]])
    for name,g in [('lab',lab),('stratum',st)]:
        for k in np.unique(g):
            v=g==k; pp,w,sc=fit(~v,mode); e=m._basis(pp,*args(v),terminal_mode=mode)@w-y[v]
            for act in (False,True):
                q=np.abs(e[(D[v]>0)==act]); a=q[:,:-1]; b=q[:,-1]; print(name,k,int(act),'curve/rate MAE',np.mean(a),np.mean(b),'median',np.median(a),np.median(b),'p90',np.quantile(a,.9),np.quantile(b,.9))
