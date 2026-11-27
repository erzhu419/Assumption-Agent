import runpy
import numpy as np
from scipy.optimize import least_squares
z = runpy.run_path('/tmp/rheology_final_witness.py', run_name='setup')
m,d,b,dt,t,tr,p,take = [z[k] for k in ('m','d','b','dt','t','tr','p','take')]
c=d['calibration_load_cells']; R=c['bridge_raw']; L=c['true_channel_loads'].sum(axis=1)
def basis(r,kind):
    parts=[np.ones(len(r)),*r.T]
    if kind in ('quadratic','rational'):
        parts.extend(r[:,i]*r[:,j] for i in range(4) for j in range(i,4))
    if kind=='cubic':
        parts.extend(r[:,i]*r[:,j]*r[:,k] for i in range(4) for j in range(i,4) for k in range(j,4))
    return np.column_stack(parts)
for kind in ('affine','quadratic','cubic','rational'):
    scores=[]
    for seed in range(4):
        ix=np.random.default_rng(seed).permutation(len(R)); a,j=ix[:60],ix[60:]
        X=basis(R,kind); y=L
        if kind=='rational':
            Q=np.column_stack([R[:,i]-R[:,k] for i in range(4) for k in range(i+1,4)])**2
            U=np.column_stack([X,-y[:,None]*Q]); co=np.linalg.lstsq(U[a],y[a],rcond=None)[0]
            pred=(X[j]@co[:X.shape[1]])/(1+Q[j]@co[X.shape[1]:])
        else:
            pred=X[j]@np.linalg.lstsq(X[a],y[a],rcond=None)[0]
        scores.append(np.sqrt(np.mean((pred-y[j])**2)))
    print('bridge held-out axial RMSE',kind,'median/worst',np.median(scores),max(scores))

def prepare(g): return m._prepare(g,b)
def summary(rows): return np.round(np.max(rows,axis=0),6)
for g in (tr,p):
    scores={k:[] for k in ('base','load_rational','pairwise_contrast','telemetry_pooled','telemetry_centered')}
    for lab in np.unique(g['lab_id']):
        mask=g['lab_id']==lab; old,new=take(g,~mask),take(g,mask)
        a,v=prepare(old),prepare(new)
        tau,theta=m._estimate(a,None,dt,t) if g is tr else m._estimate(prepare(tr),a,dt,t)
        F,G=[m._design(x,tau,dt,t) for x in (a,v)]
        residual=a[4]-F@theta; target=v[4]-G@theta
        time=np.r_[1.,t]; scales=np.r_[.012,np.full(len(t),.027)] if g is tr else np.r_[.0025,np.full(len(t),.005)]
        def add(k,e): scores[k].append([np.median(abs(e[:,0])),np.median(abs(e[:,1:])),np.median(abs(e[:,-1]))])
        add('base',target)
        for kind in ('load_rational','pairwise_contrast','telemetry_pooled','telemetry_centered'):
            if kind.startswith('telemetry') and 'aux_chamber_telemetry' not in old: continue
            def feature(raw,x,center=None):
                if kind=='load_rational': return (1/(1+(x[1]/4)**2))[:,None]-.5
                if kind=='pairwise_contrast':
                    h=raw['hold_channel_raw']@b[1:]+b[0]
                    return np.column_stack([h[:,i]-h[:,j] for i in range(3) for j in range(i+1,4)])
                q=raw['aux_chamber_telemetry'].copy()
                if kind=='telemetry_centered':
                    for lab_id in np.unique(raw['lab_id']):
                        j=raw['lab_id']==lab_id; q[j]-=q[j].mean(axis=0)
                return q
            X,Y=feature(old,a),feature(new,v)
            A=X[:,None,:]*time[None,:,None]; B=Y[:,None,:]*time[None,:,None]
            w=np.sqrt(a[5])[:,None]/scales
            f=least_squares(lambda co: ((A@co-residual)*w).ravel(),np.zeros(X.shape[1]),loss='cauchy')
            add(kind,target-B@f.x)
    for kind,r in scores.items():
        if r: print('worst held-out lab rate/strain/terminal', 'bench' if g is tr else 'pilot', kind, summary(r))
