import pickle
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from hegel_wormhole_skills.autogenous_prim_robust_affine_projection import robust_residual_weights
from hegel_wormhole_skills.autogenous_prim_ridge_profiled_matching import fit_ridge_nuisance


def design(r,c,kind,tau):
    x=(r-7.5)/7.5; z=(c-11.5)/11.5
    dr=np.minimum(r,15-r); dc=np.minimum(c,23-c)
    a,b=np.exp(-dr/tau),np.exp(-dc/tau)
    cols=[np.ones_like(x),x,z]
    if kind=='min': cols += [np.maximum(a,b)]
    if kind=='sum': cols += [a+b]
    if kind=='union': cols += [a+b-a*b]
    if kind=='separate': cols += [a,b]
    if kind=='corner': cols += [a+b,a*b]
    if kind=='quad': cols += [x*x,z*z,x*z]
    return np.column_stack(cols)


def fit(A,y,ridge=1e-6):
    w=np.ones(len(y))
    for _ in range(7):
        sw=np.sqrt(w); beta,*_=fit_ridge_nuisance(A*sw[:,None],y*sw,ridge=ridge)
        w=robust_residual_weights(y-A@beta,0.025)
    return beta


def arrays(plates):
    return [(np.array([a['row'] for a in t['wells']]),np.array([a['col'] for a in t['wells']]),np.array([a['well_type'] for a in t['wells']]),np.array([a['fluorescence'] for a in t['wells']]),np.array([a['compound_id'] for a in t['wells']],dtype=object),np.array([a['log10_conc_uM'] for a in t['wells']],dtype=float)) for t in plates]


def assess(data,kind,tau,loo=True):
    cv=[]; comp=[]; train=[]
    for r,c,k,y,id,dose in data:
        bg=np.median(y[k=='POS_KILL']); amp=np.median(y[k=='DMSO'])-bg
        f=(y-bg)/amp; A=design(r,c,kind,tau); ix=np.flatnonzero(k=='DMSO')
        if loo:
            for j in ix:
                tr=ix[ix!=j]; cv.append(f[j]-A[j]@fit(A[tr],f[tr]))
        beta=fit(A[ix],f[ix]); train.extend(f[ix]-A[ix]@beta)
        sel=k=='COMPOUND'; comp.extend(zip(id[sel],dose[sel],f[sel],A[sel]@beta))
    result={}
    for name in sorted({x[0] for x in comp}):
        u,v,g=np.array([x[1:] for x in comp if x[0]==name]).T
        res=least_squares(lambda p:g*expit(np.log(10)*np.exp(p[1])*(p[0]-u))-v,[u[np.argmin(np.abs(v/g-.5))],0.],bounds=([u.min()-4.,-3.],[u.max()+4.,2.]),loss='cauchy',f_scale=.035)
        result[name]=[res.x[0],np.exp(res.x[1])]
    return np.array(cv),np.array(train),result

if __name__=='__main__':
    p=pickle.load(open('/holdout_task/public_screen.pkl','rb')); data=arrays(p['plates'])
    print('DMSO locations:',list(zip(data[0][0][data[0][2]=='DMSO'],data[0][1][data[0][2]=='DMSO'])))
    for kind in ['flat','quad','min','sum','union','separate','corner']:
        for tau in ([1.] if kind in ('flat','quad') else [0.8,1.3,2.,3.,4.5,7.]):
            cv,tr,out=assess(data,kind,tau)
            e=np.abs([np.array(out[k])-list(p['gt'][k].values()) for k in out]); w=np.argmax(e,axis=0)
            print(f'{kind:9s} {tau:4.1f} controls CV rms/max={np.sqrt(np.mean(cv**2)):.4f}/{max(abs(cv)):.4f} train rms={np.std(tr):.4f}  IC/slope max={e.max(axis=0).round(4)} {w}')
