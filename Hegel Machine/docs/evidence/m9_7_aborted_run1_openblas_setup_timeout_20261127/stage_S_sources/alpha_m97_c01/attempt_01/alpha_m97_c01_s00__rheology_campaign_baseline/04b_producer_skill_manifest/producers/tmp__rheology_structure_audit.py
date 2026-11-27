import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c01_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root+'public_alpha_m97_c01_s00.pkl','rb'))[0]['dataset']
tr = d['train']
dt,t = d['past_dt'],d['eval_times']
all_data,n = m._prepare(d)
past,hold,x = (v[:n] for v in all_data)
print('regime: banding',d['shear_banding_index'],'max calibrated |past|',np.max(np.abs(all_data[0])),'critical',d['crit_stress'])
y = tr['strain_curves_obs']-tr['strain_0'][:,None]
r = tr['instant_rate_obs']
lab = tr['lab_id']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
# Constitutive alternatives in the same coordinates; each correction vanishes at t=0.
def make_basis(logs):
    a,_,gap = m._design(logs,(past,hold,x),dt,t)
    taus = np.exp(logs)
    expected_gap = []
    for tau in taus:
        elapsed_after = np.cumsum(dt[::-1])[::-1]-dt
        expected_gap.append(hold - past @ (np.exp(-elapsed_after/tau)*(-np.expm1(-dt/tau))))
    np.testing.assert_allclose(gap, np.column_stack(expected_gap), atol=2e-14, rtol=0)
    s=(gap[:,0]-gap[:,1]); q=(gap[:,0]+gap[:,1]); h=hold
    terms = [s*q,h*s,h*q, s*q*h, h**2, h/(1+s*s)-h]
    terms += [x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]
    c = np.column_stack(terms)
    rate = np.column_stack((gap/taus, np.ones(len(x)), x))
    return a,rate,a,t[None,:,None]*c[:,None,:],c

def scales(e):
    return max(1e-7, 1.4826*np.median(np.abs(e-np.median(e))))

def refit(a,b,y0,r0,keep,theta):
    def residual(p):
        return np.r_[((a[keep]@p-y[keep])/y0).ravel(),(b[keep]@p-r[keep])/r0]
    return least_squares(residual,theta,loss='cauchy',x_scale='jac',max_nfev=100).x

scores={}
for l in np.unique(lab):
    keep = lab!=l
    te = lab==l
    logs,w = m._fit(tuple(v[keep] for v in (past,hold,x)),dt,t,y[keep],r[keep])
    a,b,_,c,cr = make_basis(logs)
    clean = keep & complete
    sc,sr = scales(a[clean]@w-y[clean]),scales(b[clean]@w-r[clean])
    ac,bc = np.concatenate((a,c),axis=-1),np.column_stack((b,cr))
    # Only training labs get fixed-effect slopes; the withheld lab receives the physical slope.
    u = np.unique(lab[keep]); dummies = (lab[:,None]==u[None,1:]).astype(float)
    an = np.concatenate((a,t[None,:,None]*dummies[:,None,:]),axis=-1)
    bn = np.column_stack((b,dummies))
    wn = refit(an,bn,sc,sr,keep,np.r_[w,np.zeros(dummies.shape[1])])
    wc = refit(ac,bc,sc,sr,keep,np.r_[w,np.zeros(cr.shape[1])])
    signed = np.median((a[te]@w-y[te])[complete[te]],axis=0)
    print('lab',l,'complete-coupon median residual first,last:',np.round(signed[[0,-1]],6),'training nuisance slope range:',np.round(np.ptp(np.r_[0.,wn[7:]]),6))
    for label,A,B,p in [('baseline',a,b,w),('nonlinear',ac,bc,wc),('lab_nuisance',a,b,wn[:7])]:
        e = np.abs(A[te]@p-y[te]); er = np.abs(B[te]@p-r[te]); ok=complete[te]
        metrics=[np.median(e,axis=0).max(),np.quantile(er,.9),np.median(e[ok],axis=0).max(),np.quantile(er[ok],.9)]
        scores.setdefault(label,[]).append(metrics)
        print(label,np.round(metrics,6))
for label,rows in scores.items():
    print('WORST',label,np.round(np.max(rows,axis=0),6))
